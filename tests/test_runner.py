"""The async runner end to end, with fake feature and judge providers (DeepEval runs for real)."""

from datetime import UTC, datetime

import pytest

from regress.dataset import GoldenCase, GoldenDataset, case_id
from regress.eval.runner import run_eval
from regress.prompts import load_prompt
from regress.providers.base import FatalProviderError, LLMRequest
from tests.conftest import FakeProvider, classification_json, judge_json

EMAILS = {
    "I was charged twice, refund one please": "billing",
    "The app crashes every time I open it": "technical",
    "I cannot log in after resetting my password": "account",
}


@pytest.fixture
def dataset() -> GoldenDataset:
    cases = [
        GoldenCase(id=case_id(e), input_email=e, expected_category=c, ideal_summary="Customer S.",
                   expected_difficulty="easy")  # fmt: skip
        for e, c in EMAILS.items()
    ]
    return GoldenDataset(dataset_version=3, content_hash="h", built_at=datetime.now(UTC), source="t",
                         case_count=len(cases), cases=sorted(cases, key=lambda c: c.id))  # fmt: skip


def feature(wrong: set[str] = frozenset(), broken: set[str] = frozenset()) -> FakeProvider:
    def respond(req: LLMRequest) -> str:
        email = req.messages[-1].content
        if email in broken:
            raise TimeoutError("simulated timeout after retries")
        category = EMAILS[email]
        return classification_json("general" if email in wrong else category)

    return FakeProvider(respond)


def judge(score: int = 4) -> FakeProvider:
    return FakeProvider(lambda req: judge_json(score))


async def test_scores_every_case_and_records_metadata(dataset, settings):
    run = await run_eval(load_prompt("prompts/v1.yaml"), dataset, settings,
                         feature_provider=feature(), judge_provider=judge(4), kind="baseline")  # fmt: skip

    assert run.record.kind == "baseline" and run.record.dataset_version == 3
    assert run.record.aggregates.accuracy == 1.0 and run.record.aggregates.pass_rate == 1.0
    assert all(r.relevance_score == 4.0 for r in run.results)  # GEval's 0-1 mapped back to 1-5
    assert all(r.cost_usd > 0 and r.judge_cost_usd > 0 for r in run.results)


async def test_wrong_category_or_low_relevance_fails_the_case(dataset, settings):
    wrong = next(iter(EMAILS))
    run = await run_eval(load_prompt("prompts/v1.yaml"), dataset, settings,
                         feature_provider=feature(wrong={wrong}), judge_provider=judge(2), kind="run")  # fmt: skip

    assert run.record.aggregates.accuracy == pytest.approx(2 / 3)
    assert run.record.aggregates.pass_rate == 0.0  # relevance 2 < pass minimum of 3


async def test_a_provider_error_fails_one_case_and_the_run_continues(dataset, settings):
    broken = next(iter(EMAILS))
    run = await run_eval(load_prompt("prompts/v1.yaml"), dataset, settings,
                         feature_provider=feature(broken={broken}), judge_provider=judge(), kind="run")  # fmt: skip

    failed = [r for r in run.results if r.error]
    assert len(failed) == 1 and "TimeoutError" in failed[0].error and not failed[0].passed
    assert run.record.aggregates.error_count == 1
    assert run.record.aggregates.case_count == 3  # never dropped from the denominator


async def test_a_fatal_provider_error_aborts_the_whole_run(dataset, settings):
    def respond(req):
        raise FatalProviderError("HTTP 402: Insufficient Balance")

    with pytest.raises(FatalProviderError, match="402"):
        await run_eval(load_prompt("prompts/v1.yaml"), dataset, settings,
                       feature_provider=FakeProvider(respond), judge_provider=judge(), kind="run")  # fmt: skip


async def test_limit_selects_the_first_cases_by_id(dataset, settings):
    run = await run_eval(load_prompt("prompts/v1.yaml"), dataset, settings,
                         feature_provider=feature(), judge_provider=judge(), kind="run", limit=2)  # fmt: skip

    assert [r.case_id for r in run.results] == [c.id for c in dataset.cases[:2]]
