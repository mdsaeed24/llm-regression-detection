"""Async eval runner: PromptConfig + golden dataset -> a scored, stored-ready EvalRun."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from regress.config import Settings
from regress.cost import estimate_cost
from regress.dataset import GoldenCase, GoldenDataset
from regress.eval.relevance import RelevanceJudge
from regress.eval.results import CaseResult, EvalRun, RunKind, RunRecord, compute_aggregates
from regress.feature import EmailClassifier
from regress.prompts import PromptConfig
from regress.providers.base import FatalProviderError, LLMProvider

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


async def run_eval(
    prompt: PromptConfig,
    dataset: GoldenDataset,
    settings: Settings,
    *,
    feature_provider: LLMProvider,
    judge_provider: LLMProvider,
    kind: RunKind,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
) -> EvalRun:
    """Classify and score every case (or the first ``limit``), concurrently.

    Cases run in dataset order (sorted by ID), so ``--limit N`` always selects the
    same N cases. A ``FatalProviderError`` aborts the run; any other per-case
    failure is recorded on that case and the run continues.
    """
    cases = dataset.cases[:limit] if limit else dataset.cases
    classifier = EmailClassifier(
        prompt, feature_provider, settings.feature, settings.runtime.validation_retries
    )
    judge = RelevanceJudge(judge_provider, settings.judge)
    semaphore = asyncio.Semaphore(settings.runtime.concurrency)
    done = 0

    async def run_one(case: GoldenCase) -> CaseResult:
        nonlocal done
        async with semaphore:
            result = await _evaluate_case(case, classifier, judge, settings, prompt.model)
        done += 1
        if progress:
            progress(done, len(cases))
        return result

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_one(case)) for case in cases]
    except* FatalProviderError as eg:
        raise eg.exceptions[0] from None

    results = [t.result() for t in tasks]
    record = RunRecord(
        run_id=_new_run_id(prompt.version),
        created_at=datetime.now(UTC),
        kind=kind,
        prompt_version=prompt.version,
        prompt_hash=prompt.content_hash,
        model=prompt.model,
        dataset_version=dataset.dataset_version,
        dataset_hash=dataset.content_hash,
        git_sha=current_git_sha(),
        aggregates=compute_aggregates(results),
    )
    return EvalRun(record=record, results=results)


async def _evaluate_case(
    case: GoldenCase,
    classifier: EmailClassifier,
    judge: RelevanceJudge,
    settings: Settings,
    feature_model: str,
) -> CaseResult:
    base = CaseResult(
        case_id=case.id,
        expected_category=case.expected_category,
        expected_difficulty=case.expected_difficulty,
    )

    try:
        outcome = await classifier.classify(case.input_email)
    except FatalProviderError:
        raise
    except Exception as exc:  # survived provider retries: an infra failure on this case
        logger.warning("Case %s: classifier call failed: %s", case.id, exc)
        return base.model_copy(update={"error": f"provider error: {type(exc).__name__}: {exc}"})

    scored = base.model_copy(
        update={
            "attempts": outcome.attempts,
            "latency_ms": outcome.latency_ms,
            "usage": outcome.usage,
            "cost_usd": estimate_cost(outcome.usage, settings.pricing_for(feature_model)),
            "raw_outputs": outcome.raw_outputs,
        }
    )
    if outcome.classification is None:
        return scored.model_copy(update={"error": f"invalid output: {outcome.error}"})

    classification = outcome.classification
    scored = scored.model_copy(
        update={
            "predicted_category": classification.category,
            "summary": classification.summary,
            "category_match": classification.category == case.expected_category,
        }
    )

    try:
        verdict = await judge.score(case.input_email, classification.summary, case.ideal_summary)
    except FatalProviderError:
        raise
    except Exception as exc:
        # Without a relevance score the case cannot pass; flag it rather than guess.
        logger.warning("Case %s: judge failed: %s", case.id, exc)
        return scored.model_copy(update={"error": f"judge error: {type(exc).__name__}: {exc}"})

    return scored.model_copy(
        update={
            "relevance_score": verdict.score,
            "relevance_reason": verdict.reason,
            "judge_usage": verdict.usage,
            "judge_cost_usd": estimate_cost(verdict.usage, settings.pricing_for(settings.judge.model)),
            "passed": scored.category_match
            and verdict.score >= settings.evaluation.relevance_pass_min,
        }
    )


def _new_run_id(prompt_version: str) -> str:
    """Sortable and readable: 20261012-142233-v2-1a2b3c."""
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{prompt_version}-{uuid.uuid4().hex[:6]}"


def current_git_sha() -> str | None:
    """HEAD commit, suffixed '-dirty' if the tree has uncommitted changes; None outside git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha
