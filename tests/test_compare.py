"""Baseline diff: deltas, regressions, improvements, shared-case comparison."""

import pytest

from regress.eval.compare import compare_runs
from tests.conftest import make_result, make_run


def test_identical_runs_have_no_changes():
    results = [make_result("a"), make_result("b", "general", "general")]
    c = compare_runs(make_run(results, "base"), make_run(results, "new"))

    assert c.accuracy_delta_pp == 0
    assert c.pass_rate_delta_pp == 0
    assert c.avg_relevance_delta == 0
    assert c.regressions == [] and c.improvements == []
    assert c.compared_cases == 2


def test_pass_to_fail_is_a_regression_with_a_reason():
    base = make_run([make_result("a"), make_result("b")], "base")
    new = make_run([make_result("a", predicted="general"), make_result("b")], "new")

    c = compare_runs(base, new)

    assert [r.case_id for r in c.regressions] == ["a"]
    assert c.regressions[0].reason == "category billing -> general"
    assert c.regressions[0].before.passed and not c.regressions[0].after.passed
    assert c.accuracy_delta_pp == pytest.approx(-50.0)


def test_relevance_drop_alone_can_regress_a_case():
    base = make_run([make_result("a", relevance=4)], "base")
    new = make_run([make_result("a", relevance=2)], "new")

    c = compare_runs(base, new)

    assert c.accuracy_delta_pp == 0  # category still right ...
    assert [r.reason for r in c.regressions] == ["relevance 4 -> 2"]  # ... but the case fails
    assert c.avg_relevance_delta == pytest.approx(-2.0)


def test_fail_to_pass_is_an_improvement():
    base = make_run([make_result("a", predicted="account")], "base")
    new = make_run([make_result("a")], "new")

    c = compare_runs(base, new)

    assert [i.case_id for i in c.improvements] == ["a"]
    assert c.regressions == []


def test_errored_case_reason_is_the_error_itself():
    base = make_run([make_result("a")], "base")
    new = make_run([make_result("a", predicted=None, relevance=None, error="provider error: Timeout")], "new")

    c = compare_runs(base, new)

    assert c.regressions[0].reason == "provider error: Timeout"


def test_per_category_deltas():
    base = make_run(
        [make_result("b1"), make_result("b2"), make_result("g1", "general", "general")], "base"
    )
    new = make_run(
        [make_result("b1"), make_result("b2", predicted="account"), make_result("g1", "general", "general")], "new"
    )

    c = compare_runs(base, new)

    assert c.per_category["billing"].cases == 2
    assert c.per_category["billing"].delta_pp == pytest.approx(-50.0)
    assert c.per_category["general"].delta_pp == 0
    assert "technical" not in c.per_category  # no cases -> no row


def test_only_shared_cases_are_compared():
    # Baseline ran 3 cases; the new run used --limit 2 and also has a new case.
    base = make_run([make_result("a"), make_result("b"), make_result("c", predicted="general")], "base")
    new = make_run([make_result("a"), make_result("b"), make_result("d")], "new")

    c = compare_runs(base, new)

    assert c.compared_cases == 2
    assert c.only_in_baseline == ["c"] and c.only_in_run == ["d"]
    assert c.accuracy_before == 1.0  # the failing case "c" is not shared, so it does not count


def test_dataset_version_mismatch_is_flagged():
    base = make_run([make_result("a")], "base", dataset_version=1)
    new = make_run([make_result("a")], "new", dataset_version=2)

    assert compare_runs(base, new).dataset_mismatch


def test_runs_without_shared_cases_raise():
    with pytest.raises(ValueError, match="share no cases"):
        compare_runs(make_run([make_result("a")], "base"), make_run([make_result("b")], "new"))
