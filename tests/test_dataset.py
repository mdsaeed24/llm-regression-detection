"""Golden dataset tooling: validation messages, stable IDs, versioning, PII and quality checks.

Emails here are synthetic test fixtures, not golden cases.
"""

import textwrap

import pytest

from regress.dataset import (
    GoldenCaseDraft,
    assign_ids,
    case_id,
    check_cases,
    content_hash,
    coverage_issues,
    diff_datasets,
    next_version,
    parse_draft,
    scan_pii,
)


def draft(cases_yaml: str) -> str:
    return "cases:\n" + textwrap.indent(textwrap.dedent(cases_yaml).strip(), "") + "\n"


VALID = """
- input_email: |
    I was charged twice this month, please refund one.
  expected_category: billing
  ideal_summary: Customer was charged twice and wants one charge refunded.
  expected_difficulty: easy
- input_email: |
    The app crashes on launch.
  expected_category: Technical
  ideal_summary: Customer's app crashes on launch.
  expected_difficulty: medium
"""


def test_valid_draft_parses_and_normalises_labels():
    parsed = parse_draft(draft(VALID))

    assert parsed.issues == []
    assert [c.expected_category for c in parsed.cases] == ["billing", "technical"]  # case-insensitive
    assert parsed.case_lines == [2, 7]


def test_every_error_names_the_case_and_line():
    text = draft("""
    - input_email: fine email one
      expected_category: billing
      ideal_summary: S.
      expected_difficulty: easy
    - input_email: second email
      expected_category: shipping
      ideal_summary: S.
      expected_difficulty: easy
    - input_email: third email
      expected_categry: billing
      ideal_summary: S.
      expected_difficulty: easy
    """)

    parsed = parse_draft(text)
    messages = {(i.case_number, i.line): i.message for i in parsed.issues}

    assert len(parsed.cases) == 1  # the valid case still parses
    assert "must be one of billing, technical, account, general" in messages[(2, 6)]
    assert any("did you mean 'expected_category'" in m for (n, _), m in messages.items() if n == 3)


def test_tabs_produce_a_yaml_error_with_a_hint():
    parsed = parse_draft("cases:\n\t- input_email: x\n")

    assert parsed.issues[0].level == "error"
    assert "spaces, not tabs" in parsed.issues[0].message


def test_manual_ids_are_rejected():
    parsed = parse_draft(draft("""
    - id: gc_123
      input_email: x
      expected_category: billing
      ideal_summary: S.
      expected_difficulty: easy
    """))

    assert "IDs are assigned automatically" in parsed.issues[0].message


def test_case_id_ignores_case_and_whitespace_but_not_content():
    assert case_id("Refund  me\nplease") == case_id("refund me please")
    assert case_id("refund me please") != case_id("refund me now")
    assert case_id("x").startswith("gc_")


def test_ids_and_hash_do_not_depend_on_order():
    cases = assign_ids(parse_draft(draft(VALID)).cases)

    assert content_hash(cases) == content_hash(list(reversed(cases)))


def test_version_bumps_only_when_content_changes():
    from regress.dataset import build_dataset

    cases = assign_ids(parse_draft(draft(VALID)).cases)
    v1 = build_dataset(cases, 1, "draft", content_hash(cases))

    assert next_version(None, "h") == (1, True)
    assert next_version(v1, content_hash(cases)) == (1, False)
    assert next_version(v1, "different") == (2, True)


def test_relabels_are_reported_separately_from_edits():
    old = assign_ids(parse_draft(draft(VALID)).cases)
    new = [old[0].model_copy(update={"expected_category": "account"}),
           old[1].model_copy(update={"notes": "clarified"})]  # fmt: skip

    diff = diff_datasets(old, new)

    assert diff.relabeled == [f"{old[0].id}: billing -> account"]
    assert diff.edited == [old[1].id]


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail me at jane.doe@example.com", "email address"),
        ("call +44 7911 123456 now", "phone number"),
        ("card 4111 1111 1111 1111 declined", "card number"),
        ("see https://example.com/order", "URL"),
        ("order 98765432 is late", "long digit sequence (order/card/account no.)"),
        ("thanks @janedoe99", "social media handle"),
    ],
)
def test_pii_is_detected(text, kind):
    assert kind in [k for k, _ in scan_pii(text)]


@pytest.mark.parametrize(
    "text",
    ["@Company your app is down", "order [ORDER_ID] is late", "I paid 49.99 on 5/10", "waited 3 hours"],
)
def test_placeholders_and_small_numbers_are_not_pii(text):
    assert scan_pii(text) == []


def cases(*specs: tuple[str, str]) -> list[GoldenCaseDraft]:
    return [GoldenCaseDraft(input_email=e, expected_category=c, ideal_summary="Customer S.",
                            expected_difficulty="easy") for e, c in specs]  # fmt: skip


def check(drafts, **kwargs):
    from regress.dataset import ParsedDraft

    parsed = ParsedDraft(cases=drafts, case_numbers=list(range(1, len(drafts) + 1)),
                         case_lines=[None] * len(drafts), issues=[])  # fmt: skip
    return check_cases(parsed, **kwargs)


def test_exact_duplicates_are_errors():
    issues = check(cases(("Refund me please", "billing"), ("refund   ME please", "billing")))

    assert [i.level for i in issues] == ["error"]
    assert "exact duplicate of case #1" in issues[0].message


def test_near_duplicates_with_different_labels_are_called_out():
    issues = check(cases(("My card was charged twice for the March plan", "billing"),
                         ("My card was charged twice for the March plan!!", "account")))  # fmt: skip

    assert any("labels differ" in i.message for i in issues)


def test_leakage_from_few_shot_examples_is_flagged():
    issues = check(
        cases(("Do you have a dark mode on the roadmap? Love the product.", "general")),
        leakage_texts={"v1.yaml example 4": "Do you have a dark mode on the roadmap? Love the product otherwise."},
    )

    assert any("eval leakage" in i.message for i in issues)


def test_coverage_reports_thin_categories_and_empty_cells():
    issues = coverage_issues(cases(*[("e%d" % i, "billing") for i in range(10)]))
    messages = " ".join(i.message for i in issues)

    assert "category 'general' is 0/10" in messages
    assert "no cases for:" in messages
