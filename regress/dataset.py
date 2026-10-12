"""Golden dataset: schema, validation, stable IDs, quality checks, versioning.

The human-written source of truth is ``data/golden_draft.yaml``. This module
turns it into ``data/golden.json``, which is what the eval engine consumes.
Everything here is pure (no I/O beyond explicit load/save helpers) so it can
be unit-tested without files or network.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from regress.models import CATEGORIES, Category

Difficulty = Literal["easy", "medium", "hard"]
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

NEAR_DUPLICATE_RATIO = 0.85
ID_PREFIX = "gc_"
ID_HASH_LEN = 10


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class GoldenCaseDraft(BaseModel):
    """One hand-written case as it appears in golden_draft.yaml (no ID)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    input_email: str = Field(min_length=1)
    expected_category: Category
    ideal_summary: str = Field(min_length=1)
    expected_difficulty: Difficulty
    notes: str = ""

    @field_validator("expected_category", "expected_difficulty", mode="before")
    @classmethod
    def _lowercase(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("ideal_summary")
    @classmethod
    def _one_line(cls, value: str) -> str:
        """Block scalars can sneak in line breaks; a summary is one line."""
        return " ".join(value.split())

    @field_validator("notes", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        return "" if value is None else value


class GoldenCase(GoldenCaseDraft):
    """A case with its stable, content-derived ID (as stored in golden.json)."""

    id: str


class GoldenDataset(BaseModel):
    """The compiled dataset consumed by the eval engine."""

    dataset_version: int = Field(ge=1)
    content_hash: str
    built_at: datetime
    source: str
    case_count: int
    cases: list[GoldenCase]


# --------------------------------------------------------------------------- #
# Parsing with line numbers
# --------------------------------------------------------------------------- #
class Issue(BaseModel):
    """A validation error or warning tied to a case number and source line."""

    level: Literal["error", "warning"]
    message: str
    case_number: int | None = None  # 1-based position in the draft
    line: int | None = None  # 1-based line in the YAML file

    def format(self) -> str:
        where = []
        if self.case_number is not None:
            where.append(f"case #{self.case_number}")
        if self.line is not None:
            where.append(f"line {self.line}")
        prefix = f"[{', '.join(where)}] " if where else ""
        return f"{self.level.upper():7} {prefix}{self.message}"


class ParsedDraft(BaseModel):
    """Result of parsing the draft: valid cases plus any problems found."""

    cases: list[GoldenCaseDraft]
    case_numbers: list[int]  # draft position of each valid case
    case_lines: list[int | None]
    issues: list[Issue]
    ignored_examples: int = 0


def parse_draft(text: str) -> ParsedDraft:
    """Parse golden_draft.yaml text, validating each case independently.

    All cases are checked even if early ones fail, so the author sees every
    problem in one pass instead of fixing them one at a time.
    """
    try:
        data = yaml.safe_load(text)
        root = yaml.compose(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark else None
        hint = " (check indentation; YAML needs spaces, not tabs)"
        return ParsedDraft(
            cases=[],
            case_numbers=[],
            case_lines=[],
            issues=[Issue(level="error", message=f"YAML syntax error: {exc}{hint}", line=line)],
        )

    if not isinstance(data, dict) or "cases" not in data:
        return ParsedDraft(
            cases=[],
            case_numbers=[],
            case_lines=[],
            issues=[Issue(level="error", message="file must have a top-level 'cases:' list")],
        )

    raw_cases = data.get("cases") or []
    examples = data.get("examples") or []
    lines = _case_start_lines(root)
    issues: list[Issue] = []
    cases: list[GoldenCaseDraft] = []
    numbers: list[int] = []
    case_lines: list[int | None] = []

    if not isinstance(raw_cases, list):
        issues.append(Issue(level="error", message="'cases' must be a list of cases (- input_email: ...)"))
        raw_cases = []

    for index, raw in enumerate(raw_cases):
        number = index + 1
        line = lines[index] if index < len(lines) else None
        if not isinstance(raw, dict):
            issues.append(
                Issue(level="error", message="case is not a mapping of fields", case_number=number, line=line)
            )
            continue
        if "id" in raw:
            issues.append(
                Issue(
                    level="error",
                    message="remove the 'id' field - IDs are assigned automatically",
                    case_number=number,
                    line=line,
                )
            )
            continue
        try:
            cases.append(GoldenCaseDraft.model_validate(raw))
            numbers.append(number)
            case_lines.append(line)
        except ValidationError as exc:
            for err in exc.errors():
                issues.append(
                    Issue(
                        level="error",
                        message=_describe_validation_error(err),
                        case_number=number,
                        line=line,
                    )
                )

    return ParsedDraft(
        cases=cases,
        case_numbers=numbers,
        case_lines=case_lines,
        issues=issues,
        ignored_examples=len(examples) if isinstance(examples, list) else 0,
    )


def _case_start_lines(root: yaml.Node | None) -> list[int]:
    """1-based line number where each item of the top-level 'cases' list starts."""
    if not isinstance(root, yaml.MappingNode):
        return []
    for key, value in root.value:
        if getattr(key, "value", None) == "cases" and isinstance(value, yaml.SequenceNode):
            return [item.start_mark.line + 1 for item in value.value]
    return []


def _describe_validation_error(err: dict) -> str:
    """Turn a Pydantic error dict into a message a dataset author can act on."""
    field = ".".join(str(p) for p in err["loc"]) or "case"
    kind = err["type"]
    if kind == "missing":
        return f"missing required field '{field}'"
    if kind == "extra_forbidden":
        guess = get_close_matches(field, list(GoldenCaseDraft.model_fields), n=1)
        if guess:
            return f"unknown field '{field}' - did you mean '{guess[0]}'?"
        known = ", ".join(GoldenCaseDraft.model_fields)
        return f"unknown field '{field}'. Allowed fields: {known}"
    if kind == "literal_error":
        allowed = CATEGORIES if field == "expected_category" else DIFFICULTIES
        return f"{field}: got {err.get('input')!r}, must be one of {', '.join(allowed)}"
    if kind == "string_too_short":
        return f"{field} is empty"
    return f"{field}: {err['msg']}"


# --------------------------------------------------------------------------- #
# IDs, hashing, versioning
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Case- and whitespace-insensitive form used for IDs and duplicate checks."""
    return " ".join(text.lower().split())


def case_id(input_email: str) -> str:
    """Stable ID derived from the email content only.

    Reordering or adding cases never changes existing IDs. Editing the email
    text does - deliberately, because it is then a different test input and
    its history against old runs is no longer comparable.
    """
    digest = hashlib.sha256(normalize_text(input_email).encode()).hexdigest()
    return f"{ID_PREFIX}{digest[:ID_HASH_LEN]}"


def content_hash(cases: list[GoldenCase]) -> str:
    """Order-independent hash of all case content (labels included)."""
    payload = [c.model_dump(mode="json") for c in sorted(cases, key=lambda c: c.id)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def assign_ids(drafts: list[GoldenCaseDraft]) -> list[GoldenCase]:
    """Attach stable IDs to validated drafts."""
    return [GoldenCase(id=case_id(d.input_email), **d.model_dump()) for d in drafts]


def next_version(previous: GoldenDataset | None, new_hash: str) -> tuple[int, bool]:
    """Return (version, changed). Version bumps only when content changes."""
    if previous is None:
        return 1, True
    if previous.content_hash == new_hash:
        return previous.dataset_version, False
    return previous.dataset_version + 1, True


class DatasetDiff(BaseModel):
    """What changed between two dataset builds."""

    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    relabeled: list[str] = Field(default_factory=list)  # human-readable lines
    edited: list[str] = Field(default_factory=list)  # summary/difficulty/notes only


def diff_datasets(old: list[GoldenCase], new: list[GoldenCase]) -> DatasetDiff:
    """Compare by ID. A changed expected_category is called out separately,
    because it silently changes pass/fail for existing baselines."""
    old_by_id = {c.id: c for c in old}
    new_by_id = {c.id: c for c in new}
    diff = DatasetDiff(
        added=sorted(new_by_id.keys() - old_by_id.keys()),
        removed=sorted(old_by_id.keys() - new_by_id.keys()),
    )
    for cid in sorted(old_by_id.keys() & new_by_id.keys()):
        before, after = old_by_id[cid], new_by_id[cid]
        if before.expected_category != after.expected_category:
            diff.relabeled.append(f"{cid}: {before.expected_category} -> {after.expected_category}")
        elif before != after:
            diff.edited.append(cid)
    return diff


# --------------------------------------------------------------------------- #
# Quality checks (all produce warnings, never block the build)
# --------------------------------------------------------------------------- #
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "URL": re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|net|org|io|in|co)(?:/\S*)?\b", re.I),
    "card number": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,7}\b"),
    "phone number": re.compile(r"(?<!\w)\+?\(?\d{2,4}\)?(?:[ .-]?\(?\d{2,4}\)?){2,4}(?!\w)"),
    "long digit sequence (order/card/account no.)": re.compile(r"\d(?:[ -]?\d){5,}"),
    "social media handle": re.compile(r"(?<![\w.])@(?!Company\b)\w{2,}"),
}
_PHONE_MIN_DIGITS = 10


def scan_pii(text: str) -> list[tuple[str, str]]:
    """Return (kind, matched_text) for anything that looks like leftover personal data.

    Heuristic and deliberately noisy: a false positive costs a glance, a false
    negative leaks a real customer's details into a public repo.
    Placeholders like [ORDER_ID] are fine and never match.
    """
    hits: list[tuple[str, str]] = []
    seen_spans: list[tuple[int, int]] = []
    for kind, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            span = match.span()
            if any(s <= span[0] < e for s, e in seen_spans):
                continue  # already reported under a more specific kind
            value = match.group(0)
            if kind == "phone number" and sum(ch.isdigit() for ch in value) < _PHONE_MIN_DIGITS:
                continue
            hits.append((kind, value))
            seen_spans.append(span)
    return hits


def find_near_duplicates(
    texts: list[str], threshold: float = NEAR_DUPLICATE_RATIO
) -> list[tuple[int, int, float]]:
    """Pairs (i, j, similarity) of texts that are suspiciously similar.

    O(n^2) with SequenceMatcher is fine for a hand-written set of ~100 cases;
    ``quick_ratio`` is an upper bound, so it cheaply skips most pairs.
    """
    normalized = [normalize_text(t) for t in texts]
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            matcher = SequenceMatcher(None, normalized[i], normalized[j], autojunk=False)
            if matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio >= threshold:
                pairs.append((i, j, ratio))
    return pairs


def check_cases(
    parsed: ParsedDraft,
    leakage_texts: dict[str, str] | None = None,
    example_texts: list[str] | None = None,
) -> list[Issue]:
    """Duplicate, PII, summary-style, and prompt-leakage checks on valid cases.

    Exact duplicates are errors (they would collide on ID). Everything else is
    a warning: near-duplicates can be intentional variants (e.g. a heavy-typo
    rewrite of another case), and PII detection is heuristic.
    """
    issues: list[Issue] = []
    cases, numbers, lines = parsed.cases, parsed.case_numbers, parsed.case_lines

    # Exact duplicates (same normalized email -> same ID).
    first_seen: dict[str, int] = {}
    for k, case in enumerate(cases):
        cid = case_id(case.input_email)
        if cid in first_seen:
            issues.append(
                Issue(
                    level="error",
                    message=f"exact duplicate of case #{numbers[first_seen[cid]]} (same input_email)",
                    case_number=numbers[k],
                    line=lines[k],
                )
            )
        else:
            first_seen[cid] = k

    # Near-duplicates among first occurrences (exact dups are already errors),
    # with a louder message if their labels disagree.
    unique = sorted(first_seen.values())
    for a, b, ratio in find_near_duplicates([cases[k].input_email for k in unique]):
        i, j = unique[a], unique[b]
        conflict = cases[i].expected_category != cases[j].expected_category
        detail = (
            f" AND their labels differ ({cases[i].expected_category} vs "
            f"{cases[j].expected_category}) - check labeling consistency"
            if conflict
            else " - fine if intentional (e.g. a typo/style variant), otherwise remove one"
        )
        issues.append(
            Issue(
                level="warning",
                message=f"near-duplicate of case #{numbers[i]} ({ratio:.0%} similar){detail}",
                case_number=numbers[j],
                line=lines[j],
            )
        )

    for k, case in enumerate(cases):
        # Leftover personal data in any free-text field.
        for field in ("input_email", "ideal_summary", "notes"):
            for kind, value in scan_pii(getattr(case, field)):
                issues.append(
                    Issue(
                        level="warning",
                        message=f"possible {kind} in {field}: {value!r} - anonymize or use a placeholder",
                        case_number=numbers[k],
                        line=lines[k],
                    )
                )

        # Summary style: the prompt asks for one sentence, so should the gold.
        words = len(case.ideal_summary.split())
        if re.search(r"[.!?]\s+\S", case.ideal_summary):
            issues.append(
                Issue(
                    level="warning",
                    message="ideal_summary looks like more than one sentence",
                    case_number=numbers[k],
                    line=lines[k],
                )
            )
        if words > 30:
            issues.append(
                Issue(
                    level="warning",
                    message=f"ideal_summary is {words} words; aim for <= 25",
                    case_number=numbers[k],
                    line=lines[k],
                )
            )

        # Leakage: a golden case that mirrors a prompt's few-shot example
        # measures memorization of the prompt, not generalization.
        for source, text in (leakage_texts or {}).items():
            ratio = SequenceMatcher(None, normalize_text(case.input_email), normalize_text(text)).ratio()
            if ratio >= 0.7:
                issues.append(
                    Issue(
                        level="warning",
                        message=f"{ratio:.0%} similar to a few-shot example in {source} (eval leakage)",
                        case_number=numbers[k],
                        line=lines[k],
                    )
                )

        # A format example copied into 'cases' and never rewritten.
        for text in example_texts or []:
            if SequenceMatcher(None, normalize_text(case.input_email), normalize_text(text)).ratio() >= 0.85:
                issues.append(
                    Issue(
                        level="warning",
                        message="looks like one of the format examples - replace it with a real case",
                        case_number=numbers[k],
                        line=lines[k],
                    )
                )

    return issues


def coverage_issues(cases: list[GoldenCaseDraft], min_share: float = 0.15) -> list[Issue]:
    """Warn about under-represented categories, difficulties, and empty cells."""
    issues: list[Issue] = []
    n = len(cases)
    if n == 0:
        return issues
    by_cat = Counter(c.expected_category for c in cases)
    by_diff = Counter(c.expected_difficulty for c in cases)
    for cat in CATEGORIES:
        if by_cat[cat] / n < min_share:
            issues.append(
                Issue(level="warning", message=f"category '{cat}' is {by_cat[cat]}/{n} cases (< {min_share:.0%})")
            )
    for diff in DIFFICULTIES:
        if by_diff[diff] / n < min_share:
            issues.append(
                Issue(level="warning", message=f"difficulty '{diff}' is {by_diff[diff]}/{n} cases (< {min_share:.0%})")
            )
    cells = Counter((c.expected_category, c.expected_difficulty) for c in cases)
    empty = [f"{cat}/{diff}" for cat in CATEGORIES for diff in DIFFICULTIES if cells[(cat, diff)] == 0]
    if empty:
        issues.append(Issue(level="warning", message=f"no cases for: {', '.join(empty)}"))
    return issues


def distribution_table(cases: list[GoldenCaseDraft]) -> str:
    """Category x difficulty table with row/column totals."""
    cells = Counter((c.expected_category, c.expected_difficulty) for c in cases)
    n = len(cases) or 1
    header = f"{'category':<11}" + "".join(f"{d:>8}" for d in DIFFICULTIES) + f"{'total':>8}{'share':>8}"
    rows = [header, "-" * len(header)]
    for cat in CATEGORIES:
        counts = [cells[(cat, d)] for d in DIFFICULTIES]
        total = sum(counts)
        rows.append(
            f"{cat:<11}" + "".join(f"{x:>8}" for x in counts) + f"{total:>8}{total / n:>8.0%}"
        )
    col_totals = [sum(cells[(cat, d)] for cat in CATEGORIES) for d in DIFFICULTIES]
    rows.append("-" * len(header))
    rows.append(f"{'total':<11}" + "".join(f"{x:>8}" for x in col_totals) + f"{len(cases):>8}")
    rows.append(f"{'share':<11}" + "".join(f"{x / n:>8.0%}" for x in col_totals))
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #
def load_dataset(path: Path | str) -> GoldenDataset:
    """Load and validate a compiled golden.json."""
    return GoldenDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def build_dataset(
    cases: list[GoldenCase], version: int, source: str, digest: str
) -> GoldenDataset:
    """Assemble the compiled dataset (cases sorted by ID for stable diffs)."""
    ordered = sorted(cases, key=lambda c: c.id)
    return GoldenDataset(
        dataset_version=version,
        content_hash=digest,
        built_at=datetime.now(UTC).replace(microsecond=0),
        source=source,
        case_count=len(ordered),
        cases=ordered,
    )


def save_dataset(dataset: GoldenDataset, path: Path | str) -> None:
    """Write golden.json with stable formatting (readable diffs in code review)."""
    Path(path).write_text(dataset.model_dump_json(indent=2) + "\n", encoding="utf-8")
