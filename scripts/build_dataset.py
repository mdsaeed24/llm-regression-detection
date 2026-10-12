"""Compile data/golden_draft.yaml into data/golden.json.

Steps: validate every case (all errors reported at once, with case number and
line) -> assign stable hash-based IDs -> detect duplicates/near-duplicates ->
warn about leftover personal data and prompt leakage -> print coverage ->
write golden.json, bumping dataset_version only if content changed.

Exit codes: 0 ok, 1 validation errors (nothing written), 2 warnings with --strict.

Usage:
    python scripts/build_dataset.py [--draft PATH] [--output PATH] [--strict] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from regress.dataset import (  # noqa: E402
    GoldenDataset,
    Issue,
    assign_ids,
    build_dataset,
    check_cases,
    content_hash,
    coverage_issues,
    diff_datasets,
    distribution_table,
    load_dataset,
    next_version,
    parse_draft,
    save_dataset,
)
from regress.prompts import load_prompt  # noqa: E402

DEFAULT_DRAFT = Path("data/golden_draft.yaml")
DEFAULT_OUTPUT = Path("data/golden.json")
DEFAULT_PROMPTS_DIR = Path("prompts")


def few_shot_texts(prompts_dir: Path) -> dict[str, str]:
    """Few-shot emails from every prompt version, keyed '<file> example N'."""
    texts: dict[str, str] = {}
    for path in sorted(prompts_dir.glob("*.yaml")):
        try:
            prompt = load_prompt(path)
        except Exception as exc:  # a broken prompt shouldn't block dataset work
            print(f"note: skipping leakage check for {path}: {exc}", file=sys.stderr)
            continue
        for n, example in enumerate(prompt.few_shot_examples, start=1):
            texts[f"{path.name} example {n}"] = example.email
    return texts


def example_texts(draft_text: str) -> list[str]:
    """input_email of the format examples in the draft's 'examples:' section."""
    data = yaml.safe_load(draft_text)
    examples = (data.get("examples") or []) if isinstance(data, dict) else []
    return [e["input_email"] for e in examples if isinstance(e, dict) and "input_email" in e]


def print_issues(issues: list[Issue]) -> None:
    for issue in sorted(issues, key=lambda i: (i.level != "error", i.case_number or 0)):
        print("  " + issue.format())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures (CI)")
    parser.add_argument("--check", action="store_true", help="validate and report only; never write")
    args = parser.parse_args()

    draft_text = args.draft.read_text(encoding="utf-8")
    parsed = parse_draft(draft_text)
    issues = list(parsed.issues)
    if parsed.cases:
        issues += check_cases(parsed, few_shot_texts(args.prompts_dir), example_texts(draft_text))
        issues += coverage_issues(parsed.cases)

    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    total = len(parsed.cases) + len({i.case_number for i in parsed.issues if i.case_number})

    print(f"Draft: {args.draft}  ({total} cases, {parsed.ignored_examples} format examples ignored)")
    if errors:
        print(f"\n{len(errors)} error(s) - fix these, nothing was written:")
        print_issues(errors)
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        print_issues(warnings)
    if parsed.cases:
        print("\nDistribution (valid cases):")
        print("  " + distribution_table(parsed.cases).replace("\n", "\n  "))

    if errors:
        return 1
    if not parsed.cases:
        print("\nNo cases yet - add some under 'cases:' in the draft.")
        return 1

    cases = assign_ids(parsed.cases)
    digest = content_hash(cases)
    previous: GoldenDataset | None = load_dataset(args.output) if args.output.exists() else None
    version, changed = next_version(previous, digest)

    print()
    if previous and changed:
        diff = diff_datasets(previous.cases, cases)
        print(
            f"Changes vs v{previous.dataset_version}: +{len(diff.added)} added, "
            f"-{len(diff.removed)} removed, {len(diff.edited)} edited, {len(diff.relabeled)} relabeled"
        )
        for line in diff.relabeled:
            print(f"  RELABELED {line}  (affects pass/fail vs existing baselines)")

    if args.check:
        print(f"--check: would write dataset_version={version} ({'changed' if changed else 'unchanged'})")
    elif not changed:
        print(f"Unchanged: {args.output} stays at dataset_version={version} ({digest})")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(build_dataset(cases, version, str(args.draft), digest), args.output)
        print(f"Wrote {args.output}: dataset_version={version}, {len(cases)} cases, hash {digest}")

    return 2 if (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
