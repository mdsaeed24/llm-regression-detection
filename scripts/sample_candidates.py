"""Build a shortlist of real customer messages to hand-adapt into golden cases.

Reads the Kaggle "Customer Support on Twitter" dump (data/raw/twcs.csv), keeps
customer-written tweets, anonymizes them, filters noise, and draws a diverse,
reproducible sample into data/candidates.csv.

The output deliberately contains NO labels and NO coverage-group names: the
keyword groups below exist only to make the sample diverse, and showing them
next to a tweet would anchor the human labeler. Final order is shuffled.

Usage:
    python scripts/sample_candidates.py [--n 200] [--seed 42] [--coverage-report]
"""

from __future__ import annotations

import argparse
import html
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("data/raw/twcs.csv")
DEFAULT_OUTPUT = Path("data/candidates.csv")
MIN_WORDS = 4
NEAR_DUP_JACCARD = 0.6

# --- Anonymization patterns (applied in this order) ---------------------------
URL_RE = r"(?:https?://|www\.)\S+"
EMAIL_RE = r"[\w.+-]+@[\w-]+\.[\w.-]+"
# The Kaggle dump already replaced some emails with this literal token.
KAGGLE_EMAIL_TOKEN_RE = r"__email__"
# Handles addressing the brand at the start, plus separators like "@a, @b:".
LEADING_MENTIONS_RE = r"^(?:[\s,.:;]*@\w+)+[\s,.:;]*"
MENTION_RE = r"@\w+"
# Card-shaped: four groups of four digits (13-19 contiguous digits are far more
# often tracking/order numbers in this data, so those fall through to ORDER_ID).
CARD_RE = r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}\b"
# Dash-joined IDs with a 5+ digit group (e.g. 112-4589921-7712) are order numbers,
# not phones - phone groups are at most 4 digits.
DASHED_ID_RE = r"\b\d+(?:-\d+)*-\d{5,}(?:-\d+)*\b|\b\d{5,}(?:-\d+)+\b"
# Phone-shaped: optional +, then 10+ digits with common separators.
PHONE_RE = r"(?<!\w)\+?\(?\d{1,4}\)?(?:[ .-]?\(?\d{2,4}\)?){2,4}(?!\w)"
# Any token containing a run of 5+ digits: order, ticket, tracking (1Z999AA1...).
LONG_DIGITS_RE = r"\b\w*\d{5,}\w*\b"
# 8+ char codes mixing letters and 3+ digits: payment/transaction refs (LK7735RDGD).
# Short codes like flight numbers (AA3472) are left alone - not identifying.
ALNUM_ID_RE = r"\b(?=[A-Za-z0-9]{8,}\b)(?=[A-Za-z]*\d(?:[A-Za-z]*\d){2})(?=\d*[A-Za-z])[A-Za-z0-9]+\b"
PLACEHOLDER_RE = re.compile(r"\[(?:EMAIL|CARD_NUMBER|PHONE|ORDER_ID)\]")

# --- Language heuristics ----------------------------------------------------
# A real language-ID model (fastText/langdetect) is overkill here and too slow
# for ~1.5M tweets in pure Python; function-word counting is fast and good
# enough because every candidate is read by a human anyway.
EN_FUNCTION_WORDS = frozenset(
    "the a an and or but is are was were be been i you my me we our your it this "
    "that to of in on for with have has had do does did not no can't cant don't "
    "dont won't is it's im i'm just why what when how there they them get got "
    "will would should could am at from so if".split()
)
OTHER_LANG_FUNCTION_WORDS = frozenset(
    # Spanish / Portuguese / French / German / Italian / Dutch markers that do not
    # collide with common English words.
    "que el los las por para una del pero muy estoy tengo gracias hola não você "
    "obrigado está les des pas mais avec pour cette merci bonjour ich nicht und "
    "das ist ein eine mit bitte danke sono della grazie niet een het".split()
)
# Distinctive romanized-Hindi tokens. Ambiguous ones ("ho", "se", "kar", "tha",
# "to", "me", "hi") are excluded to avoid false positives on English typos.
HINGLISH_WORDS = frozenset(
    "hai hain nahi nahin nhi kya kyu kyun kyon mera meri mere mujhe muje aap "
    "aapka aapki aapke apna bhi yaar raha rahi rahe gaya gayi karo karna kiya "
    "diya thi hoga abhi kab kaise kuch paisa paise bhai bahut bohot bhot wala "
    "wali liye lekin matlab chahiye sakta sakte hua hui jaldi".split()
)
TOKEN_RE = re.compile(r"[a-zà-ÿ']+")
NON_ASCII_LETTER_RE = re.compile(r"[^\W\d_a-zA-Z]")

# --- Coverage groups (diversity only; never written to the output) -----------
TOPIC_PATTERNS: dict[str, str] = {
    "money": r"\b(?:refund\w*|charg\w*|bill\w*|payment\w*|paid|pay|invoice\w*|"
    r"subscription\w*|price\w*|fee\w*|overcharg\w*|money|credit card|debit)\b",
    "access": r"\b(?:log ?in|logged|password\w*|account\w*|sign ?in|locked|hack\w*|"
    r"verif\w*|2fa|username|reset)\b",
    "broken": r"\b(?:broken|broke|error\w*|not working|doesn'?t work|won'?t (?:load|work|open)|"
    r"crash\w*|bug\w*|glitch\w*|freez\w*|outage|down|slow|keeps? (?:failing|stopping))\b",
    "question": r"(?:^|\s)(?:how|when|where|what|why|is there|do you|can i|does)\b[^?]*\?",
}
SARCASM_RE = (
    r"\b(?:thanks a lot|thanks for nothing|great job|good job|love how|love that|"
    r"so much for|oh great|yeah right|just what i needed|fantastic|brilliant|"
    r"wonderful|awesome|amazing|totally|clearly|of course|nice one)\b|/s\b|🙄|😒|🙃"
)
MULTI_ISSUE_RE = r"\b(?:also|and another|on top of that|plus|not only|as well as|second(?:ly)?)\b"


@dataclass
class Funnel:
    """Row counts at each filtering stage, for the summary printout."""

    inbound: int = 0
    openers: int = 0
    after_min_words: int = 0
    after_language: int = 0
    hinglish_kept: int = 0
    after_exact_dedup: int = 0
    after_near_dedup: int = 0
    sampled: int = 0
    near_dups_skipped_in_sampling: int = 0


def anonymize(text: pd.Series) -> pd.Series:
    """Remove handles/URLs and mask contact or ID-like numbers (vectorized)."""
    s = text.fillna("").map(html.unescape)
    s = s.str.replace(URL_RE, "", regex=True)
    s = s.str.replace(EMAIL_RE, "[EMAIL]", regex=True)
    s = s.str.replace(KAGGLE_EMAIL_TOKEN_RE, "[EMAIL]", regex=True)
    s = s.str.replace(LEADING_MENTIONS_RE, "", regex=True)  # addressing the brand
    s = s.str.replace(MENTION_RE, "@Company", regex=True)  # inline mentions
    s = s.str.replace(CARD_RE, "[CARD_NUMBER]", regex=True)
    s = s.str.replace(DASHED_ID_RE, "[ORDER_ID]", regex=True)
    # Long digit runs first, so the phone pattern only sees separator-formatted
    # numbers; an unformatted 10-digit phone is still masked, just as an ID.
    s = s.str.replace(LONG_DIGITS_RE, "[ORDER_ID]", regex=True)
    s = s.str.replace(ALNUM_ID_RE, "[ORDER_ID]", regex=True)
    s = s.str.replace(PHONE_RE, _mask_phone, regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def _mask_phone(match: re.Match[str]) -> str:
    """Only treat a match as a phone number if it has 10+ digits (not '2017 2018')."""
    digits = sum(ch.isdigit() for ch in match.group(0))
    return "[PHONE]" if digits >= 10 else match.group(0)


def word_count(text: pd.Series) -> pd.Series:
    """Count words containing at least one letter (placeholders don't count)."""
    return text.str.replace(PLACEHOLDER_RE.pattern, " ", regex=True).str.count(r"\S*[A-Za-z]\S*")


def classify_language(text: str) -> str:
    """Return 'en', 'hinglish', or 'other' using function-word heuristics."""
    if len(NON_ASCII_LETTER_RE.findall(text)) > 0.2 * max(len(text), 1):
        return "other"  # mostly non-Latin script (e.g. Japanese, Arabic)
    tokens = TOKEN_RE.findall(text.lower())
    en = sum(t in EN_FUNCTION_WORDS for t in tokens)
    hindi = len(set(tokens) & HINGLISH_WORDS)
    other = sum(t in OTHER_LANG_FUNCTION_WORDS for t in tokens)
    if hindi >= 2:
        return "hinglish"
    if en >= 2 and en > other:
        return "en"
    return "other"


def signature(text: pd.Series) -> pd.Series:
    """Aggressive normalization for near-duplicate detection.

    Lowercase, drop placeholders/digits/punctuation, collapse whitespace - so
    templated tweets differing only by order number or punctuation collide.
    """
    s = text.str.lower().str.replace(PLACEHOLDER_RE.pattern.lower(), " ", regex=True)
    s = s.str.replace(r"@company", " ", regex=False)
    s = s.str.replace(r"[^a-z\s]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def tag_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean membership masks for each coverage group (multi-label)."""
    lower = df["cleaned_text"].str.lower()
    topics = {name: lower.str.contains(pat, regex=True) for name, pat in TOPIC_PATTERNS.items()}
    topic_hits = sum(m.astype(int) for m in topics.values())

    letters = df["cleaned_text"].str.count(r"[A-Za-z]")
    upper = df["cleaned_text"].str.count(r"[A-Z]")
    groups: dict[str, pd.Series] = dict(topics)
    groups["very_short"] = df["word_count"] <= 7
    groups["very_long"] = df["word_count"] >= 35
    groups["caps_angry"] = (letters >= 15) & (upper >= 0.6 * letters) | df["cleaned_text"].str.contains(
        r"(?:\b[A-Z]{3,}\b.*){3,}|!!!", regex=True
    )
    groups["sarcasm"] = lower.str.contains(SARCASM_RE, regex=True)
    groups["multi_issue"] = (topic_hits >= 2) | (
        (topic_hits >= 1) & lower.str.contains(MULTI_ISSUE_RE, regex=True)
    )
    groups["hinglish"] = df["lang"] == "hinglish"
    # Tweets matching nothing are a group too, so the keyword lists don't
    # define what "a support message" looks like.
    groups["unmatched"] = ~pd.concat(groups.values(), axis=1).any(axis=1)
    return groups


def load_clean(path: Path, chunksize: int, include_replies: bool, funnel: Funnel) -> pd.DataFrame:
    """Stream the CSV in chunks; keep inbound tweets that survive cleaning filters.

    By default only conversation openers are kept (no ``in_response_to_tweet_id``):
    a follow-up like "see you in court next week" is unlabelable without its
    thread, whereas an opener is self-contained - like a support email.
    """
    kept: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "bool",
            "text": "string",
            "in_response_to_tweet_id": "float64",
        },
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk = chunk[chunk["inbound"]]
        funnel.inbound += len(chunk)
        if not include_replies:
            chunk = chunk[chunk["in_response_to_tweet_id"].isna()]
        funnel.openers += len(chunk)

        chunk = chunk.assign(cleaned_text=anonymize(chunk["text"]))
        chunk = chunk.assign(word_count=word_count(chunk["cleaned_text"]))
        chunk = chunk[chunk["word_count"] >= MIN_WORDS]
        funnel.after_min_words += len(chunk)

        chunk = chunk.assign(lang=chunk["cleaned_text"].map(classify_language))
        chunk = chunk[chunk["lang"] != "other"]
        funnel.after_language += len(chunk)
        funnel.hinglish_kept += int((chunk["lang"] == "hinglish").sum())

        kept.append(chunk.drop(columns=["text", "inbound", "in_response_to_tweet_id"]))
    return pd.concat(kept, ignore_index=True)


def deduplicate(df: pd.DataFrame, funnel: Funnel) -> pd.DataFrame:
    """Drop exact duplicates, then near-duplicates sharing a normalized signature."""
    df = df.loc[~df["cleaned_text"].str.lower().duplicated()]
    funnel.after_exact_dedup = len(df)

    df = df.assign(sig=signature(df["cleaned_text"]))
    df = df[df["sig"].str.len() > 0].drop_duplicates(subset="sig")
    funnel.after_near_dedup = len(df)
    return df.reset_index(drop=True)


def _word_set(text: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def diversity_sample(
    df: pd.DataFrame, groups: dict[str, pd.Series], n: int, rng: np.random.Generator, funnel: Funnel
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Round-robin across coverage groups until ``n`` candidates are picked.

    Each group is independently shuffled; groups take turns contributing one
    candidate, so small groups (e.g. Hinglish) get equal representation and any
    shortfall is naturally filled by the larger groups. A candidate is skipped if
    it was already picked, comes from an author already picked, or is a
    near-duplicate (word-set Jaccard >= threshold) of a picked candidate.
    """
    queues = {
        name: iter(rng.permutation(np.flatnonzero(mask.to_numpy())))
        for name, mask in groups.items()
    }
    picked: list[int] = []
    picked_sets: list[frozenset[str]] = []
    seen_rows: set[int] = set()
    seen_authors: set[str] = set()
    coverage = dict.fromkeys(groups, 0)

    def next_valid(queue: Iterator[int]) -> int | None:
        for row in queue:
            if row in seen_rows:
                continue
            seen_rows.add(row)
            if df.at[row, "author_id"] in seen_authors:
                continue
            words = _word_set(df.at[row, "cleaned_text"])
            if any(_jaccard(words, other) >= NEAR_DUP_JACCARD for other in picked_sets):
                funnel.near_dups_skipped_in_sampling += 1
                continue
            picked_sets.append(words)
            return int(row)
        return None

    active = list(queues)
    while len(picked) < n and active:
        for name in list(active):
            if len(picked) >= n:
                break
            row = next_valid(queues[name])
            if row is None:
                active.remove(name)
                continue
            picked.append(row)
            seen_authors.add(df.at[row, "author_id"])
            coverage[name] += 1

    funnel.sampled = len(picked)
    return df.loc[picked], coverage



def build_output(sample: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle and project to the label-free output schema."""
    shuffled = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    return pd.DataFrame(
        {
            "candidate_id": [f"cand_{i:03d}" for i in range(1, len(shuffled) + 1)],
            "cleaned_text": shuffled["cleaned_text"],
            "original_tweet_id": shuffled["tweet_id"],
            "word_count": shuffled["word_count"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=200, help="number of candidates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--include-replies",
        action="store_true",
        help="also sample mid-thread replies (default: conversation openers only)",
    )
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="print aggregate per-group counts (never per-candidate) to sanity-check balance",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    funnel = Funnel()

    df = load_clean(args.input, args.chunksize, args.include_replies, funnel)
    df = deduplicate(df, funnel)
    groups = tag_groups(df)
    sample, coverage = diversity_sample(df, groups, args.n, rng, funnel)
    output = build_output(sample, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"Inbound (customer) tweets      : {funnel.inbound:>9,}")
    scope = "all inbound (replies included)" if args.include_replies else "conversation openers"
    print(f"  {scope:29}: {funnel.openers:>9,}")
    print(f"  >= {MIN_WORDS} words after cleaning    : {funnel.after_min_words:>9,}")
    print(f"  English or Hinglish          : {funnel.after_language:>9,}  (Hinglish: {funnel.hinglish_kept:,})")
    print(f"  after exact dedup            : {funnel.after_exact_dedup:>9,}")
    print(f"  after near-dup dedup         : {funnel.after_near_dedup:>9,}  (eligible pool)")
    print(f"Sampled candidates             : {funnel.sampled:>9,}  -> {args.output}")
    if args.coverage_report:
        print("\nCoverage (aggregate; a candidate counts once, for the group that drew it):")
        for name, count in coverage.items():
            print(f"  {name:12} pool={int(groups[name].sum()):>8,}  drawn={count}")
        print(f"  near-duplicates skipped during sampling: {funnel.near_dups_skipped_in_sampling}")


if __name__ == "__main__":
    main()
