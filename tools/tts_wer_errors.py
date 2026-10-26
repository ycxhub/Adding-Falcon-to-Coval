# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Expand a TTS WER run into one row per mis-generated word, for linguistic review.

``tools/tts_wer.py`` reports a WER percentage per utterance. This script reopens
the same run and re-derives the word-level alignment behind each score, so a
linguist can see the exact word that went wrong, what the ASR heard instead, and
where it sits in the original sentence.

The alignment mirrors upstream ``coval_bench.metrics.wer._build_word_errors``
(same DP matrix, same backtrace priority), extended to carry the reference index
so surrounding context can be quoted. Per-utterance error counts are asserted
against the source CSV, so a drift in either implementation fails loudly rather
than shipping a mismatched breakdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# A word carrying digits: the normalizer rewrites times, money, and percentages
# on both sides, so these positions can differ without the speech being wrong.
_HAS_DIGIT = re.compile(r"\d")
# Letters and digits fused together — order codes, tracking numbers, versions.
_ALNUM_ID = re.compile(r"(?=.*[a-z])(?=.*\d)", re.IGNORECASE)

# How a TTS engine tends to verbalize punctuation that the written prompt carries
# but the normalized reference drops. An inserted word from this set, in a
# sentence containing the matching symbol, is the voice reading the symbol aloud.
_SYMBOL_READINGS: dict[str, frozenset[str]] = {
    "#": frozenset({"hash", "pound", "number"}),
    "-": frozenset({"to", "dash", "through", "minus"}),
    "/": frozenset({"slash", "per"}),
    "&": frozenset({"and", "ampersand"}),
    "+": frozenset({"plus"}),
    "@": frozenset({"at"}),
}

# Unit abbreviations the voice may expand (or the ASR may contract).
_UNIT_FORMS: tuple[frozenset[str], ...] = (
    frozenset({"mb", "megabytes", "megabyte"}),
    frozenset({"gb", "gigabytes", "gigabyte"}),
    frozenset({"kb", "kilobytes", "kilobyte"}),
    frozenset({"km", "kilometers", "kilometres"}),
    frozenset({"kg", "kilograms"}),
    frozenset({"lb", "lbs", "pounds"}),
    frozenset({"oz", "ounces"}),
    frozenset({"mph", "miles"}),
    frozenset({"hr", "hrs", "hours"}),
    frozenset({"min", "mins", "minutes"}),
)

# Tokens that only ever appear as part of a time or an alphanumeric run once the
# normalizer has split them; capitalization in the prompt ("2:15 PM") must not
# make them look like names.
_MAX_FRAGMENT_LEN = 2

# A digit run this long is an identifier being spelled out, not a quantity.
_ID_RUN_LEN = 5


@dataclass(frozen=True)
class AlignedError:
    type: str
    ref_index: int
    hyp_index: int
    reference: str | None
    hypothesis: str | None


def align(ref_words: list[str], hyp_words: list[str]) -> list[AlignedError]:
    """Levenshtein backtrace returning each edit plus its reference position."""
    n, m = len(ref_words), len(hyp_words)
    d: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1], d[i][j - 1], d[i - 1][j])

    errors: list[AlignedError] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            errors.append(
                AlignedError("substitution", i - 1, j - 1, ref_words[i - 1], hyp_words[j - 1])
            )
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            errors.append(AlignedError("insertion", i, j - 1, None, hyp_words[j - 1]))
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            errors.append(AlignedError("deletion", i - 1, j, ref_words[i - 1], None))
            i -= 1
        else:
            i = max(0, i - 1)
            j = max(0, j - 1)

    errors.reverse()
    return errors


def context_of(ref_words: list[str], index: int, width: int = 3) -> str:
    """Quote the reference around *index*, bracketing the word that went wrong."""
    start = max(0, index - width)
    end = min(len(ref_words), index + width + 1)
    parts = list(ref_words[start:end])
    marked = index - start
    if 0 <= marked < len(parts):
        parts[marked] = f"[{parts[marked]}]"
    prefix = "… " if start > 0 else ""
    suffix = " …" if end < len(ref_words) else ""
    return f"{prefix}{' '.join(parts)}{suffix}"


def _is_fragment_of(part: str, whole: str) -> bool:
    """True when *part* is a leading or trailing piece of a longer *whole*."""
    return bool(part) and len(whole) > len(part) and (whole.startswith(part) or whole.endswith(part))


def classify(
    err: AlignedError, sentence: str, ref_words: list[str], hyp_words: list[str]
) -> str:
    """Bucket an edit by its likely source, so review effort lands on real defects.

    Neighbours matter: the normalizer splits "2:15 PM" into three reference
    tokens, so a lone "pm" or a one-letter fragment next to a digit belongs to
    the time it came from, not to the lexicon. Symbol and unit readings are
    checked before that fragment rule, since both legitimately sit next to digits.
    """
    ref = err.reference or ""
    hyp = err.hypothesis or ""
    both = f"{ref} {hyp}"

    if "$" in both:
        return "currency formatting"
    if "%" in both:
        return "percentage formatting"
    if _ALNUM_ID.search(ref) or _ALNUM_ID.search(hyp):
        return "alphanumeric ID"
    if _HAS_DIGIT.search(ref) or _HAS_DIGIT.search(hyp):
        return "number/time formatting"

    if err.type == "insertion":
        for symbol, readings in _SYMBOL_READINGS.items():
            if symbol in sentence and hyp in readings:
                return "symbol verbalization"

    for forms in _UNIT_FORMS:
        if ref in forms and hyp in forms:
            return "unit expansion"

    # A short fragment sitting inside a number run came from that number, even
    # though the fragment itself carries no digit ("2:15 PM" normalizes to the
    # three tokens "2 15 pm"). Checked before the compound rule, which would
    # otherwise claim "pm" the moment the ASR emits a bare "m".
    neighbours = ref_words[max(0, err.ref_index - 1) : err.ref_index + 2]
    if len(ref or hyp) <= _MAX_FRAGMENT_LEN and any(_HAS_DIGIT.search(w) for w in neighbours):
        if any(_ALNUM_ID.search(w) or len(w) >= _ID_RUN_LEN for w in neighbours):
            return "alphanumeric ID"
        return "number/time formatting"

    # A split or merged compound shows up as two edits: one substitution against
    # the fused form plus a stray fragment. Match the fragment to its counterpart
    # on the other side so both halves land in the same bucket.
    if err.type == "substitution" and (_is_fragment_of(hyp, ref) or _is_fragment_of(ref, hyp)):
        return "compound split/merge"
    if err.type == "insertion":
        for word in ref_words[max(0, err.ref_index - 1) : err.ref_index + 2]:
            if _is_fragment_of(hyp, word):
                return "compound split/merge"
    if err.type == "deletion":
        for word in hyp_words[max(0, err.hyp_index - 1) : err.hyp_index + 2]:
            if _is_fragment_of(ref, word):
                return "compound split/merge"

    # Capitalized mid-sentence in the written prompt → a name the voice may fumble.
    for token in sentence.split()[1:]:
        stripped = token.strip(".,:;!?\"'()").lower()
        if stripped and token[0].isupper() and stripped in {ref, hyp}:
            return "proper noun"
    return "lexical"


def describe(err: AlignedError) -> str:
    if err.type == "substitution":
        return f"Script word “{err.reference}” was heard as “{err.hypothesis}”."
    if err.type == "deletion":
        return f"Script word “{err.reference}” was not heard in the generated audio."
    return f"Extra word “{err.hypothesis}” was heard that is not in the script."


def load_sentences(manifest: Path) -> dict[str, str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    return {str(i["testcase_id"]): str(i["transcript"]) for i in items}


def prompt_sort_key(prompt_id: str) -> tuple[str, int]:
    match = re.match(r"([A-Za-z]*)(\d*)", prompt_id)
    if not match:
        return (prompt_id, 0)
    letters, digits = match.groups()
    return (letters, int(digits) if digits else 0)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=repo / "data" / "csv" / "falcon2_tts_wer_raw_fresh_tts_v1.csv",
        help="Per-utterance CSV written by tools/tts_wer.py",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Coval TTS dataset manifest supplying the original written sentence",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo / "data" / "csv" / "falcon2_tts_wer_word_errors.csv",
    )
    args = parser.parse_args()

    sentences = load_sentences(args.manifest)
    rows = list(csv.DictReader(args.raw.open(encoding="utf-8")))

    out_rows: list[dict[str, object]] = []
    for row in rows:
        ref_words = row["normalized_reference"].split()
        hyp_words = row["normalized_hypothesis"].split()
        errors = align(ref_words, hyp_words)

        counts = Counter(e.type for e in errors)
        expected = {
            "substitution": int(row["substitutions"]),
            "insertion": int(row["insertions"]),
            "deletion": int(row["deletions"]),
        }
        for kind, want in expected.items():
            if counts[kind] != want:
                raise SystemExit(
                    f"alignment drift on {row['voice']}/{row['prompt_id']}: "
                    f"{kind} {counts[kind]} != {want} recorded in {args.raw.name}"
                )

        sentence = sentences.get(row["prompt_id"], "")
        for err in errors:
            out_rows.append(
                {
                    "voice": row["voice"],
                    "prompt_id": row["prompt_id"],
                    "sentence": sentence,
                    "error_word": err.reference if err.reference is not None else err.hypothesis,
                    "error_type": err.type,
                    "heard_as": err.hypothesis or "(nothing)",
                    "description": describe(err),
                    "likely_cause": classify(err, sentence, ref_words, hyp_words),
                    "context_in_sentence": context_of(ref_words, err.ref_index),
                    "utterance_wer_pct": row["wer_percentage"],
                    "asr_model": row["asr_model"],
                    "norm_version": row["norm_version"],
                }
            )

    out_rows.sort(key=lambda r: (str(r["voice"]), prompt_sort_key(str(r["prompt_id"]))))

    fields = [
        "voice",
        "prompt_id",
        "sentence",
        "error_word",
        "error_type",
        "heard_as",
        "description",
        "likely_cause",
        "context_in_sentence",
        "utterance_wer_pct",
        "asr_model",
        "norm_version",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    by_type = Counter(str(r["error_type"]) for r in out_rows)
    by_cause = Counter(str(r["likely_cause"]) for r in out_rows)
    print(f"Wrote {args.out.relative_to(repo)} — {len(out_rows)} error rows from {len(rows)} utterances")
    print("  by type: " + ", ".join(f"{k}={v}" for k, v in by_type.most_common()))
    print("  by cause: " + ", ".join(f"{k}={v}" for k, v in by_cause.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
