# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""TTS Word Error Rate for Murf Falcon 2, scored the way Coval scores it.

Coval's TTS WER transcribes each provider's synthesized audio with one fixed
ASR model and scores that transcript against the synthesis input text, so the
number measures whether the speech is intelligible and complete. Scoring is
``whisper_normalizer``'s ``EnglishTextNormalizer`` on both sides, then a
word-level edit distance via ``jiwer`` -- see ``docs/methodology.md`` upstream.

``compute_wer`` below mirrors upstream ``coval_bench.metrics.wer`` (including
``NORM_VERSION`` and the empty-reference convention) so numbers produced here
are directly comparable to leaderboard rows. This repo packages only the Murf
provider, so the upstream module is not importable; it is used when present.

ASR backends:
  ``whisper-1``       Coval's fixed transcriber, via the OpenAI API. Requires
                      ``OPENAI_API_KEY``. Use this for leaderboard parity.
  ``faster-whisper``  Local CTranslate2 Whisper. ``whisper-1`` is Whisper
                      large-v2, so ``--local-model large-v2`` is the closest
                      offline stand-in when no OpenAI key is available.

Audio comes from WAVs on disk (``--audio-dir``) or fresh Murf synthesis
(``--synthesize``, requires ``MURF_API_KEY``).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import wave
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

# Bumped upstream on any behavioural change to normalization; recorded on every
# row so a result can be attributed to the pipeline revision that produced it.
NORM_VERSION: Literal["2"] = "2"

WordErrorType = Literal["substitution", "insertion", "deletion"]
WORD_ERROR_TYPES: tuple[WordErrorType, ...] = get_args(WordErrorType)

_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    """Canonical normalization: ``whisper_normalizer``'s ``EnglishTextNormalizer``."""
    normalized: str = _normalizer(text)
    return normalized


@dataclass(frozen=True)
class WordError:
    type: WordErrorType
    reference: str | None
    hypothesis: str | None


@dataclass(frozen=True)
class WERResult:
    wer: float
    wer_percentage: float
    incorrect_words: list[WordError]
    normalized_reference: str
    normalized_hypothesis: str
    norm_version: str = NORM_VERSION

    @property
    def error_percentages(self) -> dict[str, float]:
        """Split ``wer_percentage`` into per-type points that sum back to it."""
        share = self.wer_percentage / len(self.incorrect_words) if self.incorrect_words else 0.0
        counts = Counter(e.type for e in self.incorrect_words)
        return {f"wer_{t}s_pct": counts[t] * share for t in WORD_ERROR_TYPES}


def _build_word_errors(ref_words: list[str], hyp_words: list[str]) -> list[WordError]:
    """Build the incorrect-words list via a DP backtrace over the edit matrix."""
    n, m = len(ref_words), len(hyp_words)
    d: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for ii in range(n + 1):
        d[ii][0] = ii
    for jj in range(m + 1):
        d[0][jj] = jj
    for ii in range(1, n + 1):
        for jj in range(1, m + 1):
            if ref_words[ii - 1] == hyp_words[jj - 1]:
                d[ii][jj] = d[ii - 1][jj - 1]
            else:
                d[ii][jj] = 1 + min(d[ii - 1][jj - 1], d[ii][jj - 1], d[ii - 1][jj])

    errors: list[WordError] = []
    ii, jj = n, m
    while ii > 0 or jj > 0:
        if ii > 0 and jj > 0 and ref_words[ii - 1] == hyp_words[jj - 1]:
            ii -= 1
            jj -= 1
        elif ii > 0 and jj > 0 and d[ii][jj] == d[ii - 1][jj - 1] + 1:
            errors.append(WordError("substitution", ref_words[ii - 1], hyp_words[jj - 1]))
            ii -= 1
            jj -= 1
        elif jj > 0 and d[ii][jj] == d[ii][jj - 1] + 1:
            errors.append(WordError("insertion", None, hyp_words[jj - 1]))
            jj -= 1
        elif ii > 0 and d[ii][jj] == d[ii - 1][jj] + 1:
            errors.append(WordError("deletion", ref_words[ii - 1], None))
            ii -= 1
        else:
            # Safety: unreachable for a well-formed matrix, but avoid a spin.
            ii = max(0, ii - 1)
            jj = max(0, jj - 1)

    errors.reverse()
    return errors


def compute_wer(
    reference: str,
    hypothesis: str,
    *,
    normalizer: Callable[[str], str] | None = None,
) -> WERResult:
    """Compute Word Error Rate between *reference* and *hypothesis*."""
    _norm = normalizer if normalizer is not None else normalize_text
    norm_ref = _norm(reference)
    norm_hyp = _norm(hypothesis)
    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()

    if not ref_words and not hyp_words:
        return WERResult(0.0, 0.0, [], norm_ref, norm_hyp)

    # jiwer returns inf for an empty reference; cap at the hypothesis word count
    # so callers always get a finite, comparable float.
    if not ref_words:
        empty_ref_wer = float(len(hyp_words))
        errors = [WordError("insertion", None, w) for w in hyp_words]
        return WERResult(empty_ref_wer, empty_ref_wer * 100.0, errors, norm_ref, norm_hyp)

    raw_wer: float = jiwer.wer(norm_ref, norm_hyp)
    return WERResult(
        wer=raw_wer,
        wer_percentage=raw_wer * 100.0,
        incorrect_words=_build_word_errors(ref_words, hyp_words),
        normalized_reference=norm_ref,
        normalized_hypothesis=norm_hyp,
    )


# ---------------------------------------------------------------------------
# ASR backends
# ---------------------------------------------------------------------------


class Whisper1Transcriber:
    """Coval's fixed transcriber: OpenAI ``whisper-1``."""

    name = "whisper-1"

    def __init__(self) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the whisper-1 backend")
        self._client = OpenAI(api_key=api_key)

    def transcribe(self, wav_path: Path) -> str:
        with wav_path.open("rb") as fh:
            resp = self._client.audio.transcriptions.create(
                model="whisper-1", file=fh, response_format="text", language="en"
            )
        return resp if isinstance(resp, str) else getattr(resp, "text", "")


class FasterWhisperTranscriber:
    """Local CTranslate2 Whisper stand-in for ``whisper-1`` (Whisper large-v2)."""

    # Whisper's reference decode: temperature 0 with fallback up the ladder when
    # the compression-ratio or logprob threshold trips. Keeping the ladder
    # matters here -- it is what suppresses the repetition loops Whisper falls
    # into on near-silent lead-in, so dropping it would charge TTS for an ASR
    # artifact that ``whisper-1`` itself recovers from.
    _TEMPERATURE_LADDER = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    def __init__(self, model_size: str = "large-v2", compute_type: str = "float32") -> None:
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper:{model_size}/{compute_type}"
        self._model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    def transcribe(self, wav_path: Path) -> str:
        # No conditioning on previous text keeps each transcript a function of
        # this clip alone, so a voice cannot inherit context from its neighbours.
        segments, _ = self._model.transcribe(
            str(wav_path),
            language="en",
            beam_size=5,
            temperature=self._TEMPERATURE_LADDER,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# ---------------------------------------------------------------------------
# Audio sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Utterance:
    voice: str
    family: str
    prompt_id: str
    text: str
    wav_path: Path


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def load_from_disk(audio_dir: Path, metadata_csv: Path) -> list[Utterance]:
    """Read the WAVs already captured in this repo, using the CSV for the text."""
    rows: list[Utterance] = []
    with metadata_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            voice = row["voice"]
            wav = audio_dir / f"{voice}.wav"
            if not wav.exists():
                continue
            rows.append(
                Utterance(
                    voice=voice,
                    # Ken is the Gen2 control captured alongside the Falcon 2 pool.
                    family="Gen2" if voice == "Ken" else "Falcon 2",
                    prompt_id="sample-01",
                    text=row["text"],
                    wav_path=wav,
                )
            )
    return rows


def synthesize(voices: list[str], prompts: list[tuple[str, str]], out_dir: Path) -> list[Utterance]:
    """Synthesize every (voice, prompt) pair through the Falcon 2 provider."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner" / "src"))
    from coval_bench.config import Settings
    from coval_bench.providers.tts.murf import SAMPLE_RATE, MurfTTSProvider

    settings = Settings()
    if settings.murf_api_key is None:
        raise RuntimeError("MURF_API_KEY is required for --synthesize")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Utterance] = []

    async def run() -> None:
        for voice in voices:
            provider = MurfTTSProvider(settings, model="falcon-2", voice=voice)
            for prompt_id, text in prompts:
                result = await provider.synthesize(text)
                if getattr(result, "error", None):
                    print(f"  {voice}/{prompt_id}: synthesis failed: {result.error}")
                    continue
                wav = out_dir / f"{voice}__{prompt_id}.wav"
                with wave.open(str(wav), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(result.audio)
                rows.append(Utterance(voice, "Falcon 2", prompt_id, text, wav))
                print(f"  {voice}/{prompt_id}: {_wav_duration_sec(wav):.2f}s")

    asyncio.run(run())
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_per_utterance(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "voice",
        "family",
        "prompt_id",
        "asr_model",
        "norm_version",
        "wer",
        "wer_percentage",
        "ref_words",
        "hyp_words",
        "substitutions",
        "insertions",
        "deletions",
        "duration_sec",
        "normalized_reference",
        "normalized_hypothesis",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def write_summary(path: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_voice: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_voice.setdefault((str(row["voice"]), str(row["family"])), []).append(row)

    summary: list[dict[str, object]] = []
    for (voice, family), group in by_voice.items():
        wers = [float(r["wer_percentage"]) for r in group]
        summary.append(
            {
                "voice": voice,
                "family": family,
                "n_utterances": len(group),
                "wer_mean_pct": round(statistics.fmean(wers), 4),
                "wer_median_pct": round(statistics.median(wers), 4),
                "wer_max_pct": round(max(wers), 4),
                "total_ref_words": sum(int(r["ref_words"]) for r in group),
                "total_substitutions": sum(int(r["substitutions"]) for r in group),
                "total_insertions": sum(int(r["insertions"]) for r in group),
                "total_deletions": sum(int(r["deletions"]) for r in group),
                "asr_model": group[0]["asr_model"],
                "norm_version": group[0]["norm_version"],
            }
        )

    summary.sort(key=lambda r: (str(r["family"]) != "Falcon 2", float(r["wer_mean_pct"]), str(r["voice"])))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def print_table(summary: list[dict[str, object]]) -> None:
    header = f"{'Voice':<10} {'Family':<9} {'N':>3} {'WER %':>8} {'Sub':>4} {'Ins':>4} {'Del':>4} {'RefWords':>9}"
    print("\n" + header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['voice']:<10} {row['family']:<9} {row['n_utterances']:>3} "
            f"{float(row['wer_mean_pct']):>8.2f} {row['total_substitutions']:>4} "
            f"{row['total_insertions']:>4} {row['total_deletions']:>4} {row['total_ref_words']:>9}"
        )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asr",
        choices=["whisper-1", "faster-whisper", "auto"],
        default="auto",
        help="ASR backend. 'auto' picks whisper-1 when OPENAI_API_KEY is set.",
    )
    parser.add_argument("--local-model", default="large-v2", help="faster-whisper model size")
    parser.add_argument(
        "--compute-type",
        default="float32",
        help="faster-whisper compute type. int8 is fast but quantization damage "
        "makes large-v2 hallucinate, which shows up as phantom TTS errors.",
    )
    parser.add_argument("--audio-dir", type=Path, default=repo / "data" / "audio" / "full")
    parser.add_argument(
        "--metadata", type=Path, default=repo / "data" / "audio" / "sample_metadata.csv"
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Synthesize fresh audio through Falcon 2 instead of reading WAVs from disk.",
    )
    parser.add_argument("--voices", nargs="*", help="Voices to synthesize (with --synthesize)")
    parser.add_argument(
        "--prompts",
        type=Path,
        help="JSON prompt manifest: [{'testcase_id': ..., 'transcript': ...}] (with --synthesize)",
    )
    parser.add_argument("--out-dir", type=Path, default=repo / "data" / "csv")
    parser.add_argument("--tag", default="", help="Suffix for the output CSV filenames")
    args = parser.parse_args()

    if args.asr == "auto":
        backend = "whisper-1" if os.environ.get("OPENAI_API_KEY") else "faster-whisper"
    else:
        backend = args.asr
    transcriber = (
        Whisper1Transcriber()
        if backend == "whisper-1"
        else FasterWhisperTranscriber(args.local_model, args.compute_type)
    )
    print(f"ASR backend: {transcriber.name}  |  norm_version: {NORM_VERSION}")

    if args.synthesize:
        voices = args.voices or ["Natalie", "Miles"]
        if args.prompts:
            manifest = json.loads(args.prompts.read_text(encoding="utf-8"))
            items = manifest["items"] if isinstance(manifest, dict) else manifest
            prompts = [(str(i["testcase_id"]), str(i["transcript"])) for i in items]
        else:
            prompts = [("sample-01", "Your order shipped this morning and should arrive by Friday.")]
        print(f"Synthesizing {len(voices)} voices x {len(prompts)} prompts via Falcon 2")
        utterances = synthesize(voices, prompts, repo / "data" / "audio" / "wer_runs")
    else:
        utterances = load_from_disk(args.audio_dir, args.metadata)

    if not utterances:
        print("No audio to score.", file=sys.stderr)
        return 1

    print(f"Scoring {len(utterances)} utterance(s) from {args.audio_dir if not args.synthesize else 'fresh synthesis'}\n")

    rows: list[dict[str, object]] = []
    for utt in utterances:
        hypothesis = transcriber.transcribe(utt.wav_path)
        result = compute_wer(utt.text, hypothesis)
        counts = Counter(e.type for e in result.incorrect_words)
        rows.append(
            {
                "voice": utt.voice,
                "family": utt.family,
                "prompt_id": utt.prompt_id,
                "asr_model": transcriber.name,
                "norm_version": result.norm_version,
                "wer": round(result.wer, 6),
                "wer_percentage": round(result.wer_percentage, 4),
                "ref_words": len(result.normalized_reference.split()),
                "hyp_words": len(result.normalized_hypothesis.split()),
                "substitutions": counts["substitution"],
                "insertions": counts["insertion"],
                "deletions": counts["deletion"],
                "duration_sec": round(_wav_duration_sec(utt.wav_path), 3),
                "normalized_reference": result.normalized_reference,
                "normalized_hypothesis": result.normalized_hypothesis,
            }
        )
        flag = "" if result.wer == 0 else f"  <-- {result.normalized_hypothesis!r}"
        print(f"  {utt.voice:<10} {utt.prompt_id:<12} WER {result.wer_percentage:6.2f}%{flag}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    per_utt = args.out_dir / f"falcon2_tts_wer_raw{tag}.csv"
    summary_path = args.out_dir / f"falcon2_tts_wer_summary{tag}.csv"
    write_per_utterance(per_utt, rows)
    summary = write_summary(summary_path, rows)
    print_table(summary)

    overall = statistics.fmean(
        float(r["wer_percentage"]) for r in rows if r["family"] == "Falcon 2"
    )
    print(f"\nFalcon 2 mean WER across voices: {overall:.2f}%")
    print(f"Wrote {per_utt.relative_to(repo)}")
    print(f"Wrote {summary_path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
