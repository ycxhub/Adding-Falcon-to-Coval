# Tools

## `tts_wer.py` — TTS Word Error Rate

Scores Murf Falcon 2 on Coval's TTS WER metric: transcribe the synthesized audio
with one fixed ASR model, then score that transcript against the synthesis input
text, so the number measures whether the speech is intelligible and complete.

Scoring mirrors upstream `coval_bench.metrics.wer` — `EnglishTextNormalizer` on
both reference and hypothesis, then a word-level edit distance via `jiwer`, with
`NORM_VERSION` recorded on every row. This repo packages only the Murf provider,
so that module is not importable here; the port is used instead.

### Install

Coval's pinned scoring libraries, plus a local ASR backend:

```bash
uv venv .venv-wer --python 3.12
VIRTUAL_ENV=.venv-wer uv pip install \
  "jiwer==4.0.0" "whisper-normalizer==0.1.12" faster-whisper openai soundfile
```

### Score the WAVs in this repo

```bash
.venv-wer/bin/python tools/tts_wer.py \
  --asr faster-whisper --local-model large-v2 --compute-type float32 \
  --audio-dir data/audio/full --tag full_fp32
```

Writes `data/csv/falcon2_tts_wer_raw_<tag>.csv` (per utterance, including both
normalized strings) and `..._summary_<tag>.csv` (per voice).

**Stay at `float32`.** `int8` runs ~2x faster but quantization damage makes
large-v2 hallucinate on these clips — Korean/German tokens and repetition loops
that score as TTS errors the provider did not make.

### Leaderboard parity

Coval transcribes with OpenAI `whisper-1`. With a key set, `--asr auto` selects
it:

```bash
export OPENAI_API_KEY=...
.venv-wer/bin/python tools/tts_wer.py --audio-dir data/audio/full --tag whisper1
```

Local large-v2 is the closest offline stand-in (`whisper-1` is Whisper
large-v2), but decoding parameters differ, so numbers are indicative rather than
leaderboard-identical.

### Synthesize fresh audio

Ten reference words from a single prompt only bounds WER. To run a real prompt
set, synthesize through the provider (needs `MURF_API_KEY`):

```bash
export MURF_API_KEY=...
.venv-wer/bin/python tools/tts_wer.py --synthesize \
  --voices Natalie Miles Gordon Amara Madison Tyler Will Grant Judith \
  --prompts path/to/tts-v1.json --tag tts_v1
```

`--prompts` takes a Coval dataset manifest (or a bare list) of
`{"testcase_id", "transcript"}` items; WAVs land in `data/audio/wer_runs/`.
