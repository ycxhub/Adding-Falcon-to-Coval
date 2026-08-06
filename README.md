# Adding Falcon to Coval

Murf **Falcon 2** Text-to-Speech provider integration for [Coval Benchmarks](https://github.com/coval-ai/benchmarks).

This repository packages the contribution work intended for upstream `coval-ai/benchmarks`: a WebSocket streaming TTS provider, registry entry (early access), config/env wiring, and unit tests.

## What’s included

| Path | Purpose |
| --- | --- |
| `runner/src/coval_bench/providers/tts/murf.py` | Falcon 2 WebSocket streaming provider |
| `runner/tests/providers/tts/test_murf.py` | Unit tests (fake WebSocket) |
| `runner/src/coval_bench/registries/models.py` | Early-access registry entry + voice pool |
| `runner/src/coval_bench/registries/provider_keys.py` | `murf` → `MURF_API_KEY` |
| `runner/src/coval_bench/config.py` | `murf_api_key` settings field |
| `runner/src/coval_bench/providers/tts/__init__.py` | Provider registration |
| `runner/tests/providers/tts/conftest.py` | Test settings fixture |
| `.env.example` | Documents `MURF_API_KEY` |
| `murf-falcon-2.patch` | Full diff against Coval `main` @ `03a52f3` |
| `data/` | Leading-silence matrix CSVs, charts, and full + trimmed WAVs |
| `tools/tts_wer.py` | TTS WER harness (Coval's metric: ASR round-trip + `jiwer`) |

## Apply upstream

From a fresh clone of [coval-ai/benchmarks](https://github.com/coval-ai/benchmarks):

```bash
git apply murf-falcon-2.patch
# or cherry-pick / copy the files under runner/ and .env.example
```

Set credentials:

```bash
export MURF_API_KEY=...
```

Smoke test (from the Coval runner package):

```bash
PYTHONPATH=src python -m coval_bench tts-smoke \
  --provider murf --model falcon-2 --voice Natalie
```

## Status notes

- Registered as **early access** (benchmarked, not public on the leaderboard) until concurrency / listing is ready.
- Wire protocol verified against the live Falcon 2 streaming API (`api-key` query auth, PCM JSON envelopes, break on `final`).
- Default demo voice in the registry is currently `Natalie` with pool `Natalie`, `Miles`.

## Attribution

Prepared for contribution to Coval Benchmarks by Chaitanya Y (Murf AI).
