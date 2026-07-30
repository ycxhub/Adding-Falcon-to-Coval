---
name: latency-test
description: Measure perceived TTS TTFA latency for any provider using Coval Benchmarks methodology (arrival + leading silence). Use when the user asks to latency-test, measure TTFA, benchmark TTS latency, or compare provider first-audio latency.
disable-model-invocation: true
---

# latency-test

Measure **perceived TTFA** for any TTS provider using the principles from
[coval-ai/benchmarks](https://github.com/coval-ai/benchmarks)
([methodology](https://github.com/coval-ai/benchmarks/blob/main/docs/methodology.md)).

```
TTFA_ms = (first_audio_chunk_at − t0) × 1000
        + leading_silence_offset_ms
```

## Required inputs

Before writing or running anything, collect **all five**. Ask for any that are missing:

| Input | Example |
| --- | --- |
| Provider name | `murf`, `elevenlabs`, `cartesia` |
| Model name | `falcon-2`, `eleven_turbo_v2_5` |
| Voice name | `Natalie`, `Rachel` |
| API docs URL | streaming TTS docs for that provider |
| API key | secret — store in env / prompt locally; never commit or paste into logs/PRs |

Optional (ask only if useful; defaults in parentheses):

- Trial count (**10** minimum; default **10**)
- Sample rate if docs require it (provider default)
- Fixed prompt vs Coval `tts-v1` sample (default: sample **10** distinct prompts from [reference.md](reference.md))

## Workflow

Copy and track:

```
Latency test progress:
- [ ] 1. Inputs collected (provider, model, voice, docs URL, API key)
- [ ] 2. API docs fetched; streaming contract understood
- [ ] 3. Provider client implemented (t0 + first chunk + full PCM)
- [ ] 4. Warmup completed (handshake / HTTP pool outside t0)
- [ ] 5. ≥10 timed trials run
- [ ] 6. Stats reported (median, mean, min, max) for perceived TTFA
```

### 1. Collect inputs

Do not proceed until all five required fields are present. Prefer:

```bash
export TTS_API_KEY='...'   # never echo this
```

### 2. Read the API docs

Fetch the API docs URL. Determine:

- Transport: WebSocket streaming vs HTTP chunked
- Auth placement (header / query)
- When audio bytes first appear
- Output format (PCM16 mono preferred; convert if needed)
- Session/voice setup messages that must run **before** `t0`

### 3. Implement the client (Coval timing rules)

Hard rules from Coval — do not invent alternatives:

1. **Exclude pre-t0 setup.** TCP/TLS/WS upgrade, HTTP client warmup, and voice/session config RTT happen **before** starting the clock.
2. **`t0 = time.monotonic()`** immediately before the synthesis trigger (text send / HTTP request that starts generation).
3. **`first_audio_chunk_at`** = monotonic time when the **first non-empty audio payload** arrives.
4. Assemble the full stream into **raw signed 16-bit little-endian mono PCM** (strip WAV/RIFF headers if present).
5. Compute leading silence with `scripts/ttfa.py` (RMS **0.01**, **10 ms** frame, **1 ms** hop — same as Coval).
6. Perceived TTFA = arrival + offset. If offset cannot be computed, degrade to **arrival-only** and note it.

Reference implementation pattern (Murf-style WS):

```python
await ws.send(voice_config)          # BEFORE t0
t0 = time.monotonic()
await ws.send(text_message)          # synthesis trigger
# on first non-empty pcm:
first_audio_chunk_at = time.monotonic()
```

### 4. Warmup

Run **one discarded synthesis** (or HTTP `warmup()` equivalent) so handshake/pool cost is not in the measured trials. Then run the timed loop.

### 5. Run ≥10 trials

Use `scripts/run_latency_test.py` (or an equivalent temp script that imports `scripts/ttfa.py`).

Requirements:

- **N ≥ 10** successful trials (retry failures; report failure count separately)
- Prefer **10 distinct prompts** from the Coval `tts-v1` list in [reference.md](reference.md)
- Sequential calls are fine; do not parallelize if that would re-open cold connections mid-run for HTTP providers
- Persist per-trial: `arrival_ms`, `silence_ms`, `ttfa_ms`, prompt id, error

```bash
# From the skill directory (or pass absolute paths):
python scripts/run_latency_test.py \
  --provider-module /tmp/my_provider_client.py \
  --provider NAME --model MODEL --voice VOICE \
  --trials 10 \
  --out /tmp/ttfa_results.json
```

The provider module must expose:

```python
async def synthesize(text: str) -> tuple[bytes, float, float, int]:
    """Return (pcm16_mono, t0_monotonic, first_chunk_monotonic, sample_rate)."""
```

### 6. Report results

Print a concise summary. Always include **median, mean, min, max** for perceived `ttfa_ms`. Also show arrival and silence breakdown when available:

```
Provider / model / voice: …
Trials: N ok / F failed
Prompts: coval tts-v1 sample (or fixed: "…")

Perceived TTFA (ms)
  median: …
  mean:   …
  min:    …
  max:    …

Breakdown (ms) — median
  arrival: …
  leading silence: …
```

Optionally attach the JSON/CSV of raw trials. Do not claim leaderboard parity unless the run used Coval’s full runner + pinned `runner_sha`.

## Security

- Never commit API keys, `.env` with secrets, or request logs containing keys
- Redact keys from command lines that might be captured in shell history when possible (env var preferred)
- Delete temp scripts that hard-code secrets after the run

## Additional resources

- Methodology detail + prompt list: [reference.md](reference.md)
- Onset + TTFA helpers: [scripts/ttfa.py](scripts/ttfa.py)
- Multi-trial harness: [scripts/run_latency_test.py](scripts/run_latency_test.py)
