#!/usr/bin/env python3
"""Inworld Realtime TTS-2 latency test + save full/leadin WAVs.

Endpoint: POST https://api.inworld.ai/tts/v1/voice:stream
Model: inworld-tts-2  Voice: Ashley (docs default)
Env: INWORLD_API_KEY or TTS_API_KEY (Base64 Basic credentials)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "inworld"
TMP_OUT = ROOT / "tmp" / "inworld"
FULL_DIR = OUT / "full"
LEADIN_DIR = OUT / "leadin_amplified"
SKILL_SCRIPTS = ROOT / ".cursor" / "skills" / "latency-test" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
from ttfa import (  # noqa: E402
    compute_ttfa,
    first_audible_offset_ms,
    strip_wav_header,
    summarize,
)

URL = "https://api.inworld.ai/tts/v1/voice:stream"
MODEL = "inworld-tts-2"
VOICE = "Ashley"
SAMPLE_RATE = 24000
TARGET_PEAK = 0.5
SEED = 42
TRIALS = 10

PROMPTS: list[tuple[str, str]] = [
    (
        "A1",
        "Your order #ORD-24589 shipped via UPS tracking 1Z999AA1234567890 "
        "and will arrive between 2:30 PM and 4:45 PM tomorrow.",
    ),
    (
        "A2",
        "There's a slight delay with your $347.89 order, "
        "but we expect it to ship by Friday afternoon.",
    ),
    (
        "A3",
        'Order status update: your items are now in "Quality Check" '
        "and should ship within 24-48 hours.",
    ),
    (
        "A4",
        "Hi Ms. Garcia, your appointment with Dr. Peterson is scheduled "
        "for Tuesday, March 5th at 10:30 AM.",
    ),
    (
        "A5",
        "I can reschedule you to either 11:15 AM on Wednesday or, "
        "let me check, 2:45 PM on Thursday.",
    ),
    (
        "A6",
        "Please arrive 10 minutes early and bring your insurance card and a photo ID.",
    ),
    (
        "A7",
        "Let's fix your Wi-Fi: unplug your router for exactly 30 seconds, "
        "then wait for the green light.",
    ),
    (
        "A8",
        "Your device is downloading update version 12.4.1, "
        "about 250 MB remaining, roughly 8 minutes left.",
    ),
    (
        "A9",
        "I'm seeing error code E-1047 here, which usually means, "
        "actually, let me guide you through the solution.",
    ),
    (
        "A10",
        "For security, please confirm the last 4 digits of the card "
        "ending in 8429 used for your recent purchase.",
    ),
]

_client: httpx.AsyncClient | None = None


def _api_key() -> str:
    key = os.environ.get("INWORLD_API_KEY") or os.environ.get("TTS_API_KEY")
    if not key:
        raise SystemExit("Set INWORLD_API_KEY or TTS_API_KEY")
    return key


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def amplify_leadin(pcm: bytes, sample_rate: int) -> tuple[bytes, float | None]:
    silence_ms = first_audible_offset_ms(pcm, sample_rate)
    if silence_ms is None or silence_ms <= 0:
        return b"", silence_ms
    n = int(round(silence_ms * sample_rate / 1000.0))
    samples = np.frombuffer(pcm, dtype="<i2")
    lead = samples[:n].astype(np.float64)
    if lead.size == 0:
        return b"", silence_ms
    peak = float(np.max(np.abs(lead)))
    if peak < 1.0:
        out = np.zeros_like(lead, dtype=np.int16)
    else:
        scale = (TARGET_PEAK * 32768.0) / peak
        out = np.clip(lead * scale, -32768, 32767).astype(np.int16)
    return out.tobytes(), silence_ms


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={
                "Authorization": f"Basic {_api_key()}",
                "Content-Type": "application/json",
            },
            http2=False,
        )
    return _client


async def synthesize(text: str) -> tuple[bytes, float, float, int]:
    """Return (pcm16_mono, t0, first_chunk_at, sample_rate).

    Shared HTTP client is warmed outside t0. Clock starts immediately before
    the POST that triggers synthesis.
    """
    client = await get_client()
    payload = {
        "text": text,
        "voiceId": VOICE,
        "modelId": MODEL,
        "audioConfig": {
            "audioEncoding": "PCM",
            "sampleRateHertz": SAMPLE_RATE,
        },
        "deliveryMode": "BALANCED",
    }

    pcm_parts: list[bytes] = []
    first_audio_chunk_at: float | None = None
    buffer = ""
    decoder = json.JSONDecoder()

    def _consume_objects() -> None:
        nonlocal buffer, first_audio_chunk_at
        while True:
            s = buffer.lstrip()
            if not s:
                buffer = ""
                return
            try:
                msg, end = decoder.raw_decode(s)
            except json.JSONDecodeError:
                buffer = s
                return
            buffer = s[end:]

            if msg.get("error"):
                raise RuntimeError(json.dumps(msg["error"]))

            result = msg.get("result") or {}
            b64 = result.get("audioContent") or ""
            if not b64:
                continue
            audio = strip_wav_header(base64.b64decode(b64))
            if audio and first_audio_chunk_at is None:
                first_audio_chunk_at = time.monotonic()
            if audio:
                pcm_parts.append(audio)

    t0 = time.monotonic()
    async with client.stream("POST", URL, json=payload) as resp:
        if resp.status_code >= 400:
            body = await resp.aread()
            raise RuntimeError(f"HTTP {resp.status_code}: {body[:500]!r}")

        async for raw_bytes in resp.aiter_bytes():
            if not raw_bytes:
                continue
            buffer += raw_bytes.decode("utf-8")
            _consume_objects()

        rem = buffer.strip()
        if rem:
            raise RuntimeError(f"Trailing unparsed stream data: {rem[:200]!r}")

    if first_audio_chunk_at is None:
        raise RuntimeError("No audio chunks received from Inworld")

    return b"".join(pcm_parts), t0, first_audio_chunk_at, SAMPLE_RATE


async def main() -> None:
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    LEADIN_DIR.mkdir(parents=True, exist_ok=True)
    TMP_OUT.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    selected = rng.sample(PROMPTS, TRIALS)

    # Warm shared HTTP client + one discarded synthesis
    await get_client()
    await synthesize(selected[0][1])
    print("warmup: ok", flush=True)

    trials: list[dict] = []
    failures: list[dict] = []

    for i, (pid, text) in enumerate(selected, start=1):
        try:
            pcm, t0, first_at, sr = await synthesize(text)
            if len(pcm) % 2:
                pcm = pcm[: len(pcm) - 1]

            arrival_ms = (first_at - t0) * 1000.0
            silence_ms = first_audible_offset_ms(pcm, sr) if pcm else None
            ttfa_ms = compute_ttfa(
                t0, first_at, silence_ms if silence_ms is not None else 0.0
            )

            full_path = FULL_DIR / f"{pid}.wav"
            write_wav(full_path, pcm, sr)
            leadin_pcm, _ = amplify_leadin(pcm, sr)
            leadin_path = LEADIN_DIR / f"{pid}_leadin_amplified.wav"
            if leadin_pcm:
                write_wav(leadin_path, leadin_pcm, sr)

            row = {
                "trial": i,
                "prompt_id": pid,
                "arrival_ms": arrival_ms,
                "silence_ms": silence_ms,
                "ttfa_ms": ttfa_ms,
                "audio_bytes": len(pcm),
                "sample_rate": sr,
                "silence_degraded": silence_ms is None,
                "error": None,
            }
            trials.append(row)
            print(
                f"trial {i}/{TRIALS}: ttfa={ttfa_ms:.1f} ms "
                f"(arrival={arrival_ms:.1f}, silence={silence_ms}) "
                f"-> {full_path.name}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"trial": i, "prompt_id": pid, "error": str(exc)})
            print(f"trial {i}/{TRIALS}: FAIL {exc}", flush=True)

    ttfa_vals = [t["ttfa_ms"] for t in trials]
    arrival_vals = [t["arrival_ms"] for t in trials]
    silence_vals = [t["silence_ms"] for t in trials if t["silence_ms"] is not None]

    result = {
        "provider": "inworld",
        "model": MODEL,
        "voice": VOICE,
        "sample_rate": SAMPLE_RATE,
        "seed": SEED,
        "methodology": "coval-perceived-ttfa",
        "endpoint": URL,
        "encoding": "PCM",
        "deliveryMode": "BALANCED",
        "trials_ok": len(trials),
        "trials_failed": len(failures),
        "ttfa_ms": summarize(ttfa_vals) if ttfa_vals else None,
        "arrival_ms": summarize(arrival_vals) if arrival_vals else None,
        "silence_ms": summarize(silence_vals) if silence_vals else None,
        "trials": trials,
        "failures": failures,
        "leadin_target_peak": TARGET_PEAK,
    }

    results_path = OUT / "ttfa_results.json"
    results_path.write_text(json.dumps(result, indent=2))
    (TMP_OUT / "ttfa_results.json").write_text(json.dumps(result, indent=2))

    with (OUT / "sample_metadata.csv").open("w") as f:
        f.write("prompt_id,leading_silence_ms,full_bytes,ttfa_ms,arrival_ms,text\n")
        prompt_map = dict(selected)
        for t in trials:
            text = prompt_map[t["prompt_id"]].replace('"', '""')
            f.write(
                f'{t["prompt_id"]},{t["silence_ms"]},{t["audio_bytes"]},'
                f'{t["ttfa_ms"]},{t["arrival_ms"]},"{text}"\n'
            )

    print()
    print(f"Provider / model / voice: inworld / {MODEL} / {VOICE}")
    print(f"Trials: {result['trials_ok']} ok / {result['trials_failed']} failed")
    print("Prompts: coval tts-v1 sample (seed 42)")
    if result["ttfa_ms"]:
        s = result["ttfa_ms"]
        print("Perceived TTFA (ms)")
        print(f"  median: {s['median']:.2f}")
        print(f"  mean:   {s['mean']:.2f}")
        print(f"  min:    {s['min']:.2f}")
        print(f"  max:    {s['max']:.2f}")
    if result.get("arrival_ms") and result.get("silence_ms"):
        print("Breakdown (ms) — median")
        print(f"  arrival: {result['arrival_ms']['median']:.2f}")
        print(f"  leading silence: {result['silence_ms']['median']:.2f}")
    print(f"Audio: {FULL_DIR}")
    print(f"Lead-in: {LEADIN_DIR}")
    print(f"Results: {results_path} and {TMP_OUT / 'ttfa_results.json'}")

    if _client is not None:
        await _client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
