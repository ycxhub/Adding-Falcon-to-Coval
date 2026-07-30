#!/usr/bin/env python3
"""Synthesize Cartesia latency-test prompts; save full + lead-in amplified WAVs.

Env: CARTESIA_API_KEY or TTS_API_KEY
Writes under data/cartesia/ (full/, leadin_amplified/, metadata).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "cartesia"
FULL_DIR = OUT / "full"
LEADIN_DIR = OUT / "leadin_amplified"
SKILL_SCRIPTS = ROOT / ".cursor" / "skills" / "latency-test" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
from ttfa import first_audible_offset_ms  # noqa: E402

WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2026-03-01"
SAMPLE_RATE = 44100
MODEL_ID = "sonic-3.5"
VOICE_ID = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
TARGET_PEAK = 0.5
SEED = 42

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


def _api_key() -> str:
    key = os.environ.get("CARTESIA_API_KEY") or os.environ.get("TTS_API_KEY")
    if not key:
        raise SystemExit("Set CARTESIA_API_KEY or TTS_API_KEY")
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


async def synthesize(text: str) -> bytes:
    uri = f"{WS_URL}?cartesia_version={CARTESIA_VERSION}"
    headers = {"Authorization": f"Bearer {_api_key()}"}
    pcm_parts: list[bytes] = []

    async with websockets.connect(uri, additional_headers=headers) as ws:
        request = {
            "model_id": MODEL_ID,
            "transcript": text,
            "voice": {"mode": "id", "id": VOICE_ID},
            "language": "en",
            "context_id": str(uuid.uuid4()),
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": SAMPLE_RATE,
            },
            "continue": False,
            "max_buffer_delay_ms": 0,
        }
        await ws.send(json.dumps(request))
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("message") or json.dumps(msg))
            if msg.get("type") == "chunk":
                data_b64 = msg.get("data") or ""
                if data_b64:
                    pcm_parts.append(base64.b64decode(data_b64))
                if msg.get("done"):
                    break
            elif msg.get("type") == "done":
                break

    pcm = b"".join(pcm_parts)
    if not pcm:
        raise RuntimeError("No audio received")
    return pcm


async def main() -> None:
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    LEADIN_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    selected = rng.sample(PROMPTS, 10)

    await synthesize(selected[0][1])
    print("warmup: ok", flush=True)

    rows: list[dict] = []
    for i, (pid, text) in enumerate(selected, start=1):
        t0 = time.monotonic()
        pcm = await synthesize(text)
        elapsed = (time.monotonic() - t0) * 1000
        silence_ms = first_audible_offset_ms(pcm, SAMPLE_RATE)

        full_path = FULL_DIR / f"{pid}.wav"
        write_wav(full_path, pcm, SAMPLE_RATE)

        leadin_pcm, _ = amplify_leadin(pcm, SAMPLE_RATE)
        leadin_path = LEADIN_DIR / f"{pid}_leadin_amplified.wav"
        if leadin_pcm:
            write_wav(leadin_path, leadin_pcm, SAMPLE_RATE)
        else:
            print(f"  WARNING: no lead-in for {pid}", flush=True)

        rows.append(
            {
                "prompt_id": pid,
                "leading_silence_ms": silence_ms,
                "full_bytes": len(pcm),
                "leadin_bytes": len(leadin_pcm),
                "full_path": str(full_path.relative_to(ROOT)),
                "leadin_path": str(leadin_path.relative_to(ROOT))
                if leadin_pcm
                else None,
                "text": text,
                "synth_wall_ms": elapsed,
            }
        )
        print(
            f"{i}/10 {pid}: silence={silence_ms:.1f} ms "
            f"full={full_path.name} leadin={leadin_path.name}",
            flush=True,
        )

    meta = {
        "provider": "cartesia",
        "model": MODEL_ID,
        "voice": VOICE_ID,
        "sample_rate": SAMPLE_RATE,
        "seed": SEED,
        "leadin_target_peak": TARGET_PEAK,
        "trials": rows,
    }
    (OUT / "sample_metadata.json").write_text(json.dumps(meta, indent=2))
    with (OUT / "sample_metadata.csv").open("w") as f:
        f.write("prompt_id,leading_silence_ms,full_bytes,leadin_bytes,text\n")
        for r in rows:
            text = r["text"].replace('"', '""')
            f.write(
                f'{r["prompt_id"]},{r["leading_silence_ms"]},'
                f'{r["full_bytes"]},{r["leadin_bytes"]},"{text}"\n'
            )
    print(f"Wrote {OUT / 'sample_metadata.json'}")


if __name__ == "__main__":
    asyncio.run(main())
