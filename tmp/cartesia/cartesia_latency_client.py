# Copyright 2026 — Cartesia provider adapter for latency-test skill
"""Cartesia Sonic WebSocket adapter (Coval-aligned TTFA timing).

Env:
  CARTESIA_API_KEY or TTS_API_KEY — API key
  CARTESIA_MODEL — default sonic-3.5
  CARTESIA_VOICE — voice id
  CARTESIA_SAMPLE_RATE — default 44100
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid

import websockets

WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2026-03-01"
SAMPLE_RATE = int(os.environ.get("CARTESIA_SAMPLE_RATE", "44100"))
MODEL_ID = os.environ.get("CARTESIA_MODEL", "sonic-3.5")
VOICE_ID = os.environ.get(
    "CARTESIA_VOICE", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
)


def _api_key() -> str:
    key = os.environ.get("CARTESIA_API_KEY") or os.environ.get("TTS_API_KEY")
    if not key:
        raise RuntimeError("Set CARTESIA_API_KEY or TTS_API_KEY")
    return key


async def synthesize(text: str) -> tuple[bytes, float, float, int]:
    """Return (pcm16_mono, t0_monotonic, first_chunk_monotonic, sample_rate).

    TLS + WS upgrade happen before t0. Clock starts immediately before the
    generation request that carries the transcript.
    """
    uri = f"{WS_URL}?cartesia_version={CARTESIA_VERSION}"
    headers = {"Authorization": f"Bearer {_api_key()}"}

    pcm_parts: list[bytes] = []
    first_audio_chunk_at: float | None = None
    t0: float | None = None

    async with websockets.connect(uri, additional_headers=headers) as ws:
        context_id = str(uuid.uuid4())
        request = {
            "model_id": MODEL_ID,
            "transcript": text,
            "voice": {"mode": "id", "id": VOICE_ID},
            "language": "en",
            "context_id": context_id,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": SAMPLE_RATE,
            },
            "continue": False,
            "max_buffer_delay_ms": 0,
        }

        t0 = time.monotonic()
        await ws.send(json.dumps(request))

        async for raw in ws:
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "error":
                raise RuntimeError(
                    msg.get("message")
                    or msg.get("error")
                    or json.dumps(msg)
                )

            if msg_type == "chunk":
                data_b64 = msg.get("data") or ""
                if data_b64:
                    audio = base64.b64decode(data_b64)
                    if audio and first_audio_chunk_at is None:
                        first_audio_chunk_at = time.monotonic()
                    if audio:
                        pcm_parts.append(audio)
                if msg.get("done"):
                    break

            elif msg_type == "done":
                break

    if t0 is None or first_audio_chunk_at is None:
        raise RuntimeError("No audio chunks received from Cartesia")

    return b"".join(pcm_parts), t0, first_audio_chunk_at, SAMPLE_RATE
