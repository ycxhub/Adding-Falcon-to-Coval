# Copyright 2026 — Murf Falcon-2 provider adapter for latency-test
"""Env: MURF_API_KEY or TTS_API_KEY. Voice/model via MURF_VOICE / MURF_MODEL."""

from __future__ import annotations

import base64
import json
import os
import time
from urllib.parse import urlencode

import websockets

WS_URL = "wss://global.api.murf.ai/v1/speech/stream-input"
SAMPLE_RATE = int(os.environ.get("MURF_SAMPLE_RATE", "24000"))
MODEL = os.environ.get("MURF_MODEL", "falcon-2")
VOICE = os.environ.get("MURF_VOICE", "Gordon")
LOCALE = "en-US"


def _api_key() -> str:
    key = os.environ.get("MURF_API_KEY") or os.environ.get("TTS_API_KEY")
    if not key:
        raise RuntimeError("Set MURF_API_KEY or TTS_API_KEY")
    return key


def _strip_wav_header(chunk: bytes) -> bytes:
    if chunk.startswith(b"RIFF") and len(chunk) > 44:
        return chunk[44:]
    return chunk


async def synthesize(text: str) -> tuple[bytes, float, float, int]:
    """Return (pcm16_mono, t0, first_chunk_at, sample_rate)."""
    query = urlencode(
        {
            "api-key": _api_key(),
            "model": MODEL,
            "sample_rate": SAMPLE_RATE,
            "channel_type": "MONO",
            "format": "PCM",
        }
    )
    uri = f"{WS_URL}?{query}"

    pcm_parts: list[bytes] = []
    first_audio_chunk_at: float | None = None
    t0: float | None = None

    async with websockets.connect(uri) as ws:
        await ws.send(
            json.dumps({"voice_config": {"voiceId": VOICE, "locale": LOCALE}})
        )

        t0 = time.monotonic()
        await ws.send(json.dumps({"text": text, "end": True}))

        async for raw in ws:
            msg = json.loads(raw)
            audio_b64 = msg.get("audio") or ""
            if audio_b64:
                pcm = base64.b64decode(audio_b64)
                if not pcm_parts:
                    pcm = _strip_wav_header(pcm)
                if pcm and first_audio_chunk_at is None:
                    first_audio_chunk_at = time.monotonic()
                if pcm:
                    pcm_parts.append(pcm)
            if msg.get("final"):
                break

    if t0 is None or first_audio_chunk_at is None:
        raise RuntimeError("No audio chunks received from Murf")

    return b"".join(pcm_parts), t0, first_audio_chunk_at, SAMPLE_RATE
