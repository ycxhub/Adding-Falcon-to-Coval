# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Murf TTS provider — WebSocket streaming.

Wire protocol:
  connect (api-key, model, sample_rate, channel_type, format on the query string)
  → send JSON voice config: {"voice_config": {"voiceId": ..., "locale": ...}}
  → send JSON text:         {"text": ..., "end": true}
  → recv JSON chunks:       {"audio": "<base64-pcm>", "context_id", "trace_id"}
  → recv JSON done:         {"context_id", "final": true}  — carries no audio

The server keeps the connection open after ``final``, so the receive loop has
to break on the flag; waiting for the peer to close hangs until timeout.

Auth:   ``api-key`` query parameter on the handshake
Output: 16-bit mono PCM (base64-decoded from JSON envelopes), 24 kHz
"""

from __future__ import annotations

import base64
import json
import time
from urllib.parse import urlencode

import structlog
import websockets.asyncio.client as ws_client

from coval_bench.config import Settings
from coval_bench.providers.base import TTSProvider, TTSResult
from coval_bench.providers.tts._common import finalize_tts_result

logger: structlog.BoundLogger = structlog.get_logger(__name__)

SAMPLE_RATE = 24000
# Global router: the handshake is routed to Murf's nearest region.
_WS_URL = "wss://global.api.murf.ai/v1/speech/stream-input"
_LOCALE = "en-US"
_RIFF_MAGIC = b"RIFF"
_WAV_HEADER_BYTES = 44


class MurfTTSProvider(TTSProvider):
    """Murf TTS provider using the Falcon 2 streaming WebSocket API."""

    _VALID_MODELS = frozenset({"falcon-2"})

    def __init__(self, settings: Settings, model: str, voice: str) -> None:
        if model not in self._VALID_MODELS:
            raise ValueError(
                f"Unsupported Murf model {model!r}. Valid: {sorted(self._VALID_MODELS)}"
            )
        self._model = model
        self._voice = voice

        api_key_secret = settings.murf_api_key
        if api_key_secret is None:
            raise ValueError("murf_api_key is required in Settings")
        self._api_key = api_key_secret.get_secret_value()

    @property
    def name(self) -> str:
        return f"murf-{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def _handshake_url(self) -> str:
        """Handshake URL carrying auth and the audio contract as query parameters."""
        query = urlencode(
            {
                "api-key": self._api_key,
                "model": self._model,
                "sample_rate": SAMPLE_RATE,
                "channel_type": "MONO",
                "format": "PCM",
            }
        )
        return f"{_WS_URL}?{query}"

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech via the Murf WebSocket and return a TTSResult."""
        audio_chunks: list[bytes] = []
        start: float | None = None
        first_chunk_at: float | None = None

        # Explicit locale keeps the voice off the server's auto-detect path.
        voice_config = json.dumps({"voice_config": {"voiceId": self._voice, "locale": _LOCALE}})
        # `end` closes the context, so the server flushes without waiting for
        # more input and the connection is free for the next synthesis.
        text_message = json.dumps({"text": text, "end": True})

        try:
            async with ws_client.connect(self._handshake_url()) as ws:
                # Voice setup is sent before t0 so session config stays out of TTFA.
                await ws.send(voice_config)

                start = time.monotonic()
                await ws.send(text_message)

                async for raw in ws:
                    msg = json.loads(raw)
                    audio_b64: str = msg.get("audio", "")
                    if audio_b64:
                        pcm = base64.b64decode(audio_b64)
                        if not audio_chunks:
                            pcm = _strip_wav_header(pcm)
                        if pcm:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            audio_chunks.append(pcm)
                    if msg.get("final"):
                        break

        except Exception as exc:
            logger.warning("murf_error", provider="murf", model=self._model, exc_info=exc)
            return finalize_tts_result(
                provider="murf",
                model=self._model,
                voice=self._voice,
                pcm=b"",
                sample_rate=SAMPLE_RATE,
                audio_synthesis_start=start,
                first_audio_chunk_at=first_chunk_at,
                error=str(exc),
            )

        return finalize_tts_result(
            provider="murf",
            model=self._model,
            voice=self._voice,
            pcm=b"".join(audio_chunks),
            sample_rate=SAMPLE_RATE,
            audio_synthesis_start=start,
            first_audio_chunk_at=first_chunk_at,
        )


def _strip_wav_header(chunk: bytes) -> bytes:
    """Drop a leading RIFF header so the assembled stream is raw PCM.

    ``format=PCM`` yields headerless audio, but ``format=WAV`` prefixes the
    first chunk with a 44-byte header. Tolerating both keeps a server-side
    default change from prepending header bytes to the measured audio.
    """
    if chunk.startswith(_RIFF_MAGIC) and len(chunk) > _WAV_HEADER_BYTES:
        return chunk[_WAV_HEADER_BYTES:]
    return chunk
