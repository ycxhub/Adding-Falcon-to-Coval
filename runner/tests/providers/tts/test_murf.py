# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Murf TTS provider."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs

import pytest

from coval_bench.config import Settings
from coval_bench.providers.tts.murf import SAMPLE_RATE, MurfTTSProvider

from .conftest import FakeWebSocket, make_pcm_bytes, make_wav_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Envelopes mirror live API output: audio frames carry context_id and trace_id
# alongside the payload, and the terminator carries a `final` flag with no audio.
_TRACE_ID = "580a7a17-ce71-48ea-a6a3-d7661218a307"


def _chunk_msg(pcm: bytes) -> str:
    """Build a JSON audio envelope with base64-encoded PCM."""
    return json.dumps(
        {
            "audio": base64.b64encode(pcm).decode(),
            "context_id": None,
            "trace_id": _TRACE_ID,
        }
    )


def _final_msg() -> str:
    return json.dumps({"context_id": None, "final": True})


def _make_ws(pcm_chunks: list[bytes]) -> FakeWebSocket:
    """Return a FakeWebSocket that yields audio envelopes then a final message."""
    return FakeWebSocket([_chunk_msg(c) for c in pcm_chunks] + [_final_msg()])


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_murf_happy_path(fake_settings: Settings, tmp_path: Path) -> None:
    pcm = make_pcm_bytes(480)
    fake_ws = _make_ws([pcm, pcm, pcm, pcm])

    with patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=fake_ws):
        provider = MurfTTSProvider(fake_settings, model="falcon-2", voice="Natalie")
        result = await provider.synthesize("Hello from Murf")

    assert result.error is None, result.error
    assert result.ttfa_ms is not None and 0 < result.ttfa_ms < 10_000
    assert result.audio_path is not None and result.audio_path.exists()
    assert result.audio_path.stat().st_size > 0
    assert result.provider == "murf"
    assert result.model == "falcon-2"

    # Voice config precedes the text so session setup stays out of TTFA.
    assert len(fake_ws.sent) == 2
    voice_config = json.loads(fake_ws.sent[0])
    assert voice_config["voice_config"]["voiceId"] == "Natalie"
    assert voice_config["voice_config"]["locale"] == "en-US"

    text_msg = json.loads(fake_ws.sent[1])
    assert text_msg["text"] == "Hello from Murf"
    assert text_msg["end"] is True

    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_murf_all_models(fake_settings: Settings) -> None:
    for model in MurfTTSProvider._VALID_MODELS:
        fake_ws = _make_ws([make_pcm_bytes(240)])
        with patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=fake_ws):
            provider = MurfTTSProvider(fake_settings, model=model, voice="Natalie")
            result = await provider.synthesize("test")
        assert result.error is None, f"{model}: {result.error}"
        if result.audio_path is not None:
            result.audio_path.unlink()


def test_murf_handshake_url_carries_audio_contract(fake_settings: Settings) -> None:
    provider = MurfTTSProvider(fake_settings, model="falcon-2", voice="Natalie")
    base, _, query = provider._handshake_url().partition("?")
    params = parse_qs(query)

    assert base == "wss://global.api.murf.ai/v1/speech/stream-input"
    assert params["model"] == ["falcon-2"]
    assert params["sample_rate"] == ["24000"]
    assert params["channel_type"] == ["MONO"]
    assert params["format"] == ["PCM"]
    # 24 kHz mono PCM is what finalize_tts_result assumes when it writes the WAV.
    assert params["sample_rate"] == [str(SAMPLE_RATE)]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_murf_strips_wav_header_from_first_chunk(fake_settings: Settings) -> None:
    """A RIFF-prefixed first chunk is unwrapped so the WAV holds only PCM."""
    frames = 480
    fake_ws = FakeWebSocket([_chunk_msg(make_wav_bytes(frames)), _final_msg()])

    with patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=fake_ws):
        provider = MurfTTSProvider(fake_settings, model="falcon-2", voice="Natalie")
        result = await provider.synthesize("headered")

    assert result.error is None
    assert result.audio_path is not None
    # 44-byte RIFF header + 44-byte header written by finalize == input WAV size.
    assert result.audio_path.stat().st_size == 44 + frames * 2
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_murf_empty_audio(fake_settings: Settings) -> None:
    """Server sends final with no chunks — no audio_path, no ttfa."""
    fake_ws = FakeWebSocket([_final_msg()])

    with patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=fake_ws):
        provider = MurfTTSProvider(fake_settings, model="falcon-2", voice="Natalie")
        result = await provider.synthesize("silence")

    assert result.error is None
    assert result.audio_path is None
    assert result.ttfa_ms is None


@pytest.mark.asyncio
async def test_murf_priming_chunk_starts_the_clock(fake_settings: Settings) -> None:
    """The stream opens with a single-sample priming chunk before the bulk audio.

    That 2-byte frame is what the live API sends first, and it is non-empty, so
    it is what TTFA is measured from; the leading-silence offset in
    finalize_tts_result is what keeps the reported value honest.
    """
    fake_ws = _make_ws([make_pcm_bytes(1), make_pcm_bytes(4093), make_pcm_bytes(4096)])

    with patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=fake_ws):
        provider = MurfTTSProvider(fake_settings, model="falcon-2", voice="Natalie")
        result = await provider.synthesize("primed")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    # Every sample is retained, priming frame included: 8190 samples + 44-byte header.
    assert result.audio_path.stat().st_size == 44 + 8190 * 2
    result.audio_path.unlink()


@pytest.mark.asyncio
async def test_murf_final_envelope_may_carry_audio(fake_settings: Settings) -> None:
    """Audio riding on the final envelope is kept, not dropped by the break.

    The live API sends a bare terminator, so this is defensive: it pins the
    ordering so a server-side change could not silently lose the last chunk.
    """
    payload = base64.b64encode(make_pcm_bytes(240)).decode()
    fake_ws = FakeWebSocket([json.dumps({"audio": payload, "final": True})])

    with patch("coval_bench.providers.tts.murf.ws_client.connect", return_value=fake_ws):
        provider = MurfTTSProvider(fake_settings, model="falcon-2", voice="Natalie")
        result = await provider.synthesize("test")

    assert result.error is None
    assert result.ttfa_ms is not None
    assert result.audio_path is not None
    result.audio_path.unlink()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_murf_connection_error(fake_settings: Settings) -> None:
    with patch(
        "coval_bench.providers.tts.murf.ws_client.connect",
        side_effect=OSError("connection refused"),
    ):
        provider = MurfTTSProvider(fake_settings, model="falcon-2", voice="Natalie")
        result = await provider.synthesize("hi")

    assert result.error is not None
    assert "connection refused" in result.error
    assert result.audio_path is None
    assert result.ttfa_ms is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_murf_name_and_model(fake_settings: Settings) -> None:
    p = MurfTTSProvider(fake_settings, model="falcon-2", voice="Natalie")
    assert p.name == "murf-falcon-2"
    assert p.model == "falcon-2"


def test_murf_rejects_unsupported_model(fake_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unsupported Murf model"):
        MurfTTSProvider(fake_settings, model="not-a-real-model", voice="Natalie")


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


def test_murf_missing_api_key() -> None:
    settings_no_key = Settings(
        database_url="postgresql://runner:password@localhost:5432/benchmarks",
        dataset_bucket="test-bucket",
        dataset_id="stt-v1",
        runner_sha="test",
        murf_api_key=None,
    )
    with pytest.raises(ValueError, match="murf_api_key"):
        MurfTTSProvider(settings_no_key, model="falcon-2", voice="Natalie")
