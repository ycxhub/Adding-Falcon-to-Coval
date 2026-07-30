# Copyright 2026 — adapted from Coval Benchmarks (Apache-2.0)
# https://github.com/coval-ai/benchmarks
"""Perceived TTFA helpers matching Coval's metrics/ttfa.py."""

from __future__ import annotations

import numpy as np

_AUDIBLE_RMS_THRESHOLD = 0.01
_FRAME_SECONDS = 0.010
_HOP_SECONDS = 0.001
_PCM16_BYTES = 2
_PCM16_FULL_SCALE = 32768.0


def first_audible_offset_ms(pcm: bytes, sample_rate: int) -> float | None:
    """Offset (ms) from PCM start to first audible sample, or None if silent."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if not pcm:
        return None
    if len(pcm) % _PCM16_BYTES != 0:
        raise ValueError("pcm length is not a whole number of frames")

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _PCM16_FULL_SCALE
    if samples.size == 0:
        return None

    frame_length = max(1, round(_FRAME_SECONDS * sample_rate))
    hop_length = max(1, round(_HOP_SECONDS * sample_rate))

    pad = frame_length // 2
    padded = np.pad(samples, pad, mode="edge")
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    rms = np.sqrt(np.mean((frames - frames.mean(axis=1, keepdims=True)) ** 2, axis=1))

    audible_frames = np.where(rms > _AUDIBLE_RMS_THRESHOLD)[0]
    if audible_frames.size == 0:
        return None

    return (float(audible_frames[0]) * hop_length / sample_rate) * 1000.0


def compute_ttfa(
    audio_synthesis_start: float,
    first_audio_chunk_at: float,
    leading_silence_offset_ms: float = 0.0,
) -> float:
    """Perceived TTFA in ms = arrival + leading silence."""
    if first_audio_chunk_at < audio_synthesis_start:
        raise ValueError("first_audio_chunk_at must be >= audio_synthesis_start")
    if leading_silence_offset_ms < 0:
        raise ValueError("leading_silence_offset_ms must be >= 0")
    arrival_ms = (first_audio_chunk_at - audio_synthesis_start) * 1000.0
    return arrival_ms + leading_silence_offset_ms


def strip_wav_header(chunk: bytes) -> bytes:
    """Drop a leading 44-byte RIFF/WAV header if present."""
    if chunk.startswith(b"RIFF") and len(chunk) > 44:
        return chunk[44:]
    return chunk


def summarize(values: list[float]) -> dict[str, float]:
    """Return median, mean, min, max for a list of latency samples."""
    if not values:
        raise ValueError("values must be non-empty")
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
