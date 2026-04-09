"""Decode arbitrary audio to mono float32 using ffmpeg (avoids PyAV resampler bugs on some systems)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def load_audio_ffmpeg_mono_f32(path: Path, *, sample_rate: int) -> np.ndarray:
    """PCM s16le mono at ``sample_rate``, converted to float32 [-1, 1]."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[:4000]
        raise RuntimeError(f"ffmpeg failed to decode audio (exit {proc.returncode}): {err}")
    if not proc.stdout:
        raise RuntimeError("ffmpeg produced no audio data")
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio
