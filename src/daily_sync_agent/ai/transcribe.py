"""Local speech-to-text with faster-whisper."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from daily_sync_agent.ai.audio_decode import load_audio_ffmpeg_mono_f32
from daily_sync_agent.ai.whisper_models import normalize_whisper_device

logger = logging.getLogger(__name__)

_HALLUCINATION_SILENCE_THRESHOLD_S = 2.0


def _transcribe_kwargs() -> dict[str, object]:
    """Conservative defaults that reduce trailing hallucinations on silence/noise."""
    return {
        "beam_size": 5,
        "condition_on_previous_text": False,
        "vad_filter": True,
        "word_timestamps": True,
        "hallucination_silence_threshold": _HALLUCINATION_SILENCE_THRESHOLD_S,
    }


def _cpu_fallback_compute_type(compute_type: str) -> str:
    """Map GPU-oriented compute types to CPU-safe CTranslate2 settings."""
    c = (compute_type or "default").strip().lower()
    if c in ("float16", "bfloat16"):
        return "int8"
    if c == "int8_float16":
        return "int8_float32"
    if c in ("int8", "int8_float32", "float32", "default"):
        return c
    return "default"


def _looks_like_cuda_runtime_missing(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "libcublas",
        "libcudnn",
        "libcudart",
        "libcuda",
        "libnvrtc",
        "cuda error",
        "no cuda gpus are available",
    )
    return any(n in msg for n in needles)


def _looks_like_cuda_oom_or_pressure(exc: BaseException) -> bool:
    """OOM or allocation failure when another process uses most VRAM."""
    m = str(exc).lower()
    if "out of memory" in m:
        return True
    if "failed to allocate" in m and "cuda" in m:
        return True
    if "cuda" in m and ("memory" in m or "oom" in m):
        return True
    return False


def _transcribe_once(
    audio_path: Path,
    *,
    model_size: str,
    device: str,
    compute_type: str,
    unload_model_after_task: bool = False,
) -> str:
    from faster_whisper import WhisperModel

    model = None
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        sr = model.feature_extractor.sampling_rate
        # Decode with ffmpeg + numpy so we never hit faster-whisper's PyAV path (can crash with
        # UnicodeDecodeError in av.error on some locales when resampling FLAC/etc.).
        audio = load_audio_ffmpeg_mono_f32(audio_path, sample_rate=sr)
        kwargs = _transcribe_kwargs()
        logger.debug(
            "Whisper transcribe model=%s device=%s compute_type=%s options=%s",
            model_size,
            device,
            compute_type,
            kwargs,
        )
        segments, _info = model.transcribe(audio, **kwargs)
        parts: list[str] = []
        for seg in segments:
            parts.append(seg.text.strip())
        return "\n".join(parts).strip()
    finally:
        if unload_model_after_task and model is not None:
            logger.debug("Releasing Whisper model resources after transcription task")
            del model
            gc.collect()


def transcribe_file(
    audio_path: Path,
    *,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "default",
    unload_model_after_task: bool = False,
) -> str:
    dev = normalize_whisper_device(device)
    try:
        return _transcribe_once(
            audio_path,
            model_size=model_size,
            device=dev,
            compute_type=compute_type,
            unload_model_after_task=unload_model_after_task,
        )
    except (RuntimeError, OSError) as e:
        if dev not in ("cuda", "auto"):
            raise
        recoverable = _looks_like_cuda_runtime_missing(e) or _looks_like_cuda_oom_or_pressure(e)
        if not recoverable:
            raise
        cpu_ct = _cpu_fallback_compute_type(compute_type)
        logger.warning(
            "GPU/CUDA issue (%s). Retrying Whisper on CPU (compute_type=%s).",
            e,
            cpu_ct,
        )
        return _transcribe_once(
            audio_path,
            model_size=model_size,
            device="cpu",
            compute_type=cpu_ct,
            unload_model_after_task=unload_model_after_task,
        )
