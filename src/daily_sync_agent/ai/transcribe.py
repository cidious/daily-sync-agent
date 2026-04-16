"""Local speech-to-text with faster-whisper."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Callable

from daily_sync_agent.ai.audio_decode import load_audio_ffmpeg_mono_f32
from daily_sync_agent.ai.whisper_models import normalize_whisper_device

logger = logging.getLogger(__name__)

_HALLUCINATION_SILENCE_THRESHOLD_S = 2.0
_LOW_VRAM_RELEASE_TIMEOUT_S = 30.0
_LOW_VRAM_RELEASE_POLL_S = 0.25
_LOW_VRAM_VERIFY_MIN_USED_MIB = 512
_LOW_VRAM_RESIDUAL_TARGET_MIB = 128


def _nvidia_compute_app_pids() -> set[int] | None:
    """Best-effort set of PIDs currently listed by nvidia-smi compute apps query."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, PermissionError) as e:
        logger.debug("Low-VRAM cleanup: nvidia-smi unavailable for PID query: %s", e)
        return None
    except Exception as e:
        logger.debug("Low-VRAM cleanup: failed to query NVIDIA compute app PIDs: %s", e)
        return None

    if proc.returncode != 0:
        return None

    out: set[int] = set()
    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            out.add(int(line.split()[0]))
        except Exception:
            continue
    return out


def _descendant_pids(root_pid: int) -> set[int]:
    """Best-effort recursive PID tree for ``root_pid`` using ``ps``."""
    try:
        proc = subprocess.run(
            ["ps", "-e", "-o", "pid=,ppid="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as e:
        logger.debug("Low-VRAM cleanup: failed to inspect process tree: %s", e)
        return set()

    if proc.returncode != 0:
        return set()

    by_parent: dict[int, set[int]] = {}
    for raw_line in (proc.stdout or "").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except Exception:
            continue
        by_parent.setdefault(ppid, set()).add(pid)

    out: set[int] = set()
    stack = [root_pid]
    while stack:
        parent = stack.pop()
        children = by_parent.get(parent, set())
        for child in children:
            if child in out:
                continue
            out.add(child)
            stack.append(child)
    return out


def _tracked_gpu_process_pids(root_pid: int) -> set[int] | None:
    """Intersection of NVIDIA compute PIDs with this process subtree (root + descendants)."""
    gpu_pids = _nvidia_compute_app_pids()
    if gpu_pids is None:
        return None
    tracked = {root_pid}
    tracked.update(_descendant_pids(root_pid))
    return gpu_pids.intersection(tracked)


def _current_process_nvidia_vram_mib(*, pid: int | None = None) -> int | None:
    """Best-effort NVIDIA VRAM usage for the current PID, or ``None`` when unavailable."""
    target_pid = os.getpid() if pid is None else pid
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, PermissionError) as e:
        logger.debug("Low-VRAM cleanup: nvidia-smi unavailable: %s", e)
        return None
    except Exception as e:
        logger.debug("Low-VRAM cleanup: could not query nvidia-smi: %s", e)
        return None

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        logger.debug(
            "Low-VRAM cleanup: nvidia-smi returned %s (%s)",
            proc.returncode,
            stderr[:400] if stderr else "empty stderr",
        )
        return None

    used_mib = 0
    saw_pid = False
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            row_used_mib = int(parts[1].split()[0])
        except (ValueError, IndexError):
            continue
        if row_pid != target_pid:
            continue
        saw_pid = True
        used_mib += max(0, row_used_mib)
    return used_mib if saw_pid else 0


def _wait_for_whisper_vram_release(
    before_release_vram_mib: int | None,
    *,
    timeout_s: float = _LOW_VRAM_RELEASE_TIMEOUT_S,
    poll_interval_s: float = _LOW_VRAM_RELEASE_POLL_S,
) -> bool | None:
    """Wait until the current PID no longer appears to hold the Whisper model in NVIDIA VRAM."""
    if before_release_vram_mib is None:
        logger.debug("Low-VRAM cleanup: VRAM verification unavailable; proceeding after local cleanup")
        return None
    if before_release_vram_mib < _LOW_VRAM_VERIFY_MIN_USED_MIB:
        logger.debug(
            "Low-VRAM cleanup: current PID GPU usage already low before release (%s MiB)",
            before_release_vram_mib,
        )
        return True

    target_mib = min(_LOW_VRAM_RESIDUAL_TARGET_MIB, max(32, before_release_vram_mib // 4))
    # If almost all model VRAM is released and no child GPU workers remain, proceed early.
    relaxed_target_mib = min(512, max(target_mib, max(160, before_release_vram_mib // 8)))
    root_pid = os.getpid()
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        current_mib = _current_process_nvidia_vram_mib()
        if current_mib is None:
            logger.debug("Low-VRAM cleanup: could not re-check current PID GPU usage after Whisper cleanup")
            return None
        if current_mib <= target_mib:
            logger.debug(
                "Low-VRAM cleanup: current PID GPU usage dropped from %s MiB to %s MiB before next AI task",
                before_release_vram_mib,
                current_mib,
            )
            return True

        tracked_gpu_pids = _tracked_gpu_process_pids(root_pid)
        child_gpu_pids = {p for p in (tracked_gpu_pids or set()) if p != root_pid}
        released_ratio = 1.0 - (float(current_mib) / float(before_release_vram_mib))
        if tracked_gpu_pids is not None and not child_gpu_pids and current_mib <= relaxed_target_mib and released_ratio >= 0.85:
            logger.debug(
                "Low-VRAM cleanup: transcription/diarization GPU workers are gone and VRAM dropped from %s MiB to %s MiB; continuing",
                before_release_vram_mib,
                current_mib,
            )
            return True

        if time.monotonic() >= deadline:
            logger.warning(
                "Low-VRAM cleanup: current PID still uses about %s MiB of NVIDIA VRAM after waiting %.1fs for Whisper to unload",
                current_mib,
                timeout_s,
            )
            return False
        time.sleep(max(0.01, poll_interval_s))


def _best_effort_torch_cuda_release() -> None:
    """Release PyTorch CUDA allocator/cache when available (best-effort)."""
    try:
        import torch
    except Exception as e:
        logger.debug("Low-VRAM cleanup: torch import skipped during CUDA cache cleanup: %s", e)
        return

    try:
        if not torch.cuda.is_available():
            return
    except Exception as e:
        logger.debug("Low-VRAM cleanup: torch.cuda availability check failed: %s", e)
        return

    for action_name, action in (
        ("synchronize", getattr(torch.cuda, "synchronize", None)),
        ("empty_cache", getattr(torch.cuda, "empty_cache", None)),
        ("ipc_collect", getattr(torch.cuda, "ipc_collect", None)),
    ):
        if not callable(action):
            continue
        try:
            action()
            logger.debug("Low-VRAM cleanup: torch.cuda.%s executed", action_name)
        except Exception as e:
            logger.debug("Low-VRAM cleanup: torch.cuda.%s failed: %s", action_name, e)


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
    diarize: bool = False,
    hf_token: str = "",
    identify_speakers: bool = False,
    speaker_profiles_path: Path | None = None,
    speaker_names_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[str, list[dict] | None]:
    """Return (transcript_text, diarization_segments_or_none)."""
    from faster_whisper import WhisperModel

    started_at = time.monotonic()
    model = None
    audio = None
    segments = None
    info = None
    segment_count = 0
    transcript = ""
    diarization_result = None
    try:
        logger.debug(
            "Whisper transcription started model=%s device=%s compute_type=%s unload_after_task=%s diarize=%s",
            model_size,
            device,
            compute_type,
            unload_model_after_task,
            diarize,
        )
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        sr = model.feature_extractor.sampling_rate
        if progress_callback:
            progress_callback(f"Transcribing audio ({model_size} on {device})…")
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
        infer_started_at = time.monotonic()
        segments, info = model.transcribe(audio, **kwargs)
        parts: list[str] = []
        segments_data: list[dict] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                parts.append(text)
            segments_data.append({
                "text": text,
                "start": seg.start,
                "end": seg.end,
            })
            segment_count += 1
        transcript = "\n".join(parts).strip()

        # Optional diarization: extract speaker information
        if diarize and hf_token:
            try:
                if progress_callback:
                    progress_callback("Running speaker diarization…")
                from daily_sync_agent.ai.diarize import diarize_speakers, merge_diarization_with_transcript
                dia_result = diarize_speakers(audio_path, hf_token, device=device)
                diarization_result = dia_result.get("speakers")
                if identify_speakers and diarization_result and speaker_profiles_path and speaker_names_path:
                    try:
                        from daily_sync_agent.ai.speaker_id import (
                            identify_speakers_from_profiles,
                            rename_diarization_speakers,
                        )

                        local_to_name = identify_speakers_from_profiles(
                            audio_path,
                            diarization_result,
                            hf_token=hf_token,
                            device=device,
                            profiles_path=speaker_profiles_path,
                            names_path=speaker_names_path,
                        )
                        diarization_result = rename_diarization_speakers(diarization_result, local_to_name)
                    except Exception as e:
                        logger.warning(
                            "Speaker identification failed; continuing with diarization labels: %s",
                            e,
                        )
                # Merge speaker info with transcript
                transcript = merge_diarization_with_transcript(segments_data, diarization_result)
                logger.debug(
                    "Diarization merged with transcript speakers=%d",
                    dia_result.get("num_speakers_detected", 0),
                )
            except Exception as e:
                logger.warning("Diarization failed; continuing with plain transcript: %s", e)
                diarization_result = None

        infer_elapsed_s = time.monotonic() - infer_started_at
        logger.debug(
            "Whisper transcription finished model=%s device=%s compute_type=%s infer_s=%.3f segments=%d chars=%d diarized=%s",
            model_size,
            device,
            compute_type,
            infer_elapsed_s,
            segment_count,
            len(transcript),
            diarization_result is not None,
        )
        return transcript, diarization_result
    finally:
        if unload_model_after_task:
            logger.debug("Releasing Whisper model resources after transcription task")
            before_release_vram_mib = None
            if device in ("auto", "cuda"):
                before_release_vram_mib = _current_process_nvidia_vram_mib()
                if before_release_vram_mib is not None:
                    logger.debug(
                        "Low-VRAM cleanup: current PID uses %s MiB of NVIDIA VRAM before Whisper release",
                        before_release_vram_mib,
                    )
            segments = None
            info = None
            audio = None
            model = None
            gc.collect()
            if device in ("auto", "cuda"):
                _best_effort_torch_cuda_release()
                _wait_for_whisper_vram_release(before_release_vram_mib)
        elapsed_s = time.monotonic() - started_at
        logger.debug(
            "Whisper stage completed model=%s device=%s compute_type=%s total_s=%.3f",
            model_size,
            device,
            compute_type,
            elapsed_s,
        )


def transcribe_file(
    audio_path: Path,
    *,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "default",
    unload_model_after_task: bool = False,
    diarize: bool = False,
    hf_token: str = "",
    identify_speakers: bool = False,
    speaker_profiles_path: Path | None = None,
    speaker_names_path: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    dev = normalize_whisper_device(device)
    started_at = time.monotonic()
    logger.debug(
        "Whisper transcribe_file start audio=%s model=%s requested_device=%s resolved_device=%s compute_type=%s unload_after_task=%s diarize=%s",
        audio_path,
        model_size,
        device,
        dev,
        compute_type,
        unload_model_after_task,
        diarize,
    )
    if progress_callback:
        progress_callback(f"Loading Whisper model '{model_size}'…")
    try:
        text, _ = _transcribe_once(
            audio_path,
            model_size=model_size,
            device=dev,
            compute_type=compute_type,
            unload_model_after_task=unload_model_after_task,
            diarize=diarize,
            hf_token=hf_token,
            identify_speakers=identify_speakers,
            speaker_profiles_path=speaker_profiles_path,
            speaker_names_path=speaker_names_path,
            progress_callback=progress_callback,
        )
        logger.debug(
            "Whisper transcribe_file success model=%s device=%s compute_type=%s total_s=%.3f chars=%d",
            model_size,
            dev,
            compute_type,
            time.monotonic() - started_at,
            len(text),
        )
        return text
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
        if progress_callback:
            progress_callback(f"GPU issue — retrying on CPU (model '{model_size}')…")
        text, _ = _transcribe_once(
            audio_path,
            model_size=model_size,
            device="cpu",
            compute_type=cpu_ct,
            unload_model_after_task=unload_model_after_task,
            diarize=diarize,
            hf_token=hf_token,
            identify_speakers=identify_speakers,
            speaker_profiles_path=speaker_profiles_path,
            speaker_names_path=speaker_names_path,
            progress_callback=progress_callback,
        )
        logger.debug(
            "Whisper transcribe_file success after CPU fallback model=%s fallback_compute_type=%s total_s=%.3f chars=%d",
            model_size,
            cpu_ct,
            time.monotonic() - started_at,
            len(text),
        )
        return text
