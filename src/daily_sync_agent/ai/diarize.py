"""Speaker diarization with Pyannote 3.1 (optional feature).

Audio is decoded to a 16 kHz mono waveform with ffmpeg and handed to Pyannote
as an in-memory ``{"waveform": Tensor, "sample_rate": int}`` dict so
**torchcodec is never used** (avoids libnppicc / CUDA-toolkit link errors).
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000  # Pyannote is highly sensitive to sample rate
_TORCHCODEC_WARNING_RE = r"(?s).*torchcodec is not installed correctly.*"


def _suppress_known_pyannote_warnings() -> None:
    """Suppress known non-fatal pyannote torchcodec warning noise."""
    warnings.filterwarnings(
        "ignore",
        message=_TORCHCODEC_WARNING_RE,
        category=UserWarning,
    )


def _import_pyannote_pipeline():
    """Import Pyannote Pipeline while silencing known non-fatal torchcodec warnings."""
    try:
        with warnings.catch_warnings():
            _suppress_known_pyannote_warnings()
            mod = importlib.import_module("pyannote.audio")
        return mod.Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Pyannote not installed. Run: pip install pyannote.audio or use install-pyannote-models.sh"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Pyannote failed to initialize. Disable speaker diarization in Preferences or fix pyannote runtime dependencies."
        ) from exc


def _load_audio_as_waveform(audio_path: Path) -> dict:
    """Decode *any* audio to a 16 kHz mono float32 torch Tensor via ffmpeg.

    Returns the ``{"waveform": Tensor, "sample_rate": int}`` dict that
    Pyannote accepts directly — bypassing torchcodec completely.
    """
    import torch

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(audio_path),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(_SAMPLE_RATE),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"ffmpeg failed to decode audio for diarization: {err}")
    if not proc.stdout:
        raise RuntimeError("ffmpeg produced no audio data for diarization")

    # int16 PCM → float32 [-1, 1] → torch tensor shaped (1, samples)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(pcm).unsqueeze(0)  # (1, T)

    logger.debug(
        "Decoded audio for diarization: %s → %d samples (%.1f s) at %d Hz",
        audio_path,
        waveform.shape[1],
        waveform.shape[1] / _SAMPLE_RATE,
        _SAMPLE_RATE,
    )
    return {"waveform": waveform, "sample_rate": _SAMPLE_RATE}


def _annotation_from_diarize_output(diarization_output):
    """Return an Annotation-like object exposing itertracks() across pyannote versions."""
    if hasattr(diarization_output, "itertracks"):
        return diarization_output
    for attr_name in ("speaker_diarization", "annotation", "diarization"):
        maybe_annotation = getattr(diarization_output, attr_name, None)
        if maybe_annotation is not None and hasattr(maybe_annotation, "itertracks"):
            return maybe_annotation
    raise RuntimeError(
        "Unsupported diarization output type from pyannote pipeline (missing itertracks)."
    )


def _resolve_diarization_device(whisper_device: str) -> str:
    """Resolve diarization runtime device from Whisper preference (auto/cpu/cuda)."""
    import torch

    requested = (whisper_device or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("Diarization requested CUDA but no CUDA device is available; falling back to CPU")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pipeline_device_for_log(pipeline) -> str:
    """Best-effort pipeline device string for debug logs."""
    dev = getattr(pipeline, "device", None)
    if dev is None:
        return "unknown"
    try:
        return str(dev)
    except Exception:
        return repr(dev)


def _place_pipeline_on_device(pipeline, device: str):
    """Move pyannote pipeline to target torch device when supported."""
    import torch

    if not hasattr(pipeline, "to"):
        logger.debug(
            "Diarization pipeline has no .to(); using default device behavior pipeline_device=%s",
            _pipeline_device_for_log(pipeline),
        )
        return pipeline
    pipeline.to(torch.device(device))
    logger.debug(
        "Diarization pipeline placed on requested_device=%s pipeline_device=%s",
        device,
        _pipeline_device_for_log(pipeline),
    )
    return pipeline


def diarize_speakers(
    audio_path: Path,
    hf_token: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str = "auto",
) -> dict[str, object]:
    """Run Pyannote speaker diarization on audio.

    Returns a dict with:
    - ``speakers``: list of segments ``[{"start": float, "end": float, "speaker": str}, ...]``
    - ``num_speakers_detected``: estimated speaker count
    """
    Pipeline = _import_pyannote_pipeline()

    if not hf_token or not hf_token.strip():
        raise RuntimeError(
            "HuggingFace token required for Pyannote. Set token in Preferences → HuggingFace token."
        )

    started_at = time.monotonic()
    resolved_device = _resolve_diarization_device(device)

    logger.debug(
        "Diarization started audio=%s hf_token_len=%d num_speakers=%s min=%s max=%s requested_device=%s resolved_device=%s",
        audio_path,
        len(hf_token),
        num_speakers,
        min_speakers,
        max_speakers,
        device,
        resolved_device,
    )

    # Load audio as in-memory waveform — avoids torchcodec entirely.
    audio_input = _load_audio_as_waveform(audio_path)

    with warnings.catch_warnings():
        _suppress_known_pyannote_warnings()
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token.strip(),
        )
        pipeline = _place_pipeline_on_device(pipeline, resolved_device)
        diarization_output = pipeline(audio_input)

    diarization = _annotation_from_diarize_output(diarization_output)

    speakers_list: list[dict[str, object]] = []
    detected_speakers: set[str] = set()
    for segment, _track, speaker in diarization.itertracks(yield_label=True):
        speakers_list.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": str(speaker),
        })
        detected_speakers.add(str(speaker))

    elapsed_s = time.monotonic() - started_at
    logger.debug(
        "Diarization finished audio=%s speakers_detected=%d segments=%d elapsed_s=%.3f resolved_device=%s pipeline_device=%s",
        audio_path,
        len(detected_speakers),
        len(speakers_list),
        elapsed_s,
        resolved_device,
        _pipeline_device_for_log(pipeline),
    )

    return {
        "speakers": speakers_list,
        "num_speakers_detected": len(detected_speakers),
    }


def merge_diarization_with_transcript(
    transcript_segments: list[dict],
    diarization_speakers: list[dict],
) -> str:
    """Merge Whisper segments with speaker labels (time-based matching)."""
    if not diarization_speakers:
        return "\n".join(seg.get("text", "") for seg in transcript_segments if seg.get("text"))

    result_lines: list[str] = []
    for seg in transcript_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = seg.get("start", 0)
        end = seg.get("end", 0)

        # Find speaker for this segment (closest match by time overlap)
        matching_speaker = None
        for dia_seg in diarization_speakers:
            dia_start = dia_seg.get("start", 0)
            dia_end = dia_seg.get("end", 0)
            # Simple overlap heuristic: if segment center is within diarization segment
            seg_center = (start + end) / 2
            if dia_start <= seg_center <= dia_end:
                matching_speaker = dia_seg.get("speaker", "Unknown")
                break

        if matching_speaker:
            result_lines.append(f"[{matching_speaker}] {text}")
        else:
            result_lines.append(text)

    return "\n".join(result_lines).strip()
