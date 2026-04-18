"""Transcribe then summarize."""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Callable

from daily_sync_agent.ai.session_date import format_event_date_from_session_dir
from daily_sync_agent.ai.summarize import summarize_text
from daily_sync_agent.ai.transcribe import transcribe_file
from daily_sync_agent.settings import AppConfig
from daily_sync_agent.settings import speaker_names_path, speaker_profiles_path

logger = logging.getLogger(__name__)


def run_transcribe_and_summarize(
    audio_path: Path,
    out_dir: Path,
    config: AppConfig,
    *,
    progress_callback: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[Path, Path | None]:
    """Write transcript.txt and optional summary.txt next to recordings; return paths."""
    if config.unload_models_after_task:
        logger.debug("Low-VRAM mode enabled: model cleanup requested after this AI task")

    transcript_path = out_dir / "transcript.txt"

    # --- Transcription (with optional timeout) ---
    timeout_s = getattr(config, "transcribe_timeout_s", 3600.0)
    try:
        if timeout_s > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    transcribe_file,
                    audio_path,
                    model_size=config.whisper_model,
                    device=config.whisper_device,
                    compute_type=config.whisper_compute_type,
                    unload_model_after_task=config.unload_models_after_task,
                    diarize=config.diarize_speakers,
                    hf_token=config.huggingface_token,
                    identify_speakers=config.identify_speakers,
                    speaker_profiles_path=speaker_profiles_path(),
                    speaker_names_path=speaker_names_path(),
                    speaker_id_similarity_threshold=config.speaker_id_similarity_threshold,
                    speaker_id_min_ref_segment_s=config.speaker_id_min_ref_segment_s,
                    speaker_id_max_ref_segments=config.speaker_id_max_ref_segments,
                    progress_callback=progress_callback,
                )
                try:
                    text = future.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(
                        f"Transcription timed out after {timeout_s:.0f}s; "
                        "increase 'transcribe_timeout_s' in config or check AI resources."
                    )
        else:
            text = transcribe_file(
                audio_path,
                model_size=config.whisper_model,
                device=config.whisper_device,
                compute_type=config.whisper_compute_type,
                unload_model_after_task=config.unload_models_after_task,
                diarize=config.diarize_speakers,
                hf_token=config.huggingface_token,
                identify_speakers=config.identify_speakers,
                speaker_profiles_path=speaker_profiles_path(),
                speaker_names_path=speaker_names_path(),
                speaker_id_similarity_threshold=config.speaker_id_similarity_threshold,
                speaker_id_min_ref_segment_s=config.speaker_id_min_ref_segment_s,
                speaker_id_max_ref_segments=config.speaker_id_max_ref_segments,
                progress_callback=progress_callback,
            )
    except Exception as exc:
        # Save whatever we have so the session is not completely lost
        if not transcript_path.is_file():
            try:
                transcript_path.write_text(
                    f"[Transcription failed: {exc}]\n", encoding="utf-8"
                )
                logger.warning("Saved partial transcript marker to %s", transcript_path)
            except OSError as write_err:
                logger.debug("Could not write partial transcript marker: %s", write_err)
        raise

    transcript_path.write_text(text + "\n", encoding="utf-8")

    # --- Cancellation check between stages ---
    if cancel_check and cancel_check():
        logger.info("AI pipeline cancelled after transcription; skipping summarization")
        return transcript_path, None

    if not config.summarize_transcript:
        logger.debug("Summarization disabled by configuration; skipping summary generation")
        return transcript_path, None

    if progress_callback:
        progress_callback("Summarizing transcript…")

    summary = summarize_text(
        text,
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        summary_mode=config.summary_mode,
        event_date_hint=format_event_date_from_session_dir(out_dir),
        timeout_s=config.ollama_request_timeout_s,
        unload_model_after_task=config.unload_models_after_task,
    )
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    return transcript_path, summary_path
