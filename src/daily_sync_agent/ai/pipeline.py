"""Transcribe then summarize."""

from __future__ import annotations

import logging
from pathlib import Path

from daily_sync_agent.ai.session_date import format_event_date_from_session_dir
from daily_sync_agent.ai.summarize import summarize_text
from daily_sync_agent.ai.transcribe import transcribe_file
from daily_sync_agent.settings import AppConfig

logger = logging.getLogger(__name__)


def run_transcribe_and_summarize(
    audio_path: Path,
    out_dir: Path,
    config: AppConfig,
) -> tuple[Path, Path | None]:
    """Write transcript.txt and optional summary.txt next to recordings; return paths."""
    if config.unload_models_after_task:
        logger.debug("Low-VRAM mode enabled: model cleanup requested after this AI task")
    text = transcribe_file(
        audio_path,
        model_size=config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
        unload_model_after_task=config.unload_models_after_task,
    )
    transcript_path = out_dir / "transcript.txt"
    transcript_path.write_text(text + "\n", encoding="utf-8")

    if not config.summarize_transcript:
        logger.debug("Summarization disabled by configuration; skipping summary generation")
        return transcript_path, None

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
