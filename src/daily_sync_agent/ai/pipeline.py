"""Transcribe then summarize."""

from __future__ import annotations

from pathlib import Path

from daily_sync_agent.ai.session_date import format_event_date_from_session_dir
from daily_sync_agent.ai.summarize import summarize_text
from daily_sync_agent.ai.transcribe import transcribe_file
from daily_sync_agent.settings import AppConfig


def run_transcribe_and_summarize(
    audio_path: Path,
    out_dir: Path,
    config: AppConfig,
) -> tuple[Path, Path]:
    """Write transcript.txt and summary.txt next to recordings; return paths."""
    text = transcribe_file(
        audio_path,
        model_size=config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
    )
    transcript_path = out_dir / "transcript.txt"
    transcript_path.write_text(text + "\n", encoding="utf-8")

    summary = summarize_text(
        text,
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        summary_mode=config.summary_mode,
        event_date_hint=format_event_date_from_session_dir(out_dir),
        timeout_s=config.ollama_request_timeout_s,
    )
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    return transcript_path, summary_path
