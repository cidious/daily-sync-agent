"""Paths and user configuration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from daily_sync_agent.platform import is_linux, is_windows

try:
    import platformdirs
except ImportError:
    platformdirs = None  # type: ignore


def _get_cache_dir() -> Path:
    """Get platform-specific cache directory."""
    if platformdirs:
        return Path(platformdirs.user_cache_dir("daily-sync-agent"))
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "daily-sync-agent"


def _get_config_dir() -> Path:
    """Get platform-specific config directory."""
    if platformdirs:
        return Path(platformdirs.user_config_dir("daily-sync-agent"))
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "daily-sync-agent"


def _get_videos_dir() -> Path:
    """Get platform-specific videos directory."""
    if is_windows():
        # Windows: use Videos under user home
        return Path.home() / "Videos"
    # Linux: respect XDG_VIDEOS_DIR or fallback to ~/Videos
    return Path(os.environ.get("XDG_VIDEOS_DIR", Path.home() / "Videos"))


def _legacy_linux_config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "daily-sync-agent" / "config.json"


def log_dir() -> Path:
    p = _get_cache_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir() -> Path:
    p = _get_videos_dir() / "DailySyncRecordings"
    p.mkdir(parents=True, exist_ok=True)
    return p


def speaker_profiles_path() -> Path:
    p = _get_config_dir() / "speaker_profiles.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def speaker_names_path() -> Path:
    p = _get_config_dir() / "speaker_names.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class AppConfig:
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""  # empty: use first model from `ollama list` / GET /api/tags
    # Max time to wait for one LLM HTTP response (local CPU models can be slow for long outputs).
    ollama_request_timeout_s: float = 900.0
    transcribe_speech: bool = True
    summarize_transcript: bool = True
    summary_mode: str = "general"  # "general" | "daily_scrum"
    whisper_model: str = "base"
    whisper_device: str = "auto"  # auto, cpu, cuda
    whisper_compute_type: str = "default"
    # Free model memory aggressively after each AI task (useful on small VRAM GPUs).
    unload_models_after_task: bool = False
    # Pyannote speaker diarization (off by default; requires HF token and model download)
    diarize_speakers: bool = False
    huggingface_token: str = ""  # HF token for accessing Pyannote models
    # Optional cross-session speaker identification using saved embeddings + editable names.
    identify_speakers: bool = False
    ffmpeg_fps: int = 25
    display: str = ":0"
    # P1 reliability: disk space guard before recording
    min_free_disk_mb: int = 500
    # P1 reliability: per-stage AI timeouts (seconds; 0 = no timeout)
    transcribe_timeout_s: float = 3600.0
    # Preferences dialog geometry (None = use Qt default placement/size)
    prefs_window_x: int | None = None
    prefs_window_y: int | None = None
    prefs_window_w: int | None = None
    prefs_window_h: int | None = None
    # Last selected capture rectangle (screen coordinates), restored on startup.
    last_capture_x: int | None = None
    last_capture_y: int | None = None
    last_capture_w: int | None = None
    last_capture_h: int | None = None
    last_capture_window_id: str | None = None
    last_capture_title: str = ""

    @staticmethod
    def config_path() -> Path:
        return _get_config_dir() / "config.json"

    @classmethod
    def load(cls) -> AppConfig:
        path = cls.config_path()
        if not path.is_file() and is_linux():
            legacy = _legacy_linux_config_path()
            if legacy != path and legacy.is_file():
                path = legacy
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text())
            cur = asdict(cls())
            cur.update({k: v for k, v in data.items() if k in cur})
            wid = cur.get("last_capture_window_id")
            if wid is not None:
                cur["last_capture_window_id"] = str(wid)
            return cls(**cur)
        except Exception:
            return cls()

    def save(self) -> None:
        path = self.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
