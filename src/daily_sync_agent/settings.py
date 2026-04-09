"""Paths and user configuration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


def _xdg_cache() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))


def _xdg_config() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def log_dir() -> Path:
    p = _xdg_cache() / "daily-sync-agent" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir() -> Path:
    base = Path(os.environ.get("XDG_VIDEOS_DIR", Path.home() / "Videos"))
    p = base / "DailySyncRecordings"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class AppConfig:
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""  # empty: use first model from `ollama list` / GET /api/tags
    # Max time to wait for one LLM HTTP response (local CPU models can be slow for long outputs).
    ollama_request_timeout_s: float = 900.0
    summary_mode: str = "general"  # "general" | "daily_scrum"
    whisper_model: str = "base"
    whisper_device: str = "auto"  # auto, cpu, cuda
    whisper_compute_type: str = "default"
    ffmpeg_fps: int = 25
    display: str = ":0"
    # Preferences dialog geometry (None = use Qt default placement/size)
    prefs_window_x: int | None = None
    prefs_window_y: int | None = None
    prefs_window_w: int | None = None
    prefs_window_h: int | None = None

    @staticmethod
    def config_path() -> Path:
        return _xdg_config() / "daily-sync-agent" / "config.json"

    @classmethod
    def load(cls) -> AppConfig:
        path = cls.config_path()
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text())
            cur = asdict(cls())
            cur.update({k: v for k, v in data.items() if k in cur})
            return cls(**cur)
        except Exception:
            return cls()

    def save(self) -> None:
        path = self.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
