"""faster-whisper model names: presets plus models found in the Hugging Face cache."""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

# Sizes listed in faster_whisper WhisperModel docstring (plus common aliases).
_WHISPER_PRESETS: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "distil-small.en",
    "medium",
    "medium.en",
    "distil-medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-large-v3",
    "large-v3-turbo",
    "turbo",
)


def _scan_hf_faster_whisper_names() -> set[str]:
    names: set[str] = set()
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return names

    try:
        cache = scan_cache_dir()
        for repo in cache.repos:
            rid = getattr(repo, "repo_id", "") or ""
            if rid.startswith("Systran/faster-whisper-"):
                suffix = rid.split("Systran/faster-whisper-", 1)[-1]
                if suffix:
                    names.add(suffix)
    except Exception as e:
        logger.debug("scan_cache_dir for Whisper models: %s", e)
    return names


def _model_sort_key(name: str) -> tuple:
    try:
        return (0, _WHISPER_PRESETS.index(name), name)
    except ValueError:
        return (1, name)


def list_whisper_models_for_combo() -> tuple[str, ...]:
    """Preset sizes plus any ``Systran/faster-whisper-*`` repos in the HF hub cache."""
    merged = set(_WHISPER_PRESETS)
    merged.update(_scan_hf_faster_whisper_names())
    return tuple(sorted(merged, key=_model_sort_key))


def normalize_whisper_device(raw: str) -> str:
    d = (raw or "").strip().lower()
    if d in ("auto", "cpu", "cuda"):
        return d
    if d in ("gpu", "nvidia", "cuda:0"):
        return "cuda"
    return "auto"
