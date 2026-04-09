"""Whisper / CTranslate2 compute_type options by device (Preferences UI)."""

from __future__ import annotations

# (label, value) — values match faster_whisper / CTranslate2.
_COMPUTE_BY_DEVICE: dict[str, tuple[tuple[str, str], ...]] = {
    "auto": (
        ("Default (recommended)", "default"),
        ("int8", "int8"),
        ("float16", "float16"),
        ("int8_float16", "int8_float16"),
        ("float32", "float32"),
    ),
    "cpu": (
        ("Default (recommended)", "default"),
        ("int8 (fast on CPU)", "int8"),
        ("int8_float32", "int8_float32"),
        ("float32", "float32"),
    ),
    "cuda": (
        ("Default (recommended)", "default"),
        ("float16 (often best on GPU)", "float16"),
        ("int8_float16", "int8_float16"),
        ("int8", "int8"),
        ("float32", "float32"),
    ),
}


def compute_type_choices(device_key: str) -> tuple[tuple[str, str], ...]:
    return _COMPUTE_BY_DEVICE.get(device_key, _COMPUTE_BY_DEVICE["auto"])


def pick_compute_index_for_value(device_key: str, saved_value: str) -> int:
    choices = compute_type_choices(device_key)
    for i, (_, val) in enumerate(choices):
        if val == saved_value:
            return i
    return 0
