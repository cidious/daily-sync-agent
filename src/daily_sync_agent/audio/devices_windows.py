"""Windows audio device discovery using FFmpeg DirectShow device names."""

from __future__ import annotations

import logging
import re
import subprocess

from daily_sync_agent.audio.devices import AudioDevices, AudioMode

logger = logging.getLogger(__name__)
_DShowAudioLine = re.compile(r'"(?P<name>.+?)"\s+\(audio\)')
_STRONG_LOOPBACK_TOKENS = (
    "stereo mix",
    "loopback",
    "what u hear",
    "wave out",
    "waveout mix",
    "monitor of",
    "virtual-audio-capturer",
    "virtual audio capturer",
    "voicemeeter output",
    "cable output",
)
_LOOPBACK_HINT_TOKENS = ("speakers", "headphones", "digital output", "hdmi", "render")
_MIC_TOKENS = ("microphone", "mic", "line in", "input", "array")


def _list_dshow_audio_device_names() -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    names: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = _DShowAudioLine.search(line)
        if not match:
            continue
        name = match.group("name").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _looks_like_loopback(name: str) -> bool:
    lower = name.lower()
    if any(token in lower for token in _STRONG_LOOPBACK_TOKENS):
        return True
    # Some DirectShow drivers expose loopback devices as playback-endpoint names.
    has_playback_hint = any(token in lower for token in _LOOPBACK_HINT_TOKENS)
    has_mic_hint = any(token in lower for token in _MIC_TOKENS)
    return has_playback_hint and not has_mic_hint


def _pick_device_name(preferred: str | None, default_name: str, choices: list[tuple[str, str]]) -> str:
    names = [name for name, _ in choices]
    if not names:
        return ""

    def _match(requested: str | None) -> str:
        if not requested:
            return ""
        requested_lower = requested.lower()
        for candidate in names:
            if candidate.lower() == requested_lower:
                return candidate
        return ""

    if preferred:
        preferred_match = _match(preferred)
        if preferred_match:
            return preferred_match
    default_match = _match(default_name)
    if default_match:
        return default_match
    return names[0]


def _has_loopback_device(devices: AudioDevices) -> bool:
    return bool(devices.default_sink or devices.sinks)


def _has_source_device(devices: AudioDevices) -> bool:
    return bool(devices.default_source or devices.sources)


def windows_audio_mode_available(mode: AudioMode, devices: AudioDevices) -> bool:
    if mode == AudioMode.MONITOR:
        return _has_loopback_device(devices)
    if mode == AudioMode.MIC:
        return _has_source_device(devices)
    return _has_loopback_device(devices) and _has_source_device(devices)


def pick_windows_audio_mode(mode: AudioMode, devices: AudioDevices) -> tuple[AudioMode, str | None]:
    has_sink = _has_loopback_device(devices)
    has_source = _has_source_device(devices)

    if mode == AudioMode.MONITOR:
        if not has_sink:
            raise RuntimeError(
                "Windows system-output capture is unavailable on this machine. Enable a Stereo Mix/loopback device or choose Microphone only."
            )
        return (mode, None)

    if mode == AudioMode.MIC:
        if not has_source:
            raise RuntimeError("No Windows microphone/capture device found.")
        return (mode, None)

    if has_sink and has_source:
        return (mode, None)
    if has_source:
        return (
            AudioMode.MIC,
            "Windows loopback/Stereo Mix is unavailable, so recording will use Microphone only.",
        )
    if has_sink:
        return (
            AudioMode.MONITOR,
            "No Windows microphone device was detected, so recording will use Playback / system output only.",
        )
    raise RuntimeError(
        "No Windows audio capture devices found. Enable a microphone or loopback device and try again."
    )


def list_windows_devices() -> AudioDevices:
    names = _list_dshow_audio_device_names()
    sources = [(name, name) for name in names]
    sinks = [(name, name) for name in names if _looks_like_loopback(name)]
    default_source = next((name for name in names if not _looks_like_loopback(name)), names[0] if names else "")
    default_sink = sinks[0][0] if sinks else ""
    logger.debug(
        "Windows DirectShow devices discovered audio=%d loopback=%d default_source=%r default_sink=%r",
        len(sources),
        len(sinks),
        default_source,
        default_sink,
    )
    return AudioDevices(
        default_sink=default_sink,
        default_source=default_source,
        sinks=sinks,
        sources=sources,
    )


def resolve_windows_input(
    mode: AudioMode,
    *,
    playback_sink: str | None,
    recording_source: str | None,
    devices: AudioDevices,
) -> tuple[list[str], list[str]]:
    sink = _pick_device_name(playback_sink, devices.default_sink, devices.sinks)
    src = _pick_device_name(recording_source, devices.default_source, devices.sources)

    if mode == AudioMode.MONITOR:
        if not sink:
            raise RuntimeError(
                "No Windows loopback audio device found. Enable a Stereo Mix/loopback device or use Microphone only."
            )
        return (["-f", "dshow", "-i", f"audio={sink}"], [])

    if mode == AudioMode.MIC:
        if not src:
            raise RuntimeError("No Windows microphone/capture device found.")
        return (["-f", "dshow", "-i", f"audio={src}"], [])

    if not sink:
        raise RuntimeError(
            "No Windows loopback audio device found for Mix mode. Enable a Stereo Mix/loopback device or switch modes."
        )
    if not src:
        raise RuntimeError("No Windows microphone/capture device found for Mix mode.")
    return (["-f", "dshow", "-i", f"audio={sink}", "-f", "dshow", "-i", f"audio={src}"], [])

