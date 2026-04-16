"""Cross-platform audio device discovery and ffmpeg input resolution."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum

from daily_sync_agent.platform import is_windows


class AudioMode(str, Enum):
    """What to capture for the recording track."""

    MONITOR = "monitor"  # loopback of playback (remote side / system audio)
    MIC = "mic"  # microphone input
    MIX = "mix"  # monitor + mic


@dataclass(frozen=True)
class AudioDevices:
    default_sink: str
    default_source: str
    sinks: list[tuple[str, str]]  # (name, description)
    sources: list[tuple[str, str]]


def _run_pactl(args: list[str]) -> str:
    r = subprocess.run(
        ["pactl", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"pactl {' '.join(args)} failed: {r.stderr or r.stdout}")
    return r.stdout.strip()


def get_default_sink() -> str:
    return _run_pactl(["get-default-sink"])


def get_default_source() -> str:
    return _run_pactl(["get-default-source"])


def _parse_short_list_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("\t")
    if len(parts) < 2:
        parts = line.split(maxsplit=1)
    if len(parts) < 2:
        return None
    name = parts[1].strip()
    desc = parts[2].strip() if len(parts) > 2 else name
    return (name, desc)


def list_sinks() -> list[tuple[str, str]]:
    out = _run_pactl(["list", "short", "sinks"])
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        p = _parse_short_list_line(line)
        if p:
            rows.append(p)
    return rows


def list_sources() -> list[tuple[str, str]]:
    out = _run_pactl(["list", "short", "sources"])
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        p = _parse_short_list_line(line)
        if p:
            rows.append(p)
    return rows


def list_devices() -> AudioDevices:
    if is_windows():
        from daily_sync_agent.audio.devices_windows import list_windows_devices

        return list_windows_devices()
    return AudioDevices(
        default_sink=get_default_sink(),
        default_source=get_default_source(),
        sinks=list_sinks(),
        sources=list_sources(),
    )


def sink_monitor_name(sink_name: str) -> str:
    """Pulse monitor source name for a sink (what you hear)."""
    if sink_name.endswith(".monitor"):
        return sink_name
    return f"{sink_name}.monitor"


def monitor_exists_for_sink(sink_name: str, sources: list[tuple[str, str]]) -> bool:
    want = sink_monitor_name(sink_name)
    names = {n for n, _ in sources}
    return want in names


def resolve_pulse_input(
    mode: AudioMode,
    *,
    playback_sink: str | None,
    recording_source: str | None,
    devices: AudioDevices,
) -> tuple[list[str], list[str]]:
    """Return (ffmpeg_input_args_list, filter_complex_or_empty).

    Each pulse input is expanded as: -f pulse -i <device>
    For MIX, returns filter_complex for amix and two -i args.
    """
    sink = playback_sink or devices.default_sink
    src = recording_source or devices.default_source

    if mode == AudioMode.MONITOR:
        mon = sink_monitor_name(sink)
        return (["-f", "pulse", "-i", mon], [])

    if mode == AudioMode.MIC:
        return (["-f", "pulse", "-i", src], [])

    # MIX: monitor + mic (ffmpeg: input 0 = x11grab, 1 = monitor, 2 = mic — filter built in capture/ffmpeg.py)
    mon = sink_monitor_name(sink)
    return (["-f", "pulse", "-i", mon, "-f", "pulse", "-i", src], [])


def resolve_audio_input(
    mode: AudioMode,
    *,
    playback_sink: str | None,
    recording_source: str | None,
    devices: AudioDevices,
) -> tuple[list[str], list[str]]:
    if is_windows():
        from daily_sync_agent.audio.devices_windows import resolve_windows_input

        return resolve_windows_input(
            mode,
            playback_sink=playback_sink,
            recording_source=recording_source,
            devices=devices,
        )
    return resolve_pulse_input(
        mode,
        playback_sink=playback_sink,
        recording_source=recording_source,
        devices=devices,
    )


def wpctl_fallback_defaults() -> tuple[str | None, str | None]:
    """Try wpctl if pactl is unavailable; returns (sink_id_or_name, source_id_or_name)."""
    try:
        r = subprocess.run(
            ["wpctl", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return (None, None)
        text = r.stdout
        sink_m = re.search(r"Default Sink:\s*(.+)", text)
        src_m = re.search(r"Default Source:\s*(.+)", text)
        sink = sink_m.group(1).strip() if sink_m else None
        source = src_m.group(1).strip() if src_m else None
        return (sink, source)
    except FileNotFoundError:
        return (None, None)
