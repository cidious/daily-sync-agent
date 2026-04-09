"""ffmpeg command construction and subprocess management."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from daily_sync_agent.audio.devices import AudioDevices, AudioMode, resolve_pulse_input
from daily_sync_agent.capture.window_x11 import WindowInfo

# libx264 + yuv420p need even width/height; window geometry from X11 can be odd (e.g. 2691x1645).
_VF_EVEN_YUV420 = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"


@dataclass
class FfmpegPaths:
    video_path: Path
    audio_only_path: Path


def build_ffmpeg_command(
    win: WindowInfo,
    *,
    display: str,
    fps: int,
    mode: AudioMode,
    devices: AudioDevices,
    playback_sink: str | None,
    recording_source: str | None,
    out: FfmpegPaths,
    ffmpeg_loglevel: str = "info",
) -> list[str]:
    """Full argv for ffmpeg: x11grab + pulse + dual file outputs."""
    geo = f"{win.width}x{win.height}"
    offset = f"+{win.x},{win.y}"
    x11 = f"{display}{offset}"

    pulse_args, _ = resolve_pulse_input(
        mode,
        playback_sink=playback_sink,
        recording_source=recording_source,
        devices=devices,
    )

    # Inputs: [0] x11grab, [1] pulse (or [1][2] for MIX)
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        ffmpeg_loglevel,
        "-y",
        "-f",
        "x11grab",
        "-framerate",
        str(fps),
        "-video_size",
        geo,
        "-i",
        x11,
        *pulse_args,
    ]

    if mode == AudioMode.MIX:
        # Video: force even dimensions for libx264. Audio: asplit so MKV + FLAC each map a branch once.
        cmd.extend(
            [
                "-filter_complex",
                f"[0:v]{_VF_EVEN_YUV420}[v];"
                "[1:a][2:a]amix=inputs=2:duration=longest:dropout_transition=0[a_mix];"
                "[a_mix]asplit=2[a_mkv][a_flac]",
                "-map",
                "[v]",
                "-map",
                "[a_mkv]",
            ]
        )
    else:
        cmd.extend(["-map", "0:v", "-map", "1:a", "-vf", _VF_EVEN_YUV420])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(out.video_path),
        ]
    )

    # Second output: audio-only (lossless flac)
    if mode == AudioMode.MIX:
        cmd.extend(["-map", "[a_flac]", "-c:a", "flac", str(out.audio_only_path)])
    else:
        cmd.extend(["-map", "1:a", "-c:a", "flac", str(out.audio_only_path)])

    return cmd


@dataclass
class RecordingProcess:
    process: subprocess.Popen[bytes]
    log_path: Path | None = None
    _log_fp: object | None = None

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self._close_log()

    def wait(self) -> int | None:
        rc = self.process.wait()
        self._close_log()
        return rc

    def _close_log(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None

    @property
    def returncode(self) -> int | None:
        return self.process.returncode


def start_recording(cmd: list[str], log_path: Path | None) -> RecordingProcess:
    log_file = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_file or subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log_file else subprocess.DEVNULL,
    )
    return RecordingProcess(process=proc, log_path=log_path, _log_fp=log_file)
