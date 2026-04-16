from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtCore import QRect

from daily_sync_agent.capture.desktop_clip import _desktop_bounds_from_windows_metrics
from daily_sync_agent.audio.devices import AudioDevices, AudioMode, resolve_audio_input
from daily_sync_agent.audio.devices_windows import list_windows_devices, pick_windows_audio_mode, windows_audio_mode_available
from daily_sync_agent.capture.ffmpeg import FfmpegPaths, build_ffmpeg_command
from daily_sync_agent.capture.window_info import WindowInfo
from daily_sync_agent.main import _enable_windows_dpi_awareness
from daily_sync_agent.settings import AppConfig


class FfmpegBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = WindowInfo(
            window_id="123",
            title="Test Window",
            x=10,
            y=20,
            width=800,
            height=600,
        )
        self.paths = FfmpegPaths(video_path=Path("recording.mkv"), audio_only_path=Path("recording.flac"))

    def test_linux_command_uses_x11grab_and_pulse(self) -> None:
        devices = AudioDevices(
            default_sink="default_sink",
            default_source="default_source",
            sinks=[("default_sink", "Default sink")],
            sources=[("default_source", "Default source")],
        )
        with patch("daily_sync_agent.capture.ffmpeg.is_windows", return_value=False), patch(
            "daily_sync_agent.audio.devices.is_windows", return_value=False
        ):
            cmd = build_ffmpeg_command(
                self.window,
                display=":0",
                fps=25,
                mode=AudioMode.MIC,
                devices=devices,
                playback_sink=None,
                recording_source=None,
                out=self.paths,
            )
        self.assertIn("x11grab", cmd)
        self.assertIn("pulse", cmd)
        self.assertIn(":0+10,20", cmd)

    def test_windows_command_uses_gdigrab_and_dshow(self) -> None:
        devices = AudioDevices(
            default_sink="Stereo Mix",
            default_source="Microphone",
            sinks=[("Stereo Mix", "Stereo Mix")],
            sources=[("Microphone", "Microphone")],
        )
        with patch("daily_sync_agent.capture.ffmpeg.is_windows", return_value=True), patch(
            "daily_sync_agent.audio.devices.is_windows", return_value=True
        ):
            cmd = build_ffmpeg_command(
                self.window,
                display=":0",
                fps=30,
                mode=AudioMode.MIC,
                devices=devices,
                playback_sink=None,
                recording_source=None,
                out=self.paths,
            )
        self.assertIn("gdigrab", cmd)
        self.assertIn("dshow", cmd)
        self.assertIn("desktop", cmd)
        self.assertIn("audio=Microphone", cmd)
        self.assertIn("-offset_x", cmd)
        self.assertIn("10", cmd)


class WindowsAudioResolutionTests(unittest.TestCase):
    def test_windows_mix_falls_back_to_microphone_when_loopback_is_missing(self) -> None:
        devices = AudioDevices(default_sink="", default_source="Mic", sinks=[], sources=[("Mic", "Mic")])

        mode, reason = pick_windows_audio_mode(AudioMode.MIX, devices)

        self.assertEqual(mode, AudioMode.MIC)
        self.assertIn("Microphone only", reason or "")

    def test_windows_mix_falls_back_to_monitor_when_microphone_is_missing(self) -> None:
        devices = AudioDevices(
            default_sink="Stereo Mix",
            default_source="",
            sinks=[("Stereo Mix", "Stereo Mix")],
            sources=[],
        )

        mode, reason = pick_windows_audio_mode(AudioMode.MIX, devices)

        self.assertEqual(mode, AudioMode.MONITOR)
        self.assertIn("system output only", reason or "")

    def test_windows_mode_availability_tracks_detected_devices(self) -> None:
        devices = AudioDevices(default_sink="", default_source="Mic", sinks=[], sources=[("Mic", "Mic")])

        self.assertFalse(windows_audio_mode_available(AudioMode.MIX, devices))
        self.assertFalse(windows_audio_mode_available(AudioMode.MONITOR, devices))
        self.assertTrue(windows_audio_mode_available(AudioMode.MIC, devices))

    def test_windows_monitor_requires_loopback_device(self) -> None:
        devices = AudioDevices(default_sink="", default_source="Mic", sinks=[], sources=[("Mic", "Mic")])
        with patch("daily_sync_agent.audio.devices.is_windows", return_value=True):
            with self.assertRaises(RuntimeError):
                resolve_audio_input(
                    AudioMode.MONITOR,
                    playback_sink=None,
                    recording_source=None,
                    devices=devices,
                )

    def test_windows_mix_uses_first_detected_sink_when_default_sink_is_empty(self) -> None:
        devices = AudioDevices(
            default_sink="",
            default_source="",
            sinks=[("Speakers (Realtek) (loopback)", "Speakers (Realtek) (loopback)")],
            sources=[("Microphone Array", "Microphone Array")],
        )
        with patch("daily_sync_agent.audio.devices.is_windows", return_value=True):
            args, _ = resolve_audio_input(
                AudioMode.MIX,
                playback_sink=None,
                recording_source=None,
                devices=devices,
            )
        self.assertIn("audio=Speakers (Realtek) (loopback)", args)
        self.assertIn("audio=Microphone Array", args)

    def test_windows_loopback_heuristics_detect_common_dshow_variants(self) -> None:
        names = [
            "Microphone Array (Realtek(R) Audio)",
            "Speakers (Realtek(R) Audio) (loopback)",
            "virtual-audio-capturer",
        ]
        with patch("daily_sync_agent.audio.devices_windows._list_dshow_audio_device_names", return_value=names):
            devices = list_windows_devices()

        sink_names = [name for name, _ in devices.sinks]
        self.assertIn("Speakers (Realtek(R) Audio) (loopback)", sink_names)
        self.assertIn("virtual-audio-capturer", sink_names)
        self.assertNotIn("Microphone Array (Realtek(R) Audio)", sink_names)


class SettingsCompatibilityTests(unittest.TestCase):
    def test_load_coerces_legacy_numeric_window_id_to_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"last_capture_window_id": 123, "last_capture_title": "Saved"}), encoding="utf-8")
            with patch.object(AppConfig, "config_path", return_value=path):
                cfg = AppConfig.load()
        self.assertEqual(cfg.last_capture_window_id, "123")
        self.assertEqual(cfg.last_capture_title, "Saved")


class WindowsHiDpiTests(unittest.TestCase):
    def test_enable_windows_dpi_awareness_prefers_per_monitor_v2_context(self) -> None:
        calls: list[int] = []

        class _FakeUser32:
            def SetProcessDpiAwarenessContext(self, ctx: object) -> int:
                value = int(getattr(ctx, "value", ctx))
                calls.append(value)
                return 1 if value == -4 else 0

        with patch("daily_sync_agent.main.is_windows", return_value=True), patch("daily_sync_agent.main.os.name", "nt"), patch(
            "ctypes.windll", SimpleNamespace(user32=_FakeUser32()), create=True
        ), patch("ctypes.c_void_p", lambda v: v):
            _enable_windows_dpi_awareness()

        self.assertEqual(calls, [-4])

    def test_windows_virtual_desktop_metrics_bounds(self) -> None:
        class _FakeUser32:
            metrics = {76: -1920, 77: 0, 78: 3840, 79: 2160}

            def GetSystemMetrics(self, code: int) -> int:
                return self.metrics.get(code, 0)

        with patch("ctypes.windll", SimpleNamespace(user32=_FakeUser32()), create=True):
            rect = _desktop_bounds_from_windows_metrics()

        self.assertEqual(rect, QRect(-1920, 0, 3840, 2160))


if __name__ == "__main__":
    unittest.main()

