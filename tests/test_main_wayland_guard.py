from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from daily_sync_agent.main import _cli_process_media, _gui_startup_block_reason, _is_wayland_session
from daily_sync_agent.settings import AppConfig


class MainWaylandGuardTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_not_wayland_by_default(self) -> None:
        self.assertIs(_is_wayland_session(), False)
        self.assertIsNone(_gui_startup_block_reason())

    @patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=True)
    def test_detect_wayland_via_session_type(self) -> None:
        self.assertIs(_is_wayland_session(), True)
        reason = _gui_startup_block_reason()
        self.assertIsInstance(reason, str)
        self.assertIn("Wayland session detected", reason or "")

    @patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True)
    def test_detect_wayland_via_display_var(self) -> None:
        self.assertIs(_is_wayland_session(), True)

    @patch("daily_sync_agent.settings.AppConfig.load", return_value=AppConfig(transcribe_speech=False))
    def test_process_mode_skips_when_transcription_is_disabled(self, _mock_cfg) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "sample.wav"
            media.write_bytes(b"x")
            rc = _cli_process_media(media, debug=False)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
