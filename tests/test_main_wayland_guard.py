from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from daily_sync_agent.main import _gui_startup_block_reason, _is_wayland_session


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


if __name__ == "__main__":
    unittest.main()

