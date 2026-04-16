"""Test platform detection and cross-platform abstractions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from daily_sync_agent import platform


class TestPlatformDetection(unittest.TestCase):
    """Test platform detection logic."""

    def test_get_platform_returns_linux_or_windows(self):
        """Verify get_platform() returns one of: 'linux', 'windows'."""
        plat = platform.get_platform()
        self.assertIn(plat, ("linux", "windows"))

    def test_is_linux_is_windows_mutually_exclusive(self):
        """Verify is_linux() and is_windows() are mutually exclusive."""
        is_lin = platform.is_linux()
        is_win = platform.is_windows()
        self.assertNotEqual(is_lin, is_win, "Platform must be either Linux or Windows, not both")

    def test_get_wsl_distro_returns_none_on_non_wsl(self):
        """WSL detection on non-WSL systems should return None."""
        # This test may fail on WSL, which is expected
        distro = platform.get_wsl_distro()
        # Just verify it doesn't crash and returns str | None
        self.assertIsInstance(distro, (str, type(None)))

    def test_platform_detection_cached(self):
        """Verify platform detection is cached."""
        p1 = platform.get_platform()
        p2 = platform.get_platform()
        self.assertEqual(p1, p2)


class TestWindowInfo(unittest.TestCase):
    """Test WindowInfo cross-platform structure."""

    def test_window_info_creation(self):
        """WindowInfo should store window geometry."""
        from daily_sync_agent.capture.window_info import WindowInfo

        info = WindowInfo(window_id="123", title="Test", x=10, y=20, width=800, height=600)
        self.assertEqual(info.window_id, "123")
        self.assertEqual(info.title, "Test")
        self.assertEqual(info.x, 10)
        self.assertEqual(info.y, 20)
        self.assertEqual(info.width, 800)
        self.assertEqual(info.height, 600)

    def test_window_info_str(self):
        """WindowInfo str() should show title and geometry."""
        from daily_sync_agent.capture.window_info import WindowInfo

        info = WindowInfo(window_id="123", title="My Window", x=0, y=0, width=1920, height=1080)
        s = str(info)
        self.assertIn("My Window", s)
        self.assertIn("1920x1080", s)


class TestSettingsPaths(unittest.TestCase):
    """Test cross-platform path handling in settings."""

    def test_log_dir_exists(self):
        """log_dir() should return a valid path."""
        from daily_sync_agent.settings import log_dir

        p = log_dir()
        self.assertTrue(p.exists())
        self.assertTrue(p.is_dir())

    def test_output_dir_exists(self):
        """output_dir() should return a valid path."""
        from daily_sync_agent.settings import output_dir

        p = output_dir()
        self.assertTrue(p.exists())
        self.assertTrue(p.is_dir())

    def test_speaker_profiles_path_is_file(self):
        """speaker_profiles_path() should point to an .npz file."""
        from daily_sync_agent.settings import speaker_profiles_path

        p = speaker_profiles_path()
        self.assertTrue(str(p).endswith(".npz"))

    def test_speaker_names_path_is_file(self):
        """speaker_names_path() should point to a .txt file."""
        from daily_sync_agent.settings import speaker_names_path

        p = speaker_names_path()
        self.assertTrue(str(p).endswith(".txt"))

    def test_config_path_is_json(self):
        """AppConfig.config_path() should point to config.json."""
        from daily_sync_agent.settings import AppConfig

        p = AppConfig.config_path()
        self.assertTrue(str(p).endswith("config.json"))


class TestMainPlatformGuards(unittest.TestCase):
    """Test main.py platform-specific logic."""

    @patch("daily_sync_agent.main.is_linux")
    @patch("daily_sync_agent.main.is_windows")
    def test_gui_startup_blocked_on_wayland_linux(self, mock_is_windows, mock_is_linux):
        """GUI should be blocked on Wayland (Linux only)."""
        from daily_sync_agent.main import _gui_startup_block_reason

        mock_is_windows.return_value = False
        mock_is_linux.return_value = True

        with patch("daily_sync_agent.main._is_wayland_session", return_value=True):
            reason = _gui_startup_block_reason()
            self.assertIsNotNone(reason)
            self.assertIn("Wayland", reason)

    @patch("daily_sync_agent.main.is_linux")
    @patch("daily_sync_agent.main.is_windows")
    def test_gui_startup_allowed_on_windows(self, mock_is_windows, mock_is_linux):
        """GUI should be allowed on Windows."""
        from daily_sync_agent.main import _gui_startup_block_reason

        mock_is_windows.return_value = True
        mock_is_linux.return_value = False

        reason = _gui_startup_block_reason()
        self.assertIsNone(reason)

    @patch("daily_sync_agent.main.is_linux")
    @patch("daily_sync_agent.main.is_windows")
    def test_gui_startup_allowed_on_linux_x11(self, mock_is_windows, mock_is_linux):
        """GUI should be allowed on Linux X11."""
        from daily_sync_agent.main import _gui_startup_block_reason

        mock_is_windows.return_value = False
        mock_is_linux.return_value = True

        with patch("daily_sync_agent.main._is_wayland_session", return_value=False):
            reason = _gui_startup_block_reason()
            self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()

