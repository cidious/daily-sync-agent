"""Windows window capture using HWND and ctypes."""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from typing import Any

from daily_sync_agent.capture.window_info import WindowInfo

logger = logging.getLogger(__name__)

# Windows API constants
GW_OWNER = 4
WS_VISIBLE = 0x10000000

# ctypes Windows API bindings
_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32


def _get_hwnd_title(hwnd: int) -> str:
    """Get window title from HWND."""
    try:
        title_len = _user32.GetWindowTextLengthW(hwnd)
        if title_len <= 0:
            return "(no title)"
        buf = ctypes.create_unicode_buffer(title_len + 1)
        _user32.GetWindowTextW(hwnd, buf, title_len + 1)
        return buf.value or "(empty)"
    except Exception as e:
        logger.debug("Failed to get window title for HWND %s: %s", hwnd, e)
        return "(error)"


def _get_hwnd_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Get window rect (x, y, width, height) from HWND."""
    try:
        rect = wintypes.RECT()
        if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        x = rect.left
        y = rect.top
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w > 0 and h > 0:
            return (x, y, w, h)
    except Exception as e:
        logger.debug("Failed to get window rect for HWND %s: %s", hwnd, e)
    return None


def _is_valid_window(hwnd: int) -> bool:
    """Check if HWND is a valid, visible, capturable window."""
    try:
        # Must be visible
        if not _user32.IsWindowVisible(hwnd):
            return False

        # Skip owner windows (tooltips, menus, etc.)
        if _user32.GetWindow(hwnd, GW_OWNER):
            return False

        # Must have a rect
        rect = _get_hwnd_rect(hwnd)
        if rect is None or rect[2] <= 0 or rect[3] <= 0:
            return False

        return True
    except Exception:
        return False


def pick_window_interactive() -> WindowInfo | None:
    """Prompt the user to choose a capturable window from a Qt list dialog."""
    from PySide6.QtWidgets import QInputDialog

    windows = enumerate_windows()
    if not windows:
        return None

    labels = [str(window) for window in windows]
    chosen, ok = QInputDialog.getItem(
        None,
        "Select window",
        "Window to record:",
        labels,
        0,
        False,
    )
    if not ok or not chosen:
        return None
    for window, label in zip(windows, labels):
        if label == chosen:
            return window
    return None


def enumerate_windows() -> list[WindowInfo]:
    """Return list of visible, capturable windows."""
    windows: list[WindowInfo] = []

    def enum_proc(hwnd: int, _: Any) -> bool:
        if _is_valid_window(hwnd):
            title = _get_hwnd_title(hwnd)
            rect = _get_hwnd_rect(hwnd)
            if rect:
                x, y, w, h = rect
                window_id = f"0x{hwnd:x}"
                window = WindowInfo(window_id=window_id, title=title, x=x, y=y, width=w, height=h)
                windows.append(window)
        return True

    try:
        enum_proc_type = ctypes.CFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        _user32.EnumWindows(enum_proc_type(enum_proc), 0)
    except Exception as e:
        logger.error("Failed to enumerate windows: %s", e)

    logger.debug("Enumerated %d windows", len(windows))
    return windows


def hwnd_from_window_info(window: WindowInfo) -> int | None:
    """Convert WindowInfo back to HWND integer."""
    try:
        return int(window.window_id, 16)
    except (ValueError, AttributeError):
        return None


def focus_window(window: WindowInfo) -> bool:
    """Bring window to focus."""
    hwnd = hwnd_from_window_info(window)
    if hwnd is None:
        return False
    try:
        _user32.SetForegroundWindow(hwnd)
        return True
    except Exception as e:
        logger.debug("Failed to focus window %s: %s", window, e)
        return False

