"""Clip a capture rectangle to the visible virtual desktop (multi-monitor safe)."""

from __future__ import annotations

import logging
import re
import subprocess

from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication

from daily_sync_agent.capture.window_info import WindowInfo
from daily_sync_agent.platform import is_linux, is_windows

logger = logging.getLogger(__name__)


def _desktop_bounds_from_windows_metrics() -> QRect | None:
    """Virtual desktop bounds in physical pixels via Win32 metrics."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        # https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-getsystemmetrics
        left = int(user32.GetSystemMetrics(76))
        top = int(user32.GetSystemMetrics(77))
        width = int(user32.GetSystemMetrics(78))
        height = int(user32.GetSystemMetrics(79))
        if width > 0 and height > 0:
            return QRect(left, top, width, height)
    except Exception as e:
        logger.debug("Win32 virtual desktop bounds unavailable: %s", e)
    return None


def _desktop_bounds_from_xrandr() -> QRect | None:
    """Union of all connected outputs — same pixel space as xwininfo / x11grab."""
    try:
        r = subprocess.run(
            ["xrandr"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if r.returncode != 0:
            return None
        united = QRect()
        found = False
        for line in r.stdout.splitlines():
            if " connected" not in line:
                continue
            m = re.search(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", line)
            if not m:
                continue
            w, h, x, y = (int(m.group(i)) for i in range(1, 5))
            united = united.united(QRect(x, y, w, h))
            found = True
        if found and not united.isNull():
            return united
    except Exception as e:
        logger.debug("xrandr desktop bounds: %s", e)
    return None


def _desktop_bounds_from_xdpyinfo() -> QRect | None:
    """Single-screen / full framebuffer size (origin 0,0)."""
    try:
        r = subprocess.run(
            ["xdpyinfo"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if r.returncode != 0:
            return None
        m = re.search(r"dimensions:\s*(\d+)x(\d+)\s*pixels", r.stdout)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            return QRect(0, 0, w, h)
    except Exception as e:
        logger.debug("xdpyinfo desktop bounds: %s", e)
    return None


def _virtual_desktop_rect_qt_fallback() -> QRect:
    """Last resort; on HiDPI X11, geometry may be logical — prefer xrandr/xdpyinfo above."""
    app = QGuiApplication.instance()
    if app is None:
        return QRect()
    united = QRect()
    for screen in app.screens():
        united = united.united(screen.geometry())
    return united


def _virtual_desktop_rect() -> QRect:
    """
    Visible framebuffer in native pixels (same as WindowInfo / ffmpeg x11grab on Linux, gdigrab on Windows).

    Qt ``QScreen.geometry()`` is often in device-independent coordinates on scaled displays;
    prefer platform-specific tools (xrandr/xdpyinfo on Linux, mss on Windows) instead.
    """
    if is_linux():
        r = _desktop_bounds_from_xrandr()
        if r is not None and not r.isNull():
            return r
        r = _desktop_bounds_from_xdpyinfo()
        if r is not None and not r.isNull():
            return r
        logger.warning("Using Qt screen geometry for desktop clip (xrandr/xdpyinfo unavailable)")
    elif is_windows():
        # Windows: try mss for physical pixel bounds
        try:
            import mss
            with mss.mss() as sct:
                united = QRect()
                for monitor in sct.monitors[1:]:  # Skip monitor 0 (virtual combined)
                    united = united.united(QRect(
                        monitor['left'],
                        monitor['top'],
                        monitor['width'],
                        monitor['height']
                    ))
                if not united.isNull():
                    return united
        except ImportError:
            logger.debug("mss not available for Windows multi-monitor support")
        except Exception as e:
            logger.debug("mss failed to detect monitors: %s", e)

        r = _desktop_bounds_from_windows_metrics()
        if r is not None and not r.isNull():
            return r

    return _virtual_desktop_rect_qt_fallback()


def clip_window_info_to_visible_desktop(info: WindowInfo) -> WindowInfo | None:
    """
    Intersect ``info`` with the visible virtual desktop so ffmpeg ``x11grab``
    never requests pixels outside the framebuffer (e.g. window partially off-screen).
    Returns ``None`` if nothing would be visible.
    """
    desktop = _virtual_desktop_rect()
    if desktop.isNull():
        logger.warning("Could not read desktop bounds; recording without desktop clip")
        return info
    win = QRect(info.x, info.y, info.width, info.height)
    clipped = win.intersected(desktop)
    if clipped.isEmpty() or clipped.width() < 1 or clipped.height() < 1:
        return None
    return WindowInfo(
        title=info.title,
        x=clipped.x(),
        y=clipped.y(),
        width=clipped.width(),
        height=clipped.height(),
        window_id=info.window_id,
    )
