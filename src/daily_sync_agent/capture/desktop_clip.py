"""Clip a capture rectangle to the visible virtual desktop (multi-monitor safe)."""

from __future__ import annotations

import logging
import re
import subprocess

from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication

from daily_sync_agent.capture.window_x11 import WindowInfo

logger = logging.getLogger(__name__)


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
    Visible framebuffer in **native X11 pixels** (same as WindowInfo / ffmpeg x11grab).

    Qt ``QScreen.geometry()`` is often in **device-independent** coordinates on scaled X11
    desktops; intersecting that with xwininfo’s **physical** window rect yields a tiny
    wrong region. Prefer ``xrandr`` / ``xdpyinfo`` instead.
    """
    r = _desktop_bounds_from_xrandr()
    if r is not None and not r.isNull():
        return r
    r = _desktop_bounds_from_xdpyinfo()
    if r is not None and not r.isNull():
        return r
    logger.warning("Using Qt screen geometry for desktop clip (xrandr/xdpyinfo unavailable)")
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
        x=clipped.x(),
        y=clipped.y(),
        width=clipped.width(),
        height=clipped.height(),
        window_id=info.window_id,
    )
