"""Helpers for mapping native X11 coordinates into Qt logical widget coordinates."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScreenCoordinateMap:
    """One screen's native-vs-logical geometry pair."""

    name: str
    native: QRect
    logical: QRect

    @property
    def scale_x(self) -> float:
        return self.native.width() / self.logical.width() if self.logical.width() > 0 else 1.0

    @property
    def scale_y(self) -> float:
        return self.native.height() / self.logical.height() if self.logical.height() > 0 else 1.0


def parse_xrandr_listmonitors(text: str) -> list[tuple[str, QRect]]:
    """Parse ``xrandr --listmonitors`` output into native X11 rectangles."""
    monitors: list[tuple[str, QRect]] = []
    pattern = re.compile(r"^\s*\d+:\s+[+*]*([^\s]+)\s+(\d+)/\d+x(\d+)/\d+\+(-?\d+)\+(-?\d+)")
    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        name = m.group(1)
        w, h, x, y = (int(m.group(i)) for i in range(2, 6))
        monitors.append((name, QRect(x, y, w, h)))
    return monitors


def _query_native_monitor_rects() -> dict[str, QRect]:
    try:
        r = subprocess.run(
            ["xrandr", "--listmonitors"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as e:
        logger.debug("xrandr --listmonitors failed: %s", e)
        return {}
    if r.returncode != 0:
        return {}
    return {name: rect for name, rect in parse_xrandr_listmonitors(r.stdout)}


def build_screen_coordinate_maps() -> list[ScreenCoordinateMap]:
    """Pair Qt logical screen geometry with native X11 geometry when available."""
    app = QGuiApplication.instance()
    if app is None:
        return []

    native_by_name = _query_native_monitor_rects()
    maps: list[ScreenCoordinateMap] = []
    for screen in app.screens():
        logical = screen.geometry()
        native = native_by_name.get(screen.name())
        if native is None:
            dpr = screen.devicePixelRatio() or 1.0
            native = QRect(
                round(logical.x() * dpr),
                round(logical.y() * dpr),
                max(1, round(logical.width() * dpr)),
                max(1, round(logical.height() * dpr)),
            )
            logger.debug(
                "Falling back to DPR-derived native geometry for screen %s: logical=%s dpr=%.3f native=%s",
                screen.name(),
                logical,
                dpr,
                native,
            )
        maps.append(ScreenCoordinateMap(name=screen.name(), native=native, logical=logical))
    return maps


def _intersection_area(a: QRect, b: QRect) -> int:
    inter = a.intersected(b)
    if inter.isEmpty():
        return 0
    return inter.width() * inter.height()


def _best_screen_for_native_rect(native_rect: QRect, screens: list[ScreenCoordinateMap]) -> ScreenCoordinateMap | None:
    if not screens:
        return None
    center = native_rect.center()
    for screen in screens:
        if screen.native.contains(center):
            return screen
    best = max(screens, key=lambda screen: _intersection_area(native_rect, screen.native))
    if _intersection_area(native_rect, best.native) > 0:
        return best
    return screens[0]


def map_native_rect_to_logical(native_rect: QRect, screens: list[ScreenCoordinateMap]) -> QRect:
    """Convert a native X11 rectangle into Qt logical coordinates for widget placement."""
    screen = _best_screen_for_native_rect(native_rect, screens)
    if screen is None:
        return native_rect
    sx = screen.scale_x
    sy = screen.scale_y
    if sx <= 0 or sy <= 0:
        return native_rect
    logical_x = screen.logical.x() + round((native_rect.x() - screen.native.x()) / sx)
    logical_y = screen.logical.y() + round((native_rect.y() - screen.native.y()) / sy)
    logical_w = max(1, round(native_rect.width() / sx))
    logical_h = max(1, round(native_rect.height() / sy))
    return QRect(logical_x, logical_y, logical_w, logical_h)
