"""Highlight a screen rectangle (saved capture region) for a few seconds."""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class WindowFrameOverlay(QWidget):
    """Frameless, topmost rectangle showing where the last capture region was.

    Use as a **top-level** window only (``parent=None``): ``setGeometry`` must be in
    **global X11 screen pixels**, matching ffmpeg ``x11grab`` and ``WindowInfo``. A
    parent widget would use **local** coordinates and misalign the highlight.
    """

    def __init__(self, rect: QRect, *, duration_ms: int = 2800, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.BypassWindowManagerHint,
        )
        self.setGeometry(rect)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._duration_ms = duration_ms

    def show_and_expire(self) -> None:
        self.show()
        self.raise_()
        QTimer.singleShot(self._duration_ms, self.deleteLater)

    def paintEvent(self, event: object) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 80, 0, 55))
        pen = QPen(QColor(255, 60, 0))
        pen.setWidth(4)
        p.setPen(pen)
        p.drawRect(self.rect().adjusted(2, 2, -3, -3))
