"""Tray icons: idle vs recording."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap


def _circle_icon(diameter: int, fill: QColor, border: QColor | None = None) -> QIcon:
    pm = QPixmap(diameter, diameter)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(fill)
    if border:
        p.setPen(border)
    else:
        p.setPen(fill)
    margin = max(1, diameter // 16)
    p.drawEllipse(margin, margin, diameter - 2 * margin, diameter - 2 * margin)
    p.end()
    return QIcon(pm)


def icon_idle() -> QIcon:
    return _circle_icon(64, QColor(120, 120, 120), QColor(80, 80, 80))


def icon_recording() -> QIcon:
    return _circle_icon(64, QColor(220, 60, 60), QColor(140, 30, 30))


def recommended_tray_size() -> QSize:
    return QSize(22, 22)
