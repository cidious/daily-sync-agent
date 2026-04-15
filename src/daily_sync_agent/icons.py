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


def _circle_icon_with_inner(diameter: int, outer_fill: QColor, outer_border: QColor, inner_fill: QColor) -> QIcon:
    """Create a circle with a smaller inner circle."""
    pm = QPixmap(diameter, diameter)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Outer circle
    p.setBrush(outer_fill)
    p.setPen(outer_border)
    margin = max(1, diameter // 16)
    p.drawEllipse(margin, margin, diameter - 2 * margin, diameter - 2 * margin)

    # Inner circle (processing indicator)
    inner_diameter = diameter // 3
    inner_margin = (diameter - inner_diameter) // 2
    p.setBrush(inner_fill)
    p.setPen(inner_fill)
    p.drawEllipse(inner_margin, inner_margin, inner_diameter, inner_diameter)

    p.end()
    return QIcon(pm)


def icon_idle() -> QIcon:
    return _circle_icon(64, QColor(120, 120, 120), QColor(80, 80, 80))


def icon_recording() -> QIcon:
    return _circle_icon(64, QColor(220, 60, 60), QColor(140, 30, 30))


def icon_processing() -> QIcon:
    """Gray circle with blue inner circle indicator for AI processing."""
    return _circle_icon_with_inner(64, QColor(120, 120, 120), QColor(80, 80, 80), QColor(70, 150, 220))


def recommended_tray_size() -> QSize:
    return QSize(22, 22)
