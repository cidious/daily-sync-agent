from __future__ import annotations

import unittest

from PySide6.QtCore import QRect

from daily_sync_agent.ui.coordinate_map import ScreenCoordinateMap, map_native_rect_to_logical, parse_xrandr_listmonitors


class CoordinateMapTests(unittest.TestCase):
    def test_parse_xrandr_listmonitors(self) -> None:
        monitors = parse_xrandr_listmonitors(
            """
Monitors: 2
 0: +*HDMI-0 3840/700x2160/400+0+0  HDMI-0
 1: +DP-0 1920/520x1080/290+3840+0  DP-0
"""
        )
        self.assertEqual(
            monitors,
            [
                ("HDMI-0", QRect(0, 0, 3840, 2160)),
                ("DP-0", QRect(3840, 0, 1920, 1080)),
            ],
        )

    def test_map_native_rect_to_logical_single_hidpi_screen(self) -> None:
        screens = [
            ScreenCoordinateMap(
                name="HDMI-0",
                native=QRect(0, 0, 3840, 2160),
                logical=QRect(0, 0, 2691, 1514),
            )
        ]
        native_rect = QRect(960, 540, 1920, 1080)
        logical = map_native_rect_to_logical(native_rect, screens)
        self.assertEqual(logical, QRect(673, 378, 1346, 757))

    def test_map_native_rect_to_logical_uses_matching_screen_origin(self) -> None:
        screens = [
            ScreenCoordinateMap(
                name="HDMI-0",
                native=QRect(0, 0, 3840, 2160),
                logical=QRect(0, 0, 2691, 1514),
            ),
            ScreenCoordinateMap(
                name="DP-0",
                native=QRect(3840, 0, 1920, 1080),
                logical=QRect(2691, 0, 1920, 1080),
            ),
        ]
        native_rect = QRect(4000, 100, 800, 400)
        logical = map_native_rect_to_logical(native_rect, screens)
        self.assertEqual(logical, QRect(2851, 100, 800, 400))


if __name__ == "__main__":
    unittest.main()
