"""Platform-agnostic window info structure."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WindowInfo:
    """Window information for capture (platform-agnostic)."""

    window_id: str  # Platform-specific ID (HWND hex string on Windows, window ID on Linux)
    title: str
    x: int          # Physical pixel position
    y: int
    width: int      # Physical pixel dimensions
    height: int

    def __str__(self) -> str:
        return f"{self.title} ({self.width}x{self.height}+{self.x}+{self.y})"

