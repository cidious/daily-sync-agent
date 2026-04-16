"""Platform detection and cross-platform abstractions.

Provides utilities for detecting the current platform (Linux/Windows) and
conditional feature availability.
"""

from __future__ import annotations

import logging
import platform as stdlib_platform

logger = logging.getLogger(__name__)

_CACHED_SYSTEM: str | None = None


def get_platform() -> str:
    """Return normalized platform name: 'linux' or 'windows'."""
    global _CACHED_SYSTEM
    if _CACHED_SYSTEM is None:
        system = stdlib_platform.system().lower()
        if system == "linux":
            _CACHED_SYSTEM = "linux"
        elif system == "windows":
            _CACHED_SYSTEM = "windows"
        else:
            # Default to linux for fallback (e.g., darwin not officially supported yet)
            _CACHED_SYSTEM = "linux"
        logger.debug("Platform detected: %s (full: %s)", _CACHED_SYSTEM, stdlib_platform.platform())
    return _CACHED_SYSTEM


def is_linux() -> bool:
    """Return True if platform is Linux."""
    return get_platform() == "linux"


def is_windows() -> bool:
    """Return True if platform is Windows."""
    return get_platform() == "windows"


def get_wsl_distro() -> str | None:
    """Return WSL distro name if running under WSL, else None."""
    try:
        with open("/proc/version", "r") as f:
            content = f.read().lower()
            if "microsoft" in content or "wsl" in content:
                # Simple heuristic; could parse /etc/os-release for accuracy
                return "wsl"
    except (FileNotFoundError, OSError):
        pass
    return None

