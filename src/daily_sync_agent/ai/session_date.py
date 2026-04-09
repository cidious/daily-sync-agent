"""Derive a human-readable event time from the recording session folder name."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def format_event_date_from_session_dir(out_dir: Path) -> str:
    """``DailySyncRecordings/YYYYMMDD-HHMMSS`` → ``YYYY-MM-DD HH:MM``."""
    name = out_dir.name
    m = re.fullmatch(r"(\d{8})-(\d{6})", name)
    if not m:
        return ""
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""
