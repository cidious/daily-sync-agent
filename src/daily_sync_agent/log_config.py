"""Verbose file logging for debug mode."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from daily_sync_agent.settings import log_dir

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _install_qt_log_bridge() -> None:
    try:
        from PySide6.QtCore import QMessageLogContext, QtMsgType, qInstallMessageHandler
    except ImportError:
        return

    log = logging.getLogger("daily_sync_agent.qt")

    def handler(mode: QtMsgType, context: QMessageLogContext, message: str) -> None:
        extra = ""
        if getattr(context, "file", None):
            extra = f" ({context.file}:{getattr(context, 'line', 0)})"
        text = f"{message}{extra}"
        if mode == QtMsgType.QtDebugMsg:
            log.debug("%s", text)
        elif mode == QtMsgType.QtInfoMsg:
            log.info("%s", text)
        elif mode == QtMsgType.QtWarningMsg:
            log.warning("%s", text)
        elif mode == QtMsgType.QtCriticalMsg:
            log.error("%s", text)
        elif mode == QtMsgType.QtFatalMsg:
            log.critical("%s", text)
        else:
            log.info("%s", text)

    qInstallMessageHandler(handler)


def setup_logging(*, debug: bool, log_file: Path | None = None) -> Path | None:
    """When debug is True, write DEBUG logs for the ``daily_sync_agent`` tree to a file."""
    if not debug:
        return None

    path = log_file
    if path is None:
        path = log_dir() / f"app-{dt.datetime.now():%Y%m%d-%H%M%S}.log"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    pkg = logging.getLogger("daily_sync_agent")
    pkg.setLevel(logging.DEBUG)
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FMT))
    pkg.addHandler(fh)
    pkg.propagate = False

    for name in ("httpx", "httpcore"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(fh)
        lg.propagate = False

    _install_qt_log_bridge()
    pkg.debug("Debug logging initialized (file=%s)", path)
    return path
