"""CLI entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

from daily_sync_agent.log_config import setup_logging
from daily_sync_agent.platform import is_linux, is_windows

logger = logging.getLogger(__name__)


def _enable_windows_dpi_awareness() -> None:
    """Enable per-monitor DPI awareness on Windows to avoid coordinate virtualization."""
    if not is_windows() or os.name != "nt":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        # Prefer the modern per-monitor v2 mode; fall back to older APIs.
        for ctx in (-4, -3, -2):
            try:
                if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(ctx)):
                    logger.debug("Enabled Windows DPI awareness context=%s", ctx)
                    return
            except Exception:
                continue
        try:
            shcore = ctypes.windll.shcore
            # PROCESS_PER_MONITOR_DPI_AWARE
            hr = shcore.SetProcessDpiAwareness(2)
            if hr in (0, 0x80070005):
                logger.debug("SetProcessDpiAwareness result=0x%x", hr)
                return
        except Exception:
            pass
        try:
            if user32.SetProcessDPIAware():
                logger.debug("Enabled Windows system DPI awareness via SetProcessDPIAware")
                return
        except Exception:
            pass
        logger.debug("Windows DPI awareness API calls were unavailable or rejected")
    except Exception as e:
        logger.debug("Failed enabling Windows DPI awareness: %s", e)


def _is_wayland_session() -> bool:
    """Check if running on Wayland (Linux only)."""
    if not is_linux():
        return False
    session_type = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
    if session_type == "wayland":
        return True
    return bool(os.environ.get("WAYLAND_DISPLAY", "").strip())


def _gui_startup_block_reason() -> str | None:
    """Return reason GUI startup is blocked, or None if allowed."""
    if is_windows():
        # Windows is fully supported
        return None

    if is_linux() and _is_wayland_session():
        return (
            "Wayland session detected. The tray GUI requires an X11 session for window capture in this version.\n"
            "Use `daily-sync-agent process <media>` for headless transcription/summarization, "
            "or log in to an X11 session for GUI recording."
        )
    return None


def _cli_process_media(media: Path, *, debug: bool) -> int:
    """Transcribe one file and optionally summarize it; write outputs beside it."""
    from daily_sync_agent.settings import AppConfig

    path = media.expanduser().resolve()
    logger.debug("CLI process mode start media=%s debug=%s", path, debug)
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        logger.error("CLI process mode failed: not a file (%s)", path)
        return 2
    cfg = AppConfig.load()
    if not cfg.transcribe_speech:
        print("Transcription is disabled in config; skipping AI processing.")
        logger.debug("CLI process mode skipped AI because transcribe_speech=false")
        return 0

    from daily_sync_agent.ai.pipeline import run_transcribe_and_summarize

    out_dir = path.parent
    print(f"Processing: {path}")
    print(f"Output dir:  {out_dir}")
    try:
        transcript_path, summary_path = run_transcribe_and_summarize(path, out_dir, cfg)
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        logger.exception("CLI process mode failed for media=%s", path)
        if debug:
            traceback.print_exc()
        return 1
    print(f"Wrote {transcript_path}")
    if summary_path is not None:
        print(f"Wrote {summary_path}")
        logger.debug("CLI process mode wrote transcript=%s summary=%s", transcript_path, summary_path)
    else:
        print("Summary generation disabled in config; skipped summary.txt")
        logger.debug("CLI process mode wrote transcript=%s summary=disabled", transcript_path)
    return 0


def _check_required_dependencies() -> str | None:
    """Check for required system/Python dependencies. Return error message if missing, None if OK."""
    import shutil

    # ffmpeg is required on all platforms
    if not shutil.which("ffmpeg"):
        return "ffmpeg not found on PATH. Install ffmpeg and add to PATH."

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Window capture with local transcription and summary (Linux X11 / Windows).",
        epilog="Global options (--debug, --log-file) must appear before the subcommand, e.g. "
        "daily-sync-agent --debug process ./recording.mkv. "
        "Note: On Linux, Wayland sessions require the `process` subcommand; the tray GUI needs X11.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logs (DEBUG) appended to a file under ~/.cache/.../daily-sync-agent/logs/ (see --log-file).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        metavar="PATH",
        help="With --debug, write logs to this file instead of a timestamped name in the cache log directory.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.add_parser(
        "gui",
        help="Start the tray application (default if no subcommand is given).",
    )
    p_process = sub.add_parser(
        "process",
        help="Transcribe a video or audio file and optionally summarize it; save transcript.txt and, if enabled, summary.txt next to it.",
    )
    p_process.add_argument(
        "media",
        type=Path,
        help="Path to a media file (ffmpeg must decode it; video or audio).",
    )

    args = parser.parse_args()
    if args.log_file is not None and not args.debug:
        parser.error("--log-file requires --debug")

    dep_error = _check_required_dependencies()
    if dep_error is not None:
        print(dep_error, file=sys.stderr)
        sys.exit(2)

    if args.command == "process":
        debug_log_path = setup_logging(debug=args.debug, log_file=args.log_file)
        if args.debug and debug_log_path is not None:
            print(f"Debug log file: {debug_log_path}")
        sys.exit(_cli_process_media(args.media, debug=args.debug))

    block = _gui_startup_block_reason()
    if block is not None:
        print(block, file=sys.stderr)
        sys.exit(2)

    # Tray GUI (default, or explicit `gui`)
    _enable_windows_dpi_awareness()

    from PySide6.QtWidgets import QApplication

    from daily_sync_agent.app import TrayApplication

    debug_log_path = setup_logging(debug=args.debug, log_file=args.log_file)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    tray = TrayApplication(debug=args.debug, debug_log_path=debug_log_path)
    app.aboutToQuit.connect(tray.shutdown)
    tray.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
