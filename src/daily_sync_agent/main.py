"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from daily_sync_agent.log_config import setup_logging


def _cli_process_media(media: Path, *, debug: bool) -> int:
    """Transcribe and summarize one file; write ``transcript.txt`` and ``summary.txt`` beside it."""
    from daily_sync_agent.ai.pipeline import run_transcribe_and_summarize
    from daily_sync_agent.settings import AppConfig

    path = media.expanduser().resolve()
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 2
    cfg = AppConfig.load()
    out_dir = path.parent
    print(f"Processing: {path}")
    print(f"Output dir:  {out_dir}")
    try:
        transcript_path, summary_path = run_transcribe_and_summarize(path, out_dir, cfg)
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        if debug:
            traceback.print_exc()
        return 1
    print(f"Wrote {transcript_path}")
    print(f"Wrote {summary_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="X11 window capture with local transcription and summary.",
        epilog="Global options (--debug, --log-file) must appear before the subcommand, e.g. "
        "daily-sync-agent --debug process ./recording.mkv",
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
        help="Transcribe and summarize a video or audio file; save transcript.txt and summary.txt next to it.",
    )
    p_process.add_argument(
        "media",
        type=Path,
        help="Path to a media file (ffmpeg must decode it; video or audio).",
    )

    args = parser.parse_args()
    if args.log_file is not None and not args.debug:
        parser.error("--log-file requires --debug")

    if args.command == "process":
        setup_logging(debug=args.debug, log_file=args.log_file)
        sys.exit(_cli_process_media(args.media, debug=args.debug))

    # Tray GUI (default, or explicit `gui`)
    from PySide6.QtWidgets import QApplication

    from daily_sync_agent.app import TrayApplication

    debug_log_path = setup_logging(debug=args.debug, log_file=args.log_file)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    tray = TrayApplication(debug=args.debug, debug_log_path=debug_log_path)
    app.aboutToQuit.connect(tray.wait_for_ai_thread)
    tray.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
