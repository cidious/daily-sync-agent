from daily_sync_agent.capture.ffmpeg import RecordingProcess, build_ffmpeg_command
from daily_sync_agent.capture.window_info import WindowInfo

# Platform-specific imports
from daily_sync_agent.platform import is_windows

if is_windows():
    from daily_sync_agent.capture.window_windows import pick_window_interactive
else:
    from daily_sync_agent.capture.window_x11 import pick_window_x11 as pick_window_interactive

__all__ = ["RecordingProcess", "WindowInfo", "build_ffmpeg_command", "pick_window_interactive"]
