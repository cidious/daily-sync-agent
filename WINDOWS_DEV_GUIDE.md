# Quick Start: Windows Support Development

## For Developers

### Running the App on Windows
```powershell
# From the repo root
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -e ".[windows]"  # Install Windows-specific deps
daily-sync-agent  # Launch GUI or use 'daily-sync-agent process /path/to/file.mkv'
```

### Running Tests
```bash
python -m unittest discover -s tests -v
python -m unittest tests.test_platform_detection -v  # Platform tests only
```

### Debugging
```bash
daily-sync-agent --debug  # Logs to platform-specific cache dir
daily-sync-agent --debug --log-file C:\temp\app.log  # Custom log file
```

## Architecture Quick Reference

### Platform Detection
```python
from daily_sync_agent.platform import is_linux, is_windows, get_platform

if is_windows():
    # Windows-specific code
elif is_linux():
    # Linux-specific code
```

### Cross-Platform Paths
```python
from daily_sync_agent.settings import log_dir, output_dir, speaker_profiles_path

# All these automatically use platformdirs for correct OS paths
log_path = log_dir()          # C:\Users\...\AppData\Local\daily-sync-agent\logs on Windows
output_path = output_dir()    # C:\Users\...\Videos\DailySyncRecordings on Windows
profiles_path = speaker_profiles_path()  # C:\Users\...\AppData\Local\daily-sync-agent\speaker_profiles.npz on Windows
```

### Window Capture
```python
from daily_sync_agent.capture import pick_window_interactive, WindowInfo

# Platform automatically selects Linux X11 or Windows HWND picker
window: WindowInfo | None = pick_window_interactive()
if window:
    print(f"Selected: {window.title} at ({window.x}, {window.y})")
    print(f"Window ID (platform-specific): {window.window_id}")
```

### Audio Devices
```python
from daily_sync_agent.audio.devices import list_devices

devices = list_devices()  # Automatically uses pactl on Linux or WASAPI on Windows
playback = devices.playback_devices
recording = devices.recording_devices
```

## Common Tasks

### Adding a New Platform
1. Create `src/daily_sync_agent/capture/window_<platform>.py` with functions:
   - `pick_window_interactive() -> WindowInfo | None`
   - `enumerate_windows() -> list[WindowInfo]`
   - `focus_window(window: WindowInfo) -> bool`

2. Update `src/daily_sync_agent/capture/__init__.py`:
   ```python
   if is_windows():
       from daily_sync_agent.capture.window_windows import pick_window_interactive
   elif is_<platform>():
       from daily_sync_agent.capture.window_<platform> import pick_window_interactive
   else:
       from daily_sync_agent.capture.window_x11 import pick_window_x11 as pick_window_interactive
   ```

3. Create `src/daily_sync_agent/audio/devices_<platform>.py` with:
   - `list_<platform>_devices() -> AudioDevices`

### Testing Platform-Specific Code
```python
import unittest
from daily_sync_agent.platform import is_windows, is_linux

class TestWindowsFeature(unittest.TestCase):
    @unittest.skipUnless(is_windows(), "Windows-only test")
    def test_hwnd_enumeration(self):
        from daily_sync_agent.capture.window_windows import enumerate_windows
        windows = enumerate_windows()
        self.assertIsInstance(windows, list)

    @unittest.skipUnless(is_linux(), "Linux-only test")
    def test_xlib_fallback(self):
        from daily_sync_agent.capture.window_x11 import pick_window_x11
        # Test X11 picker
```

### Updating Platform-Dependent Code
When you modify `capture/ffmpeg.py` or `audio/devices.py`, remember:
- Keep the shared interface in the main module
- Add platform-specific implementation in `..._windows.py` / `..._x11.py` / etc.
- Import conditionally in `__init__.py` or use try/except at runtime
- Test both platforms produce compatible output

## Files Reference

| File | Purpose | Modified for Windows |
|------|---------|----------------------|
| `platform.py` | Platform detection | NEW |
| `main.py` | CLI entry point | Updated (Wayland guard Linux-only) |
| `capture/window_info.py` | Shared window interface | NEW |
| `capture/window_windows.py` | Windows window capture | NEW |
| `capture/__init__.py` | Platform-aware imports | Updated (conditional imports) |
| `capture/desktop_clip.py` | Desktop clipping | Updated (mss for Windows) |
| `audio/devices_windows.py` | WASAPI discovery | NEW |
| `settings.py` | Config paths | Updated (platformdirs) |
| `pyproject.toml` | Dependencies | Updated (platform-specific deps) |
| `scripts/install-system-app-windows.ps1` | Windows installer | NEW |

## Known Limitations (Phase 1)

- Interactive window picker shows error (stub implementation)
- ffmpeg args still use x11grab (need gdigrab for Windows)
- Audio device names not yet mapped to ffmpeg dshow indices
- diarization/speaker-id not tested on Windows
- No GPU acceleration testing on Windows

## Next Steps (Phase 2)

1. [ ] Implement interactive window picker UI
2. [ ] Update ffmpeg args builder for gdigrab
3. [ ] Test and enable diarization on Windows
4. [ ] Add CI/CD Windows build
5. [ ] Create Windows installer (.msi)
6. [ ] Test multi-GPU scenarios

