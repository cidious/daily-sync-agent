# AGENTS.md

## Project scope
- `daily-sync-agent` is a **cross-platform tray app** (Linux X11 / Windows) for recording one selected window + audio, then running local AI (Whisper + Ollama-style HTTP) to generate `transcript.txt` and `summary.txt`.
- Main entrypoint is the console script `daily-sync-agent` from `pyproject.toml` -> `daily_sync_agent.main:main`.
- Two runtime modes share the same config and AI pipeline: GUI (`daily-sync-agent` / `daily-sync-agent gui`) and headless file processing (`daily-sync-agent process <media>`).
- **Linux**: GUI startup is intentionally blocked on Wayland sessions in `main.py` (requires X11); `process` mode works without X11.
- **Windows**: GUI support uses Win32 window enumeration + FFmpeg `gdigrab`; audio capture uses Windows audio capture devices exposed to FFmpeg (`dshow`). System-output recording depends on an available loopback / Stereo Mix-style device.

## Architecture map (read these first)
- `src/daily_sync_agent/platform.py`: Platform detection (`is_linux()`, `is_windows()`), cached for entire session.
- `src/daily_sync_agent/main.py`: CLI parsing, dependency checking, debug-log flags, dispatch to GUI vs `process` subcommand; Wayland guard on Linux only.
- `src/daily_sync_agent/app.py`: `TrayApplication` state machine (window pick, record start/stop, AI queue, Preferences UI).
- `src/daily_sync_agent/capture/ffmpeg.py`: ffmpeg argv builder + subprocess lifecycle (`RecordingProcess`).
- `src/daily_sync_agent/capture/window_info.py`: Platform-agnostic `WindowInfo` dataclass.
- `src/daily_sync_agent/capture/window_x11.py` (Linux): X11 window picking (xdotool/Xlib) and geometry via xwininfo.
- `src/daily_sync_agent/capture/window_windows.py` (Windows): HWND enumeration via ctypes and a Qt selection dialog.
- `src/daily_sync_agent/capture/desktop_clip.py`: Clip capture rect to visible desktop (xrandr on Linux, mss on Windows).
- `src/daily_sync_agent/audio/devices.py`: Platform-agnostic audio device listing.
- `src/daily_sync_agent/audio/devices_windows.py` (Windows): Windows audio-capture device discovery from FFmpeg DirectShow listings; loopback-capable devices are exposed as playback-monitor choices.
- `src/daily_sync_agent/ai/pipeline.py`: orchestrates `transcribe_file()` then `summarize_text()` and writes outputs.
- `src/daily_sync_agent/ai/transcribe.py`: Whisper integration with optional diarization support; returns transcript + optional speaker segments.
- `src/daily_sync_agent/ai/diarize.py`: Pyannote 3.1 speaker diarization (optional); converts audio to 16 kHz mono WAV (sample-rate critical) and merges speaker labels into transcript.
- `src/daily_sync_agent/ai/speaker_id.py`: optional speaker-identification profiles (embedding matching + name mapping file) applied after diarization; matches speakers across sessions via saved embeddings.
- `src/daily_sync_agent/ai/audio_decode.py`: ffmpeg-based audio decoder avoiding PyAV resampler issues; used by diarization and speaker identification.
- `src/daily_sync_agent/ai/whisper_models.py`: Whisper model name discovery (presets + Hugging Face cache scan).
- `src/daily_sync_agent/ai/whisper_compute.py`: compute_type options by device (auto/cpu/cuda) for CTranslate2 quantization in Preferences UI.
- `src/daily_sync_agent/settings.py`: persistent user config in platform-specific location (Linux XDG / Windows AppData); uses `platformdirs` library for cross-platform path resolution.

## Data flow that matters
- GUI record flow: select window -> clip to visible desktop (`capture/desktop_clip.py`) -> build ffmpeg cmd -> write `recording.mkv` + `recording.flac` under `~/Videos/DailySyncRecordings/<timestamp>/` -> enqueue AI thread.
- AI flow is intentionally audio-first: transcription always reads FLAC/audio (`ai/transcribe.py`) rather than video.
- `run_transcribe_and_summarize()` writes `transcript.txt` before summary; summary prompt can include session date parsed from folder name (`ai/session_date.py`).
- `transcribe_speech` is a master AI toggle: when disabled, GUI recording still saves media but skips AI queueing, and `process` mode exits after reporting AI is disabled.
- When `diarize_speakers` is enabled, Whisper segments are combined with Pyannote speaker detection (16 kHz mono audio via `audio_decode.py`) and speaker labels are merged into transcript (e.g., `[Speaker_1] Text`).
- When `identify_speakers` is enabled, speaker embeddings are extracted from the longest clean audio segments per speaker and matched against persisted profiles (`speaker_profiles.npz`). Matched profiles are renamed using editable names from `speaker_names.txt`, and new speakers create new profiles with auto-incrementing IDs (`speaker_01`, `speaker_02`, etc.).

## Integration points / external dependencies
- **Hard runtime deps (all platforms)**: `ffmpeg`, Python 3.11+, PySide6.
- **Linux runtime deps**: `pactl` or `wpctl` for PulseAudio/PipeWire audio discovery; `xdotool` and `xwininfo` optional (fallback to Xlib).
- **Windows optional deps**: `mss` for multi-monitor geometry; Windows recording uses FFmpeg DirectShow device names, and loopback/system-audio capture depends on an available loopback / Stereo Mix device.
- **Cross-platform deps**: `platformdirs` for XDG/AppData path resolution.
- Platform detection: `platform.py` provides cached `is_linux()` / `is_windows()` checks; used for conditional imports and feature guards.
- Window capture abstraction: `capture/window_info.py` defines shared `WindowInfo`; platform-specific implementations in `window_x11.py` (Linux) and `window_windows.py` (Windows); imported via `capture/__init__.py` conditional logic.
- Audio device discovery abstraction: `audio/devices.py` provides shared interface; platform implementations use `pactl` (Linux) or FFmpeg DirectShow audio-device discovery (Windows).
- Desktop clipping: `desktop_clip.py` uses `xrandr`/`xdpyinfo` on Linux, `mss` on Windows for multi-monitor geometry.
- ffmpeg command building: `capture/ffmpeg.py` builds platform-appropriate args (x11grab + pulse on Linux, gdigrab + dshow on Windows); both produce `recording.mkv` + `recording.flac`.
- Config paths: `settings.py` uses `platformdirs` to resolve config/cache/output dirs (backward-compatible with existing XDG paths on Linux).
- Audio decoding: `ai/audio_decode.py` uses ffmpeg (not PyAV) to avoid resampler bugs; produces mono float32 at specified sample rate.
- LLM integration is endpoint-flexible in `ai/summarize.py`: tries Ollama `/api/chat` + `/api/generate`, then OpenAI-compatible and `llama.cpp`-style endpoints when server is not Ollama.
- Whisper integration (`faster-whisper`) includes CUDA->CPU fallback on runtime/VRAM failures; preserve this behavior when modifying transcription. Compute types are device-specific (default/int8/float16/int8_float16/float32) via `ai/whisper_compute.py`.
- Whisper models are discovered in `ai/whisper_models.py` as presets (tiny, base, small, etc.) plus scanned Hugging Face cache entries; model list is refreshable in Preferences UI.
- Optional Pyannote diarization (`ai/diarize.py`, opt-in via `diarize_speakers` config): requires HuggingFace token + model license acceptance; converts audio to 16kHz mono WAV (critical for accuracy) using ffmpeg before Pyannote processing.
- Optional speaker identification (`ai/speaker_id.py`, requires diarization): embeds speaker audio segments (longest clean segments per speaker) using Pyannote's embedding model, matches against saved profiles via cosine similarity, and maintains editable speaker name mappings.
- Optional low-VRAM cleanup uses both HTTP (`keep_alive: 0`, `/api/ps`) and best-effort `ollama stop` CLI calls (`ai/summarize.py`); Whisper cleanup in `ai/transcribe.py` also drops transcription objects and, when `nvidia-smi` is available, waits for this process's NVIDIA VRAM usage to fall before summary requests begin.

## Project-specific conventions
- Keep work local-first and resilient: failures often degrade gracefully (e.g., missing audio device list, non-zero ffmpeg exit with existing output, model fallback logic).
- Settings are user-facing and persisted immediately via `AppConfig.save()`; UI changes in Preferences should round-trip through config fields.
- AI preference dependencies are centralized in `AppConfig.normalized()` and applied on both `load()` and `save()`; keep GUI/CLI behavior consistent with that single normalization path.
- Audio mode semantics are strict (`AudioMode.MIX`, `MONITOR`, `MIC`) and determine ffmpeg input indexing/filter graph; update `audio/devices.py` and `capture/ffmpeg.py` together.
- Debug logging is file-based only when `--debug` is enabled (`log_config.setup_logging`) for both GUI and `process` mode; includes Whisper/summarization stage details + timing, `httpx/httpcore`, and Qt bridge.
- `transcribe_speech` controls all AI options in Preferences; when off, Whisper/summarization/low-VRAM controls are disabled and summarization is forced off on save.
- `diarize_speakers` requires a non-empty HuggingFace token; when token is missing, normalization disables diarization and speaker identification.
- Speaker diarization follows the Whisper device preference (`auto`/`cpu`/`cuda`): `auto` prefers CUDA when available, and explicit CUDA requests degrade to CPU when CUDA is unavailable.
- Speaker identification depends on diarization and HuggingFace token: keep it opt-in (`identify_speakers` default `False`), and preserve editable `speaker_id: name` text mapping semantics (stored in `speaker_names.txt`).
- `audio_decode.py` must use ffmpeg (not PyAV) for resampling to avoid platform-specific resampler bugs; always produces mono float32 output at the specified sample rate.
- Pyannote audio conversion (`ai/diarize.py`) must always produce 16kHz mono WAV via `_audio_to_16k_mono_wav()` (sample-rate sensitivity is critical for accuracy); temporary files are cleaned up after diarization.
- `run_transcribe_and_summarize()` must keep optional-summary semantics: always write `transcript.txt`, and return/write `summary.txt` only when `summarize_transcript` is enabled.
- `unload_models_after_task` must propagate through both transcription and summarization code paths (including Whisper VRAM-release waiting in `ai/transcribe.py` and Ollama keep-alive behavior).
- **Platform compatibility**: Use `platform.is_linux()` / `platform.is_windows()` for conditional logic; avoid hard-coded Linux assumptions (e.g., X11 paths, pactl). Use `platformdirs` for config/cache paths instead of XDG env vars directly.
- **Window capture abstraction**: Wrap platform-specific window picking in `capture/__init__.py` conditional imports; add new platforms by creating `capture/window_<platform>.py` returning standardized `WindowInfo`.
- **Audio devices abstraction**: Keep `audio/devices.py` platform-agnostic; add platform backends in `audio/devices_<platform>.py` returning same interface.
- **ffmpeg args by platform**: Update `capture/ffmpeg.py` to accept platform-specific flags; test both Linux (x11grab) and Windows (gdigrab) pipelines produce correct outputs.

## Developer workflows
- **Linux**: Install deps via `./scripts/install-system-dependencies.sh --venv`; or manually: `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- **Windows**: Run `.\scripts\install-system-app-windows.ps1 -WithDeps` as Administrator; or manually: `python -m venv .venv && .venv\Scripts\activate && pip install -e . && pip install -e ".[windows]"`
- Run app: `daily-sync-agent` (GUI) or `daily-sync-agent process /path/to/media.mkv` (headless).
- Headless AI check (no GUI): `daily-sync-agent --debug process /path/to/media.mkv`.
- GUI debug run: `daily-sync-agent --debug` (logs to platform-specific cache dir).
- Run tests: `python -m unittest discover -s tests -v` (covers platform detection, window picking on both platforms, audio devices, ffmpeg commands, diarization, speaker ID, low-VRAM mode, optional summaries, and logging behavior).
- Add Windows support tests: use `@unittest.skipUnless(platform.is_windows())` and `@unittest.skipUnless(platform.is_linux())` decorators in test files.

## Change safety checklist for agents
- If touching recording: verify ffmpeg args still produce both `recording.mkv` and `recording.flac` on both Linux and Windows.
- If touching summarization: keep language-matching constraints and multi-endpoint fallback behavior.
- If touching window geometry: preserve physical-pixel clipping logic (`desktop_clip.py`) to avoid off-screen capture failures on both platforms.
- If adding config fields: update `AppConfig` defaults so `load()` remains backward-compatible with old `config.json` files.
- If touching startup/CLI dispatch: keep Wayland guard on Linux only; don't block Windows GUI.
- If touching platform detection: preserve `platform.is_linux()` / `platform.is_windows()` caching; update main.py dependency checks.
- If touching low-VRAM mode: keep `unload_models_after_task` wiring end-to-end (`ai/pipeline.py`, `ai/transcribe.py`, `ai/summarize.py`, `ai/diarize.py`, and tray shutdown cleanup).
- If touching Preferences AI controls: keep `transcribe_speech` as the master toggle for enabling/disabling all AI-related controls and processing.
- If touching speaker diarization: preserve 16kHz mono WAV requirement (sample-rate sensitivity is critical); verify audio conversion via `audio_decode.py` + ffmpeg.
- If touching speaker identification: preserve longest clean-segment selection, profile persistence in platform-specific config dir, and editable names file.
- If adding window capture for new platform: create `capture/window_<platform>.py` returning `WindowInfo` with string window_id; update `capture/__init__.py` conditional imports.
- If adding audio device discovery: create `audio/devices_<platform>.py`; keep `audio/devices.py` platform-agnostic interface; test both backends return same fields/structure.
- If changing ffmpeg command building: test both Linux (x11grab) and Windows (gdigrab/dshow) pipelines produce correct MKV + FLAC outputs; verify multi-monitor geometry handling.
