# AGENTS.md

## Project scope
- `daily-sync-agent` is a Linux **X11 tray app** for recording one selected window + Pulse/PipeWire audio, then running local AI (Whisper + Ollama-style HTTP) to generate `transcript.txt` and `summary.txt`.
- Main entrypoint is the console script `daily-sync-agent` from `pyproject.toml` -> `daily_sync_agent.main:main`.
- Two runtime modes share the same config and AI pipeline: GUI (`daily-sync-agent` / `daily-sync-agent gui`) and headless file processing (`daily-sync-agent process <media>`).
- GUI startup is intentionally blocked on Wayland sessions in `main.py`; `process` remains supported without X11.

## Architecture map (read these first)
- `src/daily_sync_agent/main.py`: CLI parsing, debug-log flags, dispatch to GUI vs `process` subcommand.
- `src/daily_sync_agent/app.py`: `TrayApplication` state machine (window pick, record start/stop, AI queue, Preferences UI).
- `src/daily_sync_agent/capture/ffmpeg.py`: ffmpeg argv builder + subprocess lifecycle (`RecordingProcess`).
- `src/daily_sync_agent/audio/devices.py`: `pactl`-based device discovery and Pulse input resolution by `AudioMode`.
- `src/daily_sync_agent/ai/pipeline.py`: orchestrates `transcribe_file()` then `summarize_text()` and writes outputs.
- `src/daily_sync_agent/settings.py`: persistent user config in `~/.config/daily-sync-agent/config.json`; log/output dirs from XDG paths.

## Data flow that matters
- GUI record flow: select window -> clip to visible desktop (`capture/desktop_clip.py`) -> build ffmpeg cmd -> write `recording.mkv` + `recording.flac` under `~/Videos/DailySyncRecordings/<timestamp>/` -> enqueue AI thread.
- AI flow is intentionally audio-first: transcription always reads FLAC/audio (`ai/transcribe.py`) rather than video.
- `run_transcribe_and_summarize()` writes `transcript.txt` before summary; summary prompt can include session date parsed from folder name (`ai/session_date.py`).
- `transcribe_speech` is a master AI toggle: when disabled, GUI recording still saves media but skips AI queueing, and `process` mode exits after reporting AI is disabled.

## Integration points / external dependencies
- Hard runtime deps: `ffmpeg`, `pactl`, X11 session (`DISPLAY`), Python 3.11+.
- Optional helpers: `xdotool` and `xwininfo` for window picking/geometry, with Xlib fallback in `capture/window_x11.py`.
- LLM integration is endpoint-flexible in `ai/summarize.py`: tries Ollama `/api/chat` + `/api/generate`, then OpenAI-compatible and `llama.cpp`-style endpoints when server is not Ollama.
- Whisper integration (`faster-whisper`) includes CUDA->CPU fallback on runtime/VRAM failures; preserve this behavior when modifying transcription.
- Optional low-VRAM cleanup uses both HTTP (`keep_alive: 0`, `/api/ps`) and best-effort `ollama stop` CLI calls (`ai/summarize.py`); Whisper cleanup in `ai/transcribe.py` also drops transcription objects and, when `nvidia-smi` is available, waits for this process’s NVIDIA VRAM usage to fall before summary requests begin.

## Project-specific conventions
- Keep work local-first and resilient: failures often degrade gracefully (e.g., missing audio device list, non-zero ffmpeg exit with existing output, model fallback logic).
- Settings are user-facing and persisted immediately via `AppConfig.save()`; UI changes in Preferences should round-trip through config fields.
- Audio mode semantics are strict (`AudioMode.MIX`, `MONITOR`, `MIC`) and determine ffmpeg input indexing/filter graph; update `audio/devices.py` and `capture/ffmpeg.py` together.
- Debug logging is file-based only when `--debug` is enabled (`log_config.setup_logging`) for both GUI and `process` mode; includes Whisper/summarization stage details + timing, `httpx/httpcore`, and Qt bridge.
- `transcribe_speech` controls all AI options in Preferences; when off, Whisper/summarization/low-VRAM controls are disabled and summarization is forced off on save.
- `run_transcribe_and_summarize()` must keep optional-summary semantics: always write `transcript.txt`, and return/write `summary.txt` only when `summarize_transcript` is enabled.
- `unload_models_after_task` must propagate through both transcription and summarization code paths (including Whisper VRAM-release waiting in `ai/transcribe.py` and Ollama keep-alive behavior).

## Developer workflows
- Install deps (recommended): `./scripts/install-system-dependencies.sh --venv`.
- Manual setup: create venv, `pip install -e .`, then run `daily-sync-agent`.
- Headless AI check (no GUI): `daily-sync-agent --debug process /path/to/media.mkv`.
- GUI debug run: `daily-sync-agent --debug` (logs under `~/.cache/daily-sync-agent/logs/`).
- Run tests: `python -m unittest discover -s tests -v`.

## Change safety checklist for agents
- If touching recording: verify ffmpeg args still produce both `recording.mkv` and `recording.flac`.
- If touching summarization: keep language-matching constraints and multi-endpoint fallback behavior.
- If touching window geometry: preserve physical-pixel clipping logic (`desktop_clip.py`) to avoid off-screen x11grab failures.
- If adding config fields: update `AppConfig` defaults so `load()` remains backward-compatible with old `config.json` files.
- If touching startup/CLI dispatch: keep Wayland guard behavior (`gui` blocked, `process` allowed).
- If touching low-VRAM mode: keep `unload_models_after_task` wiring end-to-end (`ai/pipeline.py`, `ai/transcribe.py`, `ai/summarize.py`, and tray shutdown cleanup).
- If touching Preferences AI controls: keep `transcribe_speech` as the master toggle for enabling/disabling all AI-related controls and processing.
