# Daily sync agent

Linux **X11** system-tray tool that records a **selected window** (video) together with **PulseAudio/PipeWire** audio, saves a **video file** (H.264 + AAC in MKV) and a **separate lossless audio file** (FLAC), then runs **local** models: **faster-whisper** for transcription and **Ollama** (or compatible HTTP API) for a summary of the speech.

You can also **process an existing video or audio file** from the command line (`daily-sync-agent process …`); see [CLI: transcribe and summarize a file](#cli-transcribe-and-summarize-a-file).

## Requirements

- **OS / session**: Linux with **X11** (KDE, GNOME, etc.). Wayland is not supported for window capture in this version.
- **System packages**:
  - `ffmpeg` on `PATH`
  - `pactl` (typically from **pulseaudio-utils** or your distro’s PipeWire Pulse compatibility tools) to list default sinks/sources and device names
  - Optional: `xdotool` for window picking (otherwise the app uses an X11 pointer grab)
  - Optional: `xwininfo` for geometry when using `xdotool`
- **Python**: 3.11+
- **Local AI**:
  - **Ollama** (or another local HTTP API) at the URL in Preferences (default `http://127.0.0.1:11434`). Choose the **chat model** from the dropdown (installed models from your Ollama server), or **“First available”** to use the first model in `ollama list`. The app tries several HTTP endpoints for summarization. Verify the server with `curl -s http://127.0.0.1:11434/api/tags`. Slow local models may need a higher **`ollama_request_timeout_s`** in `config.json` (default is generous).
  - **faster-whisper** — choose model size, device (auto / CPU / GPU), and compute type in Preferences. If GPU is selected but CUDA libraries are missing, transcription falls back to CPU automatically.

### Whisper (faster-whisper) models

Model files are **downloaded automatically** the first time you transcribe with a given size (from [Hugging Face](https://huggingface.co/Systran), e.g. `Systran/faster-whisper-base`). No separate installer is required.

1. **Pick a size in the app** — **Preferences → Whisper model** lists standard sizes (tiny, base, small, …) and any `Systran/faster-whisper-*` snapshots already in your local Hugging Face cache. Use **Refresh list** after downloading a new model.
2. **Pre-download in a terminal (optional)** — Run Python once so the weights are cached before recording:
   ```bash
   source /path/to/.venv/bin/activate
   python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
   ```
   Replace `base` with `small`, `large-v3`, etc. See [faster-whisper](https://github.com/SYSTRAN/faster-whisper#usage) for supported names.
3. **Cache location** — Typically `~/.cache/huggingface/hub/` (or `$HF_HOME`). Large models need several GB free.
4. **Device** — **Auto-detect** lets faster-whisper pick CPU or GPU; **GPU (CUDA)** needs a working CUDA setup for CTranslate2. **Whisper compute type** updates with the device (quantization / precision).

### Finding and installing Ollama models

1. **Browse the library** — Open [ollama.com/library](https://ollama.com/library) and search or filter by task (chat, vision, code, etc.). Each model page lists **pull** commands and tags (sizes/quantizations).
2. **Pull a model** — In a terminal, run:
   ```bash
   ollama pull <model-name>
   ```
   Examples: `ollama pull llama3.2`, `ollama pull qwen2.5:7b`. Use the exact name and tag shown on the model page.
3. **Confirm it is installed** — `ollama list` should show the model. `curl -s http://127.0.0.1:11434/api/tags` should include it under `"models"`.
4. **Use it in this app** — Open **Preferences**, set **Ollama base URL** if needed, click **Refresh list** next to **Ollama chat model**, then pick the model in the dropdown. The window position and size are remembered for the next time you open Preferences.

### System packages (automated)

If you see **“could not query audio devices … pactl”** (or similar), install OS dependencies first. From the repository root:

```bash
./scripts/install-system-dependencies.sh
```

This uses your distro’s package manager (`apt`, `dnf`, `pacman`, `zypper`, or `apk` where detected) to install **ffmpeg**, **pulseaudio-utils** (provides `pactl`), **xdotool**, **xwininfo**, and a **Python 3** toolchain (including `venv` / build tools where needed).

Optional flags:

- `./scripts/install-system-dependencies.sh --venv` — also creates `.venv` and runs `pip install -e .`
- `./scripts/install-system-dependencies.sh --with-ollama` — prompt to install [Ollama](https://ollama.com) via the official script (for summaries). Use `--yes-ollama` or `NONINTERACTIVE=1` to skip the confirmation when appropriate.
- **CUDA user-space libraries (GPU Whisper)** — by default, on **Debian/Ubuntu (apt)** the script also installs **libcublas12**, **libcudart12**, and **libnvrtc12** so `libcublas.so.12` resolves for CTranslate2 (Ubuntu **multiverse** may be enabled). Use **`--skip-cuda-runtime`** to skip these packages. The **NVIDIA driver** is not installed by this script. On Fedora, Arch, etc., the script prints a short pointer to NVIDIA’s CUDA downloads instead.

You need **sudo** access for system packages.

## Install

```bash
cd /path/to/daily-sync-agent
./scripts/install-system-dependencies.sh --venv   # optional: system deps + venv in one step
# or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the tray app:

```bash
daily-sync-agent
```

Ensure `DISPLAY` points at your X11 session (e.g. `:0`).

### CLI: transcribe and summarize a file

To **transcribe and summarize** an existing **video or audio** file (anything **ffmpeg** can decode) and write **`transcript.txt`** and **`summary.txt`** in the **same folder** as the media file:

```bash
daily-sync-agent process /path/to/recording.mkv
```

Settings come from **`~/.config/daily-sync-agent/config.json`** (same as the tray app: Whisper model, Ollama URL/model, summary format, timeouts). This subcommand does **not** start the GUI (no `DISPLAY` required for the AI step).

Global options must appear **before** the subcommand, for example:

```bash
daily-sync-agent --debug process /path/to/recording.mkv
```

Use **`daily-sync-agent gui`** to start the tray explicitly; with **no** subcommand, the tray starts by default.

### Debug mode (verbose log file)

For the **tray app**, run with **`--debug`** to write **DEBUG**-level logs (app events, ffmpeg argv, AI pipeline steps, **httpx** traffic, Qt messages) to a file under `~/.cache/daily-sync-agent/logs/`, named `app-YYYYMMDD-HHMMSS.log` by default:

```bash
daily-sync-agent --debug
```

Use **`--log-file /path/to/app.log`** with **`--debug`** to choose the path. Per-session **ffmpeg** logs are still written as `ffmpeg-<timestamp>.log` in the same directory; in debug mode the ffmpeg process uses `-loglevel verbose`.

## Usage

1. Use the tray icon → **Select window…** and click the target window (or use `xdotool` if installed).
2. Choose **Audio capture mode**:
   - **Mix**: loopback of the chosen **playback** device (what you hear) plus the chosen **recording** device (microphone), mixed together.
   - **Playback / system output**: monitor of the selected sink only.
   - **Microphone only**: selected source only.
3. Optionally set **Playback device** and **Recording device** (defaults follow PulseAudio/PipeWire defaults).
4. **Start recording** — the tray icon turns **red**. **Stop recording** when finished.
5. Output is under `~/Videos/DailySyncRecordings/<timestamp>/`:
   - `recording.mkv` — video + audio
   - `recording.flac` — audio only (used for transcription)
   - `transcript.txt` — Whisper output
   - `summary.txt` — LLM summary (format depends on **Preferences → Summary format**: **General** = one short paragraph; **Daily scrum** = structured sections — title, date, topics, solutions, tasks, blockers — in the same language as the transcript)  
   FFmpeg logs: `~/.cache/daily-sync-agent/logs/`.  
   App config: `~/.config/daily-sync-agent/config.json` (window placement, Ollama URL/model, **`summary_mode`**, **`ollama_request_timeout_s`**, Whisper options, etc.).

## Legal and privacy

Obtain consent before recording calls or meetings. Processing is **local** to your machine (Whisper + Ollama as configured); you are responsible for compliance with applicable laws and policies.

## Packaging (optional)

For a single-directory or binary distribution you can explore **PyInstaller** or **briefcase**, or ship a venv plus a small launcher script. This repository does not pin a specific packager; keep `ffmpeg` and `pactl` as external dependencies unless you bundle them.
