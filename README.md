# Daily sync agent

Linux **X11** system-tray tool that records a **selected window** (video) together with **PulseAudio/PipeWire** audio, saves a **video file** (H.264 + AAC in MKV) and a **separate lossless audio file** (FLAC), then runs **local** models: **faster-whisper** for transcription and **Ollama** (or compatible HTTP API) for a summary of the speech.

You can also **process an existing video or audio file** from the command line (`daily-sync-agent process …`); see [CLI: transcribe and summarize a file](#cli-transcribe-and-summarize-a-file).

## Requirements

- **OS / session**: Linux with **X11** (KDE, GNOME, etc.). Wayland is not supported for window capture in this version. If a Wayland session is detected, `daily-sync-agent` / `daily-sync-agent gui` exits early with a clear message; `daily-sync-agent process <media>` still works.
- **System packages**:
  - `ffmpeg` on `PATH`
  - `pactl` (typically from **pulseaudio-utils** or your distro’s PipeWire Pulse compatibility tools) to list default sinks/sources and device names
  - Optional: `xdotool` for window picking (otherwise the app uses an X11 pointer grab)
  - Optional: `xwininfo` for geometry when using `xdotool`
- **Python**: 3.11+
- **Local AI**:
  - **Transcribe speech (Preferences checkbox)** — master AI toggle. When off, recording still saves media files but transcription/summarization are skipped. The app disables all AI controls (Whisper settings, summarization, and low-VRAM mode).
  - **Ollama** (or another local HTTP API) at the URL in Preferences (default `http://127.0.0.1:11434`). Choose the **chat model** from the dropdown (installed models from your Ollama server), or **“First available”** to use the first model in `ollama list`. The app tries several HTTP endpoints for summarization. Verify the server with `curl -s http://127.0.0.1:11434/api/tags`. Slow local models may need a higher **`ollama_request_timeout_s`** in `config.json` (default is generous). If **Preferences → Summarize the transcripted text** is turned off, Ollama is not used.
  - **faster-whisper** — choose model size, device (auto / CPU / GPU), and compute type in Preferences. If GPU is selected but CUDA libraries are missing, transcription falls back to CPU automatically.
  - **Low-VRAM mode (Preferences checkbox)** — unload Whisper/Ollama models after each task to free GPU memory. Before summarization starts, the app drops Whisper transcription objects and, when `nvidia-smi` is available, waits for this process’s NVIDIA VRAM usage to fall so Ollama does not start competing with the just-finished Whisper load. On app exit, Ollama model runners are also unloaded/stopped best-effort.

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

### Auto-select and install a summarizer model (GPU/VRAM aware)

For long transcript summaries (around **32k tokens**), use the helper script:

```bash
./scripts/install-ollama-summarizer.sh
```

What it does:
- Detects GPU presence and VRAM (`nvidia-smi` / `rocm-smi`, else CPU-only mode).
- Chooses a best-fit model tier for summarization from one family (`qwen`, `gemma3`, `mistral-small`, `mistral-nemo`, `llama`) and defaults to `qwen` for `auto`.
- Generates a tuned `Modelfile` with `num_ctx`, `num_gpu`, and generation parameters.
- Pulls the chosen base model and creates a local alias (`daily-sync-summary` by default).

Useful flags:

```bash
./scripts/install-ollama-summarizer.sh --dry-run
./scripts/install-ollama-summarizer.sh --family gemma3
./scripts/install-ollama-summarizer.sh --model qwen2.5:14b --allow-cpu-offload
./scripts/install-ollama-summarizer.sh --allow-cpu-offload --prefer-bigger
./scripts/install-ollama-summarizer.sh --ctx 32768 --alias daily-sync-summary
```

Exact app settings recommended for this project after install:
- `ollama_model`: `daily-sync-summary` (or your chosen `--alias`)
- `ollama_request_timeout_s`: `1800`
- `summary_mode`: `daily_scrum`
- `summarize_transcript`: `true`
- `transcribe_speech`: `true`
- `unload_models_after_task`: `true` on small GPUs (typically under 16 GB VRAM), otherwise `false`

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

### System-wide install (launcher + menu entry)

To install the app for all users (under `/opt`, with a launcher in `/usr/share/applications` and a command in `/usr/local/bin`):

```bash
./scripts/install-system-app.sh --with-deps
```

This installs:

- `/opt/daily-sync-agent/src` (copied repository snapshot)
- `/opt/daily-sync-agent/.venv` (runtime virtualenv)
- `/usr/local/bin/daily-sync-agent` (symlink to the venv entrypoint)
- `/usr/share/applications/daily-sync-agent.desktop` (start-menu launcher)

Uninstall:

```bash
./scripts/install-system-app.sh --uninstall
```

### CLI: transcribe and summarize a file

To **transcribe** an existing **video or audio** file (anything **ffmpeg** can decode) and optionally write **`summary.txt`** in the **same folder** as the media file:

```bash
daily-sync-agent process /path/to/recording.mkv
```

Settings come from **`~/.config/daily-sync-agent/config.json`** (same as the tray app: Whisper model, whether transcription/summarization is enabled, Ollama URL/model, summary format, timeouts). This subcommand does **not** start the GUI (no `DISPLAY` required for the AI step). If `transcribe_speech` is disabled, this command exits after printing that AI processing was skipped.

Global options must appear **before** the subcommand, for example:

```bash
daily-sync-agent --debug process /path/to/recording.mkv
```

Use **`daily-sync-agent gui`** to start the tray explicitly; with **no** subcommand, the tray starts by default.

### Debug mode (verbose log file)

For the **tray app** and **CLI process mode**, run with **`--debug`** to write **DEBUG**-level logs (app events, ffmpeg argv, AI pipeline steps, Whisper model/device/compute-type + elapsed timings, summary endpoint/model + elapsed timings, **httpx** traffic, Qt messages) to a file under `~/.cache/daily-sync-agent/logs/`, named `app-YYYYMMDD-HHMMSS.log` by default:

```bash
daily-sync-agent --debug
```

Use **`--log-file /path/to/app.log`** with **`--debug`** to choose the path. Per-session **ffmpeg** logs are still written as `ffmpeg-<timestamp>.log` in the same directory; in debug mode the ffmpeg process uses `-loglevel verbose`.

## Usage
### Speaker diarization with Pyannote (optional)

Enable **speaker labels** in transcripts for multi-speaker recordings (meetings, interviews, podcasts). When enabled, Whisper transcription is combined with speaker detection to label each segment with a speaker ID.

**Setup:**

1. **Get a HuggingFace token** (free):
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click **"New token"** → name it `daily-sync-agent`
   - Set **Role** to `read`
   - Click **"Create token"** and copy it
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the model license (one-time)

2. **Download and cache the model**:
   ```bash
   ./scripts/install-pyannote-models.sh --hf-token <your-token>
   ```
   Or set the token in your environment:
   ```bash
   export HF_TOKEN="hf_xxxxx..."
   ./scripts/install-pyannote-models.sh
   ```

3. **Enable in Preferences**:
   - Open **Preferences** (right-click tray → Preferences)
   - Check **"Speaker diarization (Pyannote)"**
   - Optional: check **"Speaker identification"** to persist speaker embeddings across sessions and replace generic labels with names
   - Paste your HuggingFace token in the **"HuggingFace token"** field
   - Save

4. **Use it**:
   - Record normally. Transcripts will include speaker labels: `[Speaker_1] Good morning!` / `[Speaker_2] Hi there!`
   - Audio is automatically converted to **16 kHz mono WAV** for best diarization accuracy (Pyannote is sample-rate sensitive)
   - Diarization uses the same **Whisper device** preference from Preferences (`Auto-detect`, `CPU`, `GPU (CUDA)`). In `Auto-detect`, CUDA is used when available.
   - With **Speaker identification** enabled, the app picks the longest clean segment per detected speaker, computes embeddings, and matches them against saved speaker profiles.
   - Edit `~/.config/daily-sync-agent/speaker_names.txt` to rename profile IDs (for example `speaker_01: Alice`) and future transcripts will use those names.
   - First transcription may take longer on initial model load; subsequent runs are faster

**Notes:**
- Diarization requires Pyannote 3.1 (installed by `install-pyannote-models.sh`)
- Keep your HuggingFace token **private** in config (treated as password in Preferences UI)
- Low-VRAM mode applies: models are unloaded after each task if enabled
- Speaker profiles are stored in `~/.config/daily-sync-agent/speaker_profiles.npz`; editable names are stored in `~/.config/daily-sync-agent/speaker_names.txt`.
- Token is stored in `~/.config/daily-sync-agent/config.json`; back up securely if using automated deployments


1. Use the tray icon → **Select window…** and click the target window (or use `xdotool` if installed). The capture **rectangle is saved** in `config.json` (`last_capture_*` fields) so you can **Start recording** after restarting the app without picking again. **Left-click** the tray icon to **flash an on-screen frame** around the saved region (where the desktop sends a tray “activate” event). **Shift+left-click** toggles recording (**start** when idle, **stop** when already recording). **Right-click** opens the menu.
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
   - `summary.txt` — LLM summary when summarization is enabled (format depends on **Preferences → Summary format**: **General** = one short paragraph; **Daily scrum** = structured sections — title, date, topics, solutions, tasks, blockers — in the same language as the transcript)  
   FFmpeg logs: `~/.cache/daily-sync-agent/logs/`.  
   App config: `~/.config/daily-sync-agent/config.json` (window placement, last capture rectangle, Ollama URL/model, **`transcribe_speech`**, **`summarize_transcript`**, **`summary_mode`**, **`ollama_request_timeout_s`**, Whisper options, **`unload_models_after_task`**, etc.).

## Legal and privacy

Obtain consent before recording calls or meetings. Processing is **local** to your machine (Whisper + Ollama as configured); you are responsible for compliance with applicable laws and policies.

## Packaging (optional)

For a single-directory or binary distribution you can explore **PyInstaller** or **briefcase**, or ship a venv plus a small launcher script. This repository does not pin a specific packager; keep `ffmpeg` and `pactl` as external dependencies unless you bundle them.
