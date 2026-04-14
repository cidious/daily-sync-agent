# Pyannote Speaker Diarization Integration - Implementation Guide

## What Was Added

This document summarizes the complete integration of **Pyannote 3.1 speaker diarization** into `daily-sync-agent` for automatic speaker identification in multi-speaker recordings.

### 1. **New Files Created**

#### `src/daily_sync_agent/ai/diarize.py` (5.2 KB)
Core diarization module providing:
- `_audio_to_16k_mono_wav()`: Converts audio to 16kHz mono WAV (Pyannote requirement for accuracy)
- `diarize_speakers()`: Runs speaker diarization pipeline with HuggingFace token auth
- `merge_diarization_with_transcript()`: Merges Whisper segments with speaker labels

Key features:
- Automatic audio format conversion with ffmpeg
- Time-based speaker segment matching with transcript
- Graceful fallback to plain transcript if diarization fails
- Comprehensive debug logging with timing metrics

#### `scripts/install-pyannote-models.sh` (4.3 KB, executable)
Installation script for Pyannote models:
- Detects Python version (requires 3.11+)
- Validates HuggingFace token and model access
- Downloads/caches models to `~/.cache/huggingface/hub/`
- Supports `--dry-run` mode for previewing actions
- Accepts token via `--hf-token` flag or `HF_TOKEN` env var
- Includes step-by-step token acquisition guide

#### `tests/test_diarization.py` (4.8 KB)
Unit tests covering:
- Merging empty diarization (fallback behavior)
- Speaker label generation with overlapping segments
- Unknown speaker handling when no segment overlap
- Empty segment filtering
- Audio conversion success and error cases
- ffmpeg argument validation

### 2. **Modified Files**

#### `src/daily_sync_agent/settings.py`
Added two new config fields:
- `diarize_speakers: bool = False` - Feature toggle (off by default)
- `huggingface_token: str = ""` - HF API token (stored securely, treated as password)

Backward compatible: old configs load without these fields (defaults apply).

#### `src/daily_sync_agent/ai/transcribe.py`
Extended `_transcribe_once()` function:
- Added `diarize: bool` and `hf_token: str` parameters
- Returns `tuple[str, list[dict] | None]` (transcript + optional speaker segments)
- On diarization success, merges speaker labels into transcript
- Gracefully degrades to plain transcript if diarization fails
- Updated `transcribe_file()` wrapper to accept and forward diarization params

#### `src/daily_sync_agent/ai/pipeline.py`
Updated `run_transcribe_and_summarize()`:
- Passes `config.diarize_speakers` and `config.huggingface_token` to transcription
- Pipeline is transparent to downstream summarization (summary works with speaker-labeled text)

#### `src/daily_sync_agent/app.py`
Enhanced Preferences dialog:
- Added `QCheckBox` for "Speaker diarization (Pyannote)" feature toggle
- Added `QLineEdit` for HuggingFace token input (password echo mode)
- Integrated with existing `sync_ai_controls()` for enable/disable logic:
  - Token field only enabled when diarization checkbox is checked
  - Both require `transcribe_speech` master toggle to be enabled
- On save, persists `diarize_speakers` and `huggingface_token` to config

#### `README.md`
Added comprehensive "Speaker diarization with Pyannote (optional)" section:
- Feature description and use cases
- Step-by-step HuggingFace token setup (with direct links)
- Model download/caching instructions
- Preferences UI walkthrough
- Technical notes (16kHz mono requirement, model loading behavior, VRAM cleanup)
- Token security notes

---

## How It Works

### Recording → Transcription → Speaker Labels

1. **User enables** diarization in Preferences + provides HF token
2. **Recording ends** → ffmpeg produces MKV video + FLAC audio
3. **AI pipeline starts**:
   - Audio (FLAC at any sample rate) → **16kHz mono WAV** (auto-converted for Pyannote)
   - Whisper transcribes with time-stamped segments
   - Pyannote diarizes the 16kHz WAV → speaker segments
   - Segments matched by time → transcript text labeled with speaker IDs
4. **Output**: `transcript.txt` with speaker labels
   ```
   [Speaker_1] Good morning, everyone.
   [Speaker_2] Hi! How are you?
   [Speaker_1] Great, thanks for joining.
   ```

### Audio Format Requirement

Pyannote 3.1 is **highly sample-rate sensitive** → accuracy drops significantly if not 16kHz mono.

The pipeline handles this automatically:
- **Input**: Any ffmpeg-decodable format (MKV, MP4, WAV, etc.)
- **Conversion**: ffmpeg resamples + converts to PCM mono at 16kHz
- **Pyannote**: Processes standardized 16kHz audio
- **Cleanup**: Temporary 16kHz WAV deleted after diarization

---

## Quick Start

### 1. Get HuggingFace Token

```bash
# 1. Go to: https://huggingface.co/settings/tokens
# 2. Click "New token" → name: daily-sync-agent, role: read
# 3. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1
# 4. Copy your token (starts with hf_)
```

### 2. Install Pyannote Models

```bash
./scripts/install-pyannote-models.sh --hf-token "hf_xxxxx"

# Or via environment variable
export HF_TOKEN="hf_xxxxx"
./scripts/install-pyannote-models.sh

# Dry-run (preview without downloading)
./scripts/install-pyannote-models.sh --hf-token "hf_xxxxx" --dry-run
```

### 3. Enable in App

1. Start `daily-sync-agent`
2. Right-click tray → **Preferences**
3. Check **"Speaker diarization (Pyannote)"**
4. Paste token in **"HuggingFace token"** field
5. Click **OK**

### 4. Use It

```bash
# Record normally (Shift+click to toggle recording)
# When recording ends, AI pipeline runs:
# - Transcription with Whisper
# - Diarization with Pyannote
# - Speaker labels merged into transcript.txt
```

---

## Configuration (Advanced)

### Config File Location
`~/.config/daily-sync-agent/config.json`

### Example Config
```json
{
  "transcribe_speech": true,
  "diarize_speakers": true,
  "huggingface_token": "hf_xxxxx",
  "unload_models_after_task": true,
  "whisper_model": "base",
  "ollama_model": "daily-sync-summary",
  ...
}
```

### Disabling Features
- **Diarization off**: Leave `diarize_speakers: false` (default)
- **AI completely off**: Set `transcribe_speech: false`
- **Token refresh**: Update `huggingface_token` in config or Preferences

### VRAM Management
When `unload_models_after_task` is enabled (recommended for small GPUs):
- Whisper model unloaded after transcription
- Pyannote model unloaded after diarization
- Before summarization starts, VRAM is cleared to avoid OOM with Ollama

---

## Testing

Run diarization tests:

```bash
# Syntax validation
python3 -m py_compile tests/test_diarization.py

# Run tests (requires installed dependencies)
python3 -m unittest tests.test_diarization -v
```

Test coverage:
- ✅ Empty diarization fallback
- ✅ Speaker label merging
- ✅ Segment time overlap matching
- ✅ Empty segment filtering
- ✅ Audio format conversion
- ✅ Error handling

---

## Troubleshooting

### "Token required" error
```bash
# Ensure token is set
export HF_TOKEN="hf_xxxxx"
./scripts/install-pyannote-models.sh

# Or in Preferences: paste token, save, retry recording
```

### "Model not found" / "License not accepted"
1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
2. Click **"Agree and access repository"**
3. Generate new token with **read** access
4. Update token in Preferences

### Diarization slow on first run
- **Expected**: Pyannote model (~1-2 GB) loads first time (few seconds)
- **Subsequent runs**: Cached in memory, faster

### "ffmpeg failed to convert audio"
- Ensure `ffmpeg` is on PATH
- Check audio file is not corrupted: `ffmpeg -i recording.flac -f null -`

### Low accuracy / incorrect speaker labels
- Verify audio is clear (no background noise)
- Ensure at least 2 speakers with distinct voices
- Pyannote works best with speech longer than 1-2 seconds per speaker

---

## Integration Points

### With Whisper (Transcription)
- Happens before diarization
- Produces time-stamped segments
- Segments matched with Pyannote output by time overlap

### With Ollama (Summarization)
- Happens after diarization
- Summary input includes speaker labels (e.g., `[Speaker_1] Important point`)
- Ollama processes speaker-tagged text naturally for better context

### With Low-VRAM Mode
- Pyannote model unloaded after diarization
- Before Ollama starts, VRAM is freed
- nvidia-smi monitoring (if available) waits for VRAM release

---

## Files Summary

| File | Size | Purpose |
|------|------|---------|
| `src/daily_sync_agent/ai/diarize.py` | 5.2 KB | Core diarization logic |
| `scripts/install-pyannote-models.sh` | 4.3 KB | Model installer + token setup |
| `tests/test_diarization.py` | 4.8 KB | Unit tests |
| `src/daily_sync_agent/settings.py` | ✏️ | +2 config fields |
| `src/daily_sync_agent/ai/transcribe.py` | ✏️ | +diarization params |
| `src/daily_sync_agent/ai/pipeline.py` | ✏️ | +diarization config passing |
| `src/daily_sync_agent/app.py` | ✏️ | +Preferences UI controls |
| `README.md` | ✏️ | +Pyannote setup guide |

---

## Next Steps

1. **Test locally**: `./scripts/install-pyannote-models.sh --dry-run`
2. **Get HF token**: https://huggingface.co/settings/tokens
3. **Install models**: `./scripts/install-pyannote-models.sh --hf-token "your_token"`
4. **Enable in app**: Preferences → check diarization + paste token
5. **Record & transcribe**: Observe speaker labels in `transcript.txt`

---

## Notes for Future Maintenance

- **Pyannote 3.1 models** are hosted on Hugging Face (no local build required)
- **Token lifetime**: Indefinite unless manually revoked
- **Model updates**: Detected automatically by Pyannote SDK (no manual action)
- **Audio conversion**: Always 16kHz mono for consistency (not cached, regenerated per task)
- **Speaker ID format**: `Speaker_1`, `Speaker_2`, ... (numeric IDs assigned by Pyannote)

