#!/usr/bin/env bash
# Install Pyannote 3.1 speaker diarization models and accept Hugging Face model license.
# Requires: HuggingFace token with model access, ffmpeg, Python 3.11+, pip.

set -euo pipefail

HF_TOKEN=""
DRY_RUN=0
MODEL="pyannote/speaker-diarization-3.1"

usage() {
  cat <<'EOF'
Usage: install-pyannote-models.sh [options]

Download and cache Pyannote 3.1 speaker diarization models for daily-sync-agent.

Options:
  --hf-token <token>          HuggingFace API token (or set HF_TOKEN env var)
  --dry-run                   Show what would be done without downloading
  -h, --help                  Show this help

Environment:
  HF_TOKEN                    Your HuggingFace user access token (alternative to --hf-token)

Steps to get a token:
  1. Go to https://huggingface.co/settings/tokens
  2. Click "New token"
  3. Give it a name like "daily-sync-agent"
  4. Set "Role" to "read" (for model access)
  5. Accept Pyannote's model license at https://huggingface.co/pyannote/speaker-diarization-3.1
  6. Copy the token and use it: ./scripts/install-pyannote-models.sh --hf-token <your-token>

Example:
  ./scripts/install-pyannote-models.sh --hf-token hf_xxxxx
  # or: export HF_TOKEN=hf_xxxxx && ./scripts/install-pyannote-models.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token) HF_TOKEN="${2:-}"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

# Check env var if not set via flag
if [[ -z "$HF_TOKEN" ]]; then
  HF_TOKEN="${HF_TOKEN:-}"
fi

if [[ -z "$HF_TOKEN" ]]; then
  echo "Error: HuggingFace token required." >&2
  echo "" >&2
  usage >&2
  exit 1
fi

have() { command -v "$1" >/dev/null 2>&1; }

if ! have ffmpeg; then
  echo "ffmpeg is required. Install ffmpeg and retry." >&2
  exit 1
fi

if ! have python3; then
  echo "python3 is required. Install Python 3.11+ and retry." >&2
  exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
echo "Python version: $PY_VERSION"

echo "HuggingFace token: (***${HF_TOKEN: -8})"
echo "Model: $MODEL"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run mode:"
  echo "  1. Would verify HF token and model access"
  echo "  2. Would download and cache model files to ~/.cache/huggingface/hub/"
  echo "  3. Would show token setup instruction for Preferences"
  exit 0
fi

echo "Installing Pyannote and dependencies..."
pip install --upgrade pyannote.audio

echo ""
echo "Downloading Pyannote models to Hugging Face cache..."
export HF_TOKEN="$HF_TOKEN"
python3 << 'PYTHON_EOF'
import os
import sys
import warnings

# Suppress torchcodec/CUDA library warnings (not critical for CPU fallback)
warnings.filterwarnings("ignore", category=UserWarning)

from huggingface_hub import model_info

hf_token = os.environ.get("HF_TOKEN", "").strip()
model_name = "pyannote/speaker-diarization-3.1"

if not hf_token:
    print("Error: HF_TOKEN not set", file=sys.stderr)
    sys.exit(1)

try:
    info = model_info(model_name, token=hf_token)
    print(f"✓ Model accessible: {model_name}")
    print(f"  License: {info.siblings[0].rfilename if info.siblings else 'unknown'}")
except Exception as e:
    print(f"✗ Could not access model: {e}", file=sys.stderr)
    print("Ensure:", file=sys.stderr)
    print("  1. Token has 'read' access at https://huggingface.co/settings/tokens", file=sys.stderr)
    print("  2. You accepted the license at https://huggingface.co/pyannote/speaker-diarization-3.1", file=sys.stderr)
    sys.exit(1)

# Download model
print("\nCaching model files...")
from pyannote.audio import Pipeline
try:
    pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
    print(f"✓ Model cached successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

if [[ $? -ne 0 ]]; then
  echo ""
  echo "Failed to download/cache model. Check your token and model access." >&2
  exit 1
fi

echo ""
echo "✓ Pyannote models installed successfully!"
echo ""
echo "Next steps:"
echo "  1. Open daily-sync-agent Preferences"
echo "  2. Enable 'Speaker diarization (Pyannote)' checkbox"
echo "  3. Paste your HuggingFace token in the 'HuggingFace token' field"
echo "  4. Save Preferences"
echo "  5. Run a recording - speaker labels will appear in the transcript"
echo ""
echo "Note: First transcription with diarization may take longer (model is loaded once)."
echo "For best accuracy, Pyannote requires 16kHz mono audio (auto-converted by app)."

