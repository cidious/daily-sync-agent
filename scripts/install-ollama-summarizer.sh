#!/usr/bin/env bash
# Auto-pick and install an Ollama summarization model for long transcript summaries (~32k context).
# Detects GPU/VRAM, chooses a model family tier, generates a tuned Modelfile, and creates a local alias.

set -euo pipefail

FAMILY="auto"            # auto|qwen|gemma3|mistral-small|mistral-nemo|llama
PINNED_MODEL=""          # exact model ref, e.g. qwen2.5:14b
ALLOW_CPU_OFFLOAD=0       # 1 = allow selecting a model larger than fully fitting VRAM
PREFER_BIGGER=0           # 1 = with offload enabled, bias to next larger tier
CTX_TOKENS=32768          # long meetings/transcripts
MODEL_ALIAS="daily-sync-summary"
MODEFILE_PATH="${PWD}/Modelfile.daily-sync-summarizer"
BASE_URL=""              # optional, exported as OLLAMA_HOST
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: install-ollama-summarizer.sh [options]

Detect hardware and install a long-context summarization model alias for daily-sync-agent.

Options:
  --family <name>              Model family: auto|qwen|gemma3|mistral-small|mistral-nemo|llama
  --model <name:tag>           Pin exact model (overrides --family/auto tier selection)
  --allow-cpu-offload          Allow selecting models larger than VRAM (partial GPU + CPU offload)
  --prefer-bigger              With --allow-cpu-offload, prefer next larger tier for better quality
  --ctx <tokens>               Context size for Modelfile (default: 32768)
  --alias <name>               New local Ollama model name (default: daily-sync-summary)
  --modelfile-path <path>      Where to write generated Modelfile
  --base-url <url>             Ollama base URL; exported as OLLAMA_HOST for pull/create
  --dry-run                    Print selection and Modelfile without pulling/creating model
  -h, --help                   Show this help

Examples:
  ./scripts/install-ollama-summarizer.sh
  ./scripts/install-ollama-summarizer.sh --family gemma3 --allow-cpu-offload --prefer-bigger
  ./scripts/install-ollama-summarizer.sh --model qwen2.5:14b --allow-cpu-offload
  ./scripts/install-ollama-summarizer.sh --dry-run --family auto --ctx 32768
EOF
}

have() { command -v "$1" >/dev/null 2>&1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family) FAMILY="${2:-}"; shift ;;
    --model) PINNED_MODEL="${2:-}"; shift ;;
    --allow-cpu-offload) ALLOW_CPU_OFFLOAD=1 ;;
    --prefer-bigger) PREFER_BIGGER=1 ;;
    --ctx) CTX_TOKENS="${2:-}"; shift ;;
    --alias) MODEL_ALIAS="${2:-}"; shift ;;
    --modelfile-path) MODEFILE_PATH="${2:-}"; shift ;;
    --base-url) BASE_URL="${2:-}"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

case "$FAMILY" in
  auto|qwen|gemma3|mistral-small|mistral-nemo|llama) ;;
  *) echo "Unsupported --family: $FAMILY" >&2; exit 1 ;;
esac

if [[ ! "$CTX_TOKENS" =~ ^[0-9]+$ ]] || [[ "$CTX_TOKENS" -lt 8192 ]]; then
  echo "--ctx must be an integer >= 8192" >&2
  exit 1
fi

if [[ -n "$PINNED_MODEL" ]]; then
  PINNED_MODEL="${PINNED_MODEL//[[:space:]]/}"
  if [[ -z "$PINNED_MODEL" || "$PINNED_MODEL" != *:* ]]; then
    echo "--model must be in form name:tag (example: qwen2.5:14b)" >&2
    exit 1
  fi
fi

if [[ -n "$BASE_URL" ]]; then
  export OLLAMA_HOST="$BASE_URL"
fi

if [[ "$DRY_RUN" -ne 1 ]] && ! have ollama; then
  echo "ollama CLI is required. Install Ollama first (see README)." >&2
  exit 1
fi

# Returns two fields: vendor vram_mb
probe_gpu() {
  local vram_mb="0"
  local vendor="none"

  if have nvidia-smi; then
    # Choose the largest VRAM among visible GPUs.
    vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print m+0}')"
    if [[ "${vram_mb:-0}" -gt 0 ]]; then
      vendor="nvidia"
      echo "$vendor $vram_mb"
      return
    fi
  fi

  if have rocm-smi; then
    # rocm-smi --showmeminfo vram prints bytes; grab max seen value.
    local bytes
    bytes="$(rocm-smi --showmeminfo vram 2>/dev/null | awk -F': ' '/Total Memory \(B\)/ {gsub(/[^0-9]/,"",$2); if($2+0>m)m=$2+0} END{print m+0}')"
    if [[ "${bytes:-0}" -gt 0 ]]; then
      vram_mb="$((bytes / 1024 / 1024))"
      vendor="amd"
      echo "$vendor $vram_mb"
      return
    fi
  fi

  echo "$vendor $vram_mb"
}

# Candidate rows: family:model:param_b:layers:need_gb_for_32k_ctx
CANDIDATES=(
  "qwen:qwen2.5:3b:3:28:5"
  "qwen:qwen2.5:7b:7:32:10"
  "qwen:qwen2.5:14b:14:40:18"
  "qwen:qwen2.5:32b:32:64:40"
  "gemma3:gemma3:4b:4:34:6"
  "gemma3:gemma3:12b:12:48:16"
  "gemma3:gemma3:27b:27:62:34"
  "mistral-small:mistral-small:24b:24:60:30"
  "mistral-nemo:mistral-nemo:12b:12:40:16"
  "llama:llama3.1:8b:8:32:11"
  "llama:llama3.1:70b:70:80:80"
)

# Print best row for given family+vram. If no fit, prints smallest in family.
select_best_fit() {
  local family="$1"
  local vram_gb="$2"
  local row f model tag pb layers need best="" smallest=""

  for row in "${CANDIDATES[@]}"; do
    IFS=':' read -r f model tag pb layers need <<<"$row"
    if [[ "$family" != "auto" && "$f" != "$family" ]]; then
      continue
    fi
    if [[ "$family" == "auto" && "$f" != "qwen" ]]; then
      continue
    fi

    if [[ -z "$smallest" ]]; then
      smallest="$row"
    fi

    if (( vram_gb >= need )); then
      best="$row"
    fi
  done

  if [[ -n "$best" ]]; then
    echo "$best"
  else
    echo "$smallest"
  fi
}

# Print next larger row in same family after selected row (if any).
select_next_larger() {
  local selected="$1"
  local family="$2"
  local found=0 row f model tag pb layers need

  for row in "${CANDIDATES[@]}"; do
    IFS=':' read -r f model tag pb layers need <<<"$row"
    if [[ "$f" != "$family" ]]; then
      continue
    fi
    if [[ "$found" -eq 1 ]]; then
      echo "$row"
      return
    fi
    if [[ "$row" == "$selected" ]]; then
      found=1
    fi
  done

  echo "$selected"
}

read -r GPU_VENDOR VRAM_MB <<<"$(probe_gpu)"
VRAM_GB=$((VRAM_MB / 1024))

SELECTED_FAMILY="manual"
BASE_MODEL=""
BASE_TAG=""
PARAM_B=""
LAYERS=0
NEED_GB=0
BASE_REF=""

if [[ -n "$PINNED_MODEL" ]]; then
  BASE_REF="$PINNED_MODEL"
  for row in "${CANDIDATES[@]}"; do
    IFS=':' read -r f model tag pb layers need <<<"$row"
    if [[ "${model}:${tag}" == "$BASE_REF" ]]; then
      SELECTED_FAMILY="$f"
      BASE_MODEL="$model"
      BASE_TAG="$tag"
      PARAM_B="$pb"
      LAYERS="$layers"
      NEED_GB="$need"
      break
    fi
  done
else
  BEST_ROW="$(select_best_fit "$FAMILY" "$VRAM_GB")"
  IFS=':' read -r SELECTED_FAMILY BASE_MODEL BASE_TAG PARAM_B LAYERS NEED_GB <<<"$BEST_ROW"

  if [[ "$ALLOW_CPU_OFFLOAD" -eq 1 && "$PREFER_BIGGER" -eq 1 ]]; then
    BIGGER_ROW="$(select_next_larger "$BEST_ROW" "$SELECTED_FAMILY")"
    IFS=':' read -r _f _m _t _p _l _n <<<"$BIGGER_ROW"
    SELECTED_FAMILY="$_f"
    BASE_MODEL="$_m"
    BASE_TAG="$_t"
    PARAM_B="$_p"
    LAYERS="$_l"
    NEED_GB="$_n"
  fi

  BASE_REF="${BASE_MODEL}:${BASE_TAG}"
fi

NUM_GPU=0
OFFLOAD_MODE="cpu"
 if [[ "$VRAM_GB" -gt 0 ]]; then
  if [[ "$NEED_GB" -gt 0 ]] && (( VRAM_GB >= NEED_GB )); then
    NUM_GPU="$LAYERS"
    OFFLOAD_MODE="full_gpu"
  elif [[ "$NEED_GB" -gt 0 && "$ALLOW_CPU_OFFLOAD" -eq 1 ]]; then
    # Reserve ~1GB for runtime and map remaining VRAM to layer share.
    local_usable=$((VRAM_GB - 1))
    if (( local_usable < 1 )); then
      local_usable=1
    fi
    NUM_GPU=$((LAYERS * local_usable / NEED_GB))
    if (( NUM_GPU < 4 )); then
      NUM_GPU=4
    fi
    if (( NUM_GPU > LAYERS )); then
      NUM_GPU="$LAYERS"
    fi
    OFFLOAD_MODE="partial_gpu_cpu_offload"
  elif [[ -n "$PINNED_MODEL" ]]; then
    NUM_GPU=0
    OFFLOAD_MODE="manual_model_default_gpu_layers"
   else
     NUM_GPU=0
     OFFLOAD_MODE="cpu_due_to_vram_limit"
   fi
 fi

mkdir -p "$(dirname "$MODEFILE_PATH")"
cat >"$MODEFILE_PATH" <<EOF
FROM $BASE_REF

# Long-context summaries for transcripts up to ~32k tokens.
PARAMETER num_ctx $CTX_TOKENS
PARAMETER num_predict 1024
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_gpu $NUM_GPU
EOF

echo "Detected GPU vendor: $GPU_VENDOR"
echo "Detected VRAM: ${VRAM_MB} MB (${VRAM_GB} GB)"
echo "Selected family: $SELECTED_FAMILY"
echo "Selected base model: $BASE_REF"
if [[ -n "$PINNED_MODEL" ]]; then
  echo "Selection mode: pinned (--model)"
fi
echo "Offload mode: $OFFLOAD_MODE"
echo "Generated Modelfile: $MODEFILE_PATH"
echo "Suggested alias for daily-sync-agent: $MODEL_ALIAS"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ""
  echo "Dry run only; skipped ollama pull/create."
  echo ""
  cat "$MODEFILE_PATH"
  exit 0
fi

ollama pull "$BASE_REF"
ollama create "$MODEL_ALIAS" -f "$MODEFILE_PATH"

echo ""
echo "Model alias created: $MODEL_ALIAS"
echo "Use these app settings in ~/.config/daily-sync-agent/config.json (or Preferences):"
echo "  ollama_model = $MODEL_ALIAS"
echo "  ollama_request_timeout_s = 1800"
if (( VRAM_GB > 0 && VRAM_GB < 16 )); then
  echo "  unload_models_after_task = true"
else
  echo "  unload_models_after_task = false"
fi
echo "  summarize_transcript = true"
echo "  summary_mode = daily_scrum"

