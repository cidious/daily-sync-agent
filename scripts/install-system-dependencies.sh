#!/usr/bin/env bash
# Install system packages required to run daily-sync-agent (ffmpeg, pactl, X11 helpers, Python).
# Optionally installs CUDA 12 *user-space* libraries (libcublas, libcudart, libnvrtc) for GPU Whisper
# (faster-whisper / CTranslate2). Does not install the NVIDIA driver — install that separately.
# Run from anywhere; uses sudo where needed. See README for optional Ollama.

set -euo pipefail

WITH_OLLAMA=0
WITH_VENV=0
# 1 = install CUDA 12 runtime libs on apt-based distros (Ubuntu multiverse / Debian non-free as needed).
WITH_CUDA_RUNTIME=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Set NONINTERACTIVE=1 to skip the Ollama confirmation prompt (Ollama will not be installed unless WITH_OLLAMA=1 and you use --yes-ollama).
YES_OLLAMA=0

usage() {
  cat <<'EOF'
Usage: install-system-dependencies.sh [options]

  Installs distribution packages so this app can run:
    ffmpeg, pactl (PulseAudio/PipeWire CLI), xdotool, xwininfo,
    Python 3.11+ (where available), venv and pip helpers.

Options:
  --with-ollama   Offer to run the official Ollama install script (https://ollama.com) after packages.
  --yes-ollama    With --with-ollama, run the Ollama installer without prompting (non-interactive).
  --with-cuda-runtime
                  On Debian/Ubuntu (apt), install libcublas12, libcudart12, libnvrtc12 for GPU Whisper.
                  (Default: on; requires Ubuntu multiverse or Debian non-free where applicable.)
  --skip-cuda-runtime
                  Do not install CUDA user-space libraries.
  --venv          After system packages, create .venv in the repo and: pip install -e .
  -h, --help      Show this help.

Environment:
  NONINTERACTIVE=1   Same as --yes-ollama when combined with --with-ollama; skips prompts.

Examples:
  ./scripts/install-system-dependencies.sh
  ./scripts/install-system-dependencies.sh --venv
  ./scripts/install-system-dependencies.sh --skip-cuda-runtime
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-ollama) WITH_OLLAMA=1 ;;
    --yes-ollama) YES_OLLAMA=1 ;;
    --with-cuda-runtime) WITH_CUDA_RUNTIME=1 ;;
    --skip-cuda-runtime) WITH_CUDA_RUNTIME=0 ;;
    --venv) WITH_VENV=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Do not run this script as root; it will use sudo when needed." >&2
  exit 1
fi

have() { command -v "$1" >/dev/null 2>&1; }

if ! have sudo; then
  echo "sudo is required to install system packages." >&2
  exit 1
fi

# shellcheck source=/dev/null
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
else
  echo "Cannot detect OS (missing /etc/os-release)." >&2
  exit 1
fi

ID_LIKE="${ID_LIKE:-}"
PKG=()
# Which package manager ran the main install (for CUDA hints).
INSTALLER_KIND="unknown"

# Ubuntu: libcublas12 lives in multiverse. Debian: packages are often in contrib/non-free.
ensure_apt_cuda_repos() {
  [[ "${WITH_CUDA_RUNTIME}" -eq 1 ]] || return 0
  case "${ID:-}" in
    ubuntu|linuxmint|pop)
      sudo apt-get install -y software-properties-common
      sudo add-apt-repository -y multiverse 2>/dev/null || true
      ;;
    debian)
      if ! grep -qE '^[^#].*\scontrib' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null; then
        echo "Note: For libcublas12 on Debian, enable contrib and non-free in apt sources if install fails." >&2
      fi
      ;;
  esac
}

install_cuda_runtime_apt() {
  [[ "${WITH_CUDA_RUNTIME}" -eq 1 ]] || return 0
  echo "Installing CUDA 12 user-space libraries (Whisper GPU / libcublas.so.12)..."
  sudo apt-get install -y \
    libcublas12 \
    libcudart12 \
    libnvrtc12
}

install_apt() {
  if [[ "${WITH_CUDA_RUNTIME}" -eq 1 ]]; then
    ensure_apt_cuda_repos
  fi
  sudo apt-get update -y
  sudo apt-get install -y \
    ffmpeg \
    pulseaudio-utils \
    xdotool \
    x11-utils \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential
  if [[ "${WITH_CUDA_RUNTIME}" -eq 1 ]]; then
    install_cuda_runtime_apt
  fi
}

install_dnf() {
  # Fedora / RHEL 8+ / modern CentOS Stream
  sudo dnf install -y \
    ffmpeg \
    pulseaudio-utils \
    xdotool \
    xorg-x11-utils \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    make
}

install_yum() {
  sudo yum install -y \
    ffmpeg \
    pulseaudio-utils \
    xdotool \
    xorg-x11-utils \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    make
}

install_pacman() {
  sudo pacman -Sy --needed --noconfirm \
    ffmpeg \
    pulseaudio-utils \
    xdotool \
    xorg-xwininfo \
    python \
    python-pip \
    base-devel
}

install_zypper() {
  sudo zypper install -y \
    ffmpeg \
    pulseaudio-utils \
    xdotool \
    xorg-x11-tools \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    make
}

install_apk() {
  sudo apk add --no-cache \
    ffmpeg \
    pulseaudio-utils \
    xdotool \
    xwininfo \
    python3 \
    py3-pip \
    py3-virtualenv \
    build-base
}

case "${ID:-}" in
  debian|ubuntu|linuxmint|pop)
    echo "Using apt (${PRETTY_NAME:-$ID})..."
    INSTALLER_KIND=apt
    install_apt
    ;;
  fedora|nobara)
    echo "Using dnf (${PRETTY_NAME:-$ID})..."
    INSTALLER_KIND=dnf
    install_dnf
    ;;
  rhel|centos|rocky|almalinux)
    if have dnf; then
      echo "Using dnf (${PRETTY_NAME:-$ID})..."
      INSTALLER_KIND=dnf
      install_dnf
    else
      echo "Using yum (${PRETTY_NAME:-$ID})..."
      INSTALLER_KIND=yum
      install_yum
    fi
    ;;
  arch|manjaro|endeavouros)
    echo "Using pacman (${PRETTY_NAME:-$ID})..."
    INSTALLER_KIND=pacman
    install_pacman
    ;;
  opensuse-leap|opensuse-tumbleweed|suse)
    echo "Using zypper (${PRETTY_NAME:-$ID})..."
    INSTALLER_KIND=zypper
    install_zypper
    ;;
  alpine)
    echo "Using apk (${PRETTY_NAME:-$ID})..."
    INSTALLER_KIND=apk
    install_apk
    ;;
  *)
    if [[ " $ID_LIKE " == *" debian "* ]] || [[ " $ID_LIKE " == *" ubuntu "* ]]; then
      echo "Using apt (ID_LIKE debian/ubuntu: ${PRETTY_NAME:-$ID})..."
      INSTALLER_KIND=apt
      install_apt
    elif [[ " $ID_LIKE " == *" rhel "* ]] || [[ " $ID_LIKE " == *" fedora "* ]]; then
      if have dnf; then
        echo "Using dnf (ID_LIKE rhel/fedora: ${PRETTY_NAME:-$ID})..."
        INSTALLER_KIND=dnf
        install_dnf
      else
        echo "Using yum (ID_LIKE rhel/fedora: ${PRETTY_NAME:-$ID})..."
        INSTALLER_KIND=yum
        install_yum
      fi
    elif [[ " $ID_LIKE " == *" arch "* ]]; then
      echo "Using pacman (ID_LIKE arch: ${PRETTY_NAME:-$ID})..."
      INSTALLER_KIND=pacman
      install_pacman
    else
      echo "Unsupported distribution: ID=${ID:-unknown} VERSION_ID=${VERSION_ID:-} ID_LIKE=${ID_LIKE:-}" >&2
      echo "Install manually: ffmpeg, a package providing pactl (e.g. pulseaudio-utils), xdotool, xwininfo, Python 3.11+ with pip/venv." >&2
      exit 1
    fi
    ;;
esac

if [[ "${WITH_CUDA_RUNTIME}" -eq 1 && "${INSTALLER_KIND}" != "apt" ]]; then
  echo ""
  echo "GPU Whisper (CUDA): this script auto-installs libcublas12 / libcudart12 / libnvrtc12 on Debian/Ubuntu (apt) only."
  echo "On ${PRETTY_NAME:-$ID}, install CUDA 12 user-space libraries from your distribution or NVIDIA so libcublas.so.12 resolves:"
  echo "  https://developer.nvidia.com/cuda-downloads"
  echo "(The proprietary NVIDIA driver must be installed separately for GPU use.)"
fi

echo ""
echo "Verifying key commands..."
MISSING=()
have ffmpeg || MISSING+=("ffmpeg")
have pactl || MISSING+=("pactl")
have xdotool || MISSING+=("xdotool")
have xwininfo || MISSING+=("xwininfo")
if ((${#MISSING[@]})); then
  echo "Warning: still missing on PATH: ${MISSING[*]}" >&2
else
  echo "ffmpeg, pactl, xdotool, xwininfo: OK"
fi

py_ok=0
if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  echo "python3: $(python3 --version) (>= 3.11 OK)"
  py_ok=1
else
  echo "Warning: python3 is below 3.11 or missing. Install Python 3.11+ for this project (see README)." >&2
fi

if [[ "${WITH_CUDA_RUNTIME}" -eq 1 && "${INSTALLER_KIND}" == "apt" ]]; then
  echo ""
  if have ldconfig && ldconfig -p 2>/dev/null | grep -qF 'libcublas.so.12'; then
    echo "CUDA user-space: libcublas.so.12 is registered with the dynamic linker (ldconfig)."
  else
    echo "Warning: libcublas.so.12 not found in ldconfig -p output. GPU Whisper may still fall back to CPU." >&2
    echo "  Try: sudo ldconfig   and ensure libcublas12 is installed (Ubuntu: multiverse)." >&2
  fi
fi

if [[ "$WITH_OLLAMA" -eq 1 ]]; then
  echo ""
  run_ollama=0
  if [[ "$YES_OLLAMA" -eq 1 || "${NONINTERACTIVE:-0}" == "1" ]]; then
    run_ollama=1
  elif [[ -t 0 ]]; then
    read -r -p "Install Ollama via https://ollama.com/install.sh ? [y/N] " ans
    if [[ "${ans,,}" == "y" || "${ans,,}" == "yes" ]]; then
      run_ollama=1
    fi
  else
    echo "Not a TTY; skipping Ollama (use --yes-ollama or NONINTERACTIVE=1 to auto-install)." >&2
  fi
  if [[ "$run_ollama" -eq 1 ]]; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    echo "Skipped Ollama install."
  fi
fi

if [[ "$WITH_VENV" -eq 1 ]]; then
  if [[ "$py_ok" -ne 1 ]]; then
    echo "Skipping venv: need Python >= 3.11." >&2
  else
    echo ""
    echo "Creating venv and installing daily-sync-agent in editable mode..."
    python3 -m venv "${REPO_ROOT}/.venv"
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
    python -m pip install -U pip wheel
    pip install -e "${REPO_ROOT}"
    echo "Done. Run: source ${REPO_ROOT}/.venv/bin/activate && daily-sync-agent"
  fi
fi

echo ""
echo "System dependency install finished."
