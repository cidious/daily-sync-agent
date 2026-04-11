#!/usr/bin/env bash
set -euo pipefail

APP_NAME="daily-sync-agent"
INSTALL_ROOT="/opt/${APP_NAME}"
INSTALL_SRC="${INSTALL_ROOT}/src"
VENV_DIR="${INSTALL_ROOT}/.venv"
BIN_LINK="/usr/local/bin/${APP_NAME}"
DESKTOP_SRC_REL="packaging/${APP_NAME}.desktop"
DESKTOP_DST="/usr/share/applications/${APP_NAME}.desktop"

usage() {
  cat <<'EOF'
Install Daily sync agent system-wide.

Usage:
  scripts/install-system-app.sh [--with-deps]
  scripts/install-system-app.sh --uninstall

Options:
  --with-deps   Run scripts/install-system-dependencies.sh before installing.
  --uninstall   Remove system-wide app files, launcher, and symlink.
  -h, --help    Show this help.

Install layout:
  /opt/daily-sync-agent/src      copied repository snapshot
  /opt/daily-sync-agent/.venv    runtime virtualenv
  /usr/local/bin/daily-sync-agent -> /opt/.../.venv/bin/daily-sync-agent
  /usr/share/applications/daily-sync-agent.desktop
EOF
}

need_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    exec sudo "$0" "$@"
  fi
}

run_update_desktop_db() {
  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
  fi
}

uninstall_app() {
  rm -f "${BIN_LINK}" "${DESKTOP_DST}"
  rm -rf "${INSTALL_ROOT}"
  run_update_desktop_db
  echo "Removed ${APP_NAME} from system paths."
}

copy_repo_snapshot() {
  local repo_root="$1"
  mkdir -p "${INSTALL_SRC}"
  find "${INSTALL_SRC}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  tar \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.idea' \
    --exclude='.ruff_cache' \
    -C "${repo_root}" -cf - . | tar -C "${INSTALL_SRC}" -xf -
}

install_app() {
  local repo_root="$1"

  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install --upgrade pip wheel
  "${VENV_DIR}/bin/pip" install "${INSTALL_SRC}"

  ln -sfn "${VENV_DIR}/bin/${APP_NAME}" "${BIN_LINK}"

  install -Dm644 "${INSTALL_SRC}/${DESKTOP_SRC_REL}" "${DESKTOP_DST}"
  chmod 644 "${DESKTOP_DST}"
  run_update_desktop_db

  echo "Installed ${APP_NAME}."
  echo "Launcher: ${DESKTOP_DST}"
  echo "Binary:   ${BIN_LINK}"
}

main() {
  local with_deps=0
  local uninstall=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-deps)
        with_deps=1
        ;;
      --uninstall)
        uninstall=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done

  need_root "$@"

  local script_dir repo_root deps_script
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/.." && pwd)"
  deps_script="${repo_root}/scripts/install-system-dependencies.sh"

  if [[ ${uninstall} -eq 1 ]]; then
    uninstall_app
    exit 0
  fi

  if [[ ${with_deps} -eq 1 ]]; then
    "${deps_script}"
  fi

  mkdir -p "${INSTALL_ROOT}"
  copy_repo_snapshot "${repo_root}"
  install_app "${repo_root}"
}

main "$@"

