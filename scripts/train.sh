#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "Virtualenv not found at $VENV_DIR"
  echo "Run: bash scripts/setup.sh"
  exit 1
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m training.train

