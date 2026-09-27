#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

MODEL_PATH="${1:-}"
STEPS="${2:-1500}"
FPS="${3:-20}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "Virtualenv not found at $VENV_DIR"
  echo "Run: bash scripts/setup.sh"
  exit 1
fi

cd "$ROOT_DIR"

if [[ -n "$MODEL_PATH" ]]; then
  exec "$PYTHON_BIN" record.py --model "$MODEL_PATH" --steps "$STEPS" --fps "$FPS"
else
  exec "$PYTHON_BIN" record.py --steps "$STEPS" --fps "$FPS"
fi

