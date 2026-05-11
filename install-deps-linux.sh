#!/usr/bin/env bash
# Install system prerequisites (Debian/Ubuntu) and Python packages for this project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if command -v apt-get >/dev/null 2>&1; then
  echo "Installing apt packages (python3, venv, compilers, common OpenCV/Matplotlib libs)..."
  sudo apt-get update
  sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    build-essential \
    libgl1 \
    libglib2.0-0
fi

PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYVER" in
  3.1[0-3]) ;;
  *)
    echo "Warning: TensorFlow may not support Python $PYVER. Prefer 3.10–3.13." >&2
    ;;
esac

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Activate the environment with: source .venv/bin/activate"
