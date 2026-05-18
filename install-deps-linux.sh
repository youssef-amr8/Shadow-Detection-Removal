#!/usr/bin/env bash
# Install system prerequisites (Debian/Ubuntu) and Python packages for this project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if command -v apt-get >/dev/null 2>&1 && [ "${SKIP_APT:-0}" != "1" ]; then
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
    echo "Warning: Prefer Python 3.10–3.13 for broad PyTorch wheel support (you have $PYVER)." >&2
    ;;
esac

# Use Linux-native temp dir (avoids I/O errors unpacking 500MB+ wheels on /tmp or /mnt/d).
export TMPDIR="${TMPDIR:-$HOME/.cache/dl-project-tmp}"
mkdir -p "$TMPDIR"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip

echo "Installing Python deps (numpy, opencv, …)…"
pip install --default-timeout=300 -r requirements.txt

# Default: CPU-only PyTorch (smaller wheel, fewer WSL /tmp failures). Set INSTALL_TORCH_CUDA=1 for GPU build.
if [ "${INSTALL_TORCH_CUDA:-0}" = "1" ]; then
  echo "Installing PyTorch with CUDA (large download)…"
  pip install --default-timeout=600 torch torchvision
else
  echo "Installing PyTorch CPU wheels from pytorch.org (set INSTALL_TORCH_CUDA=1 for GPU)…"
  pip install --default-timeout=600 torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "Done. Activate: source .venv/bin/activate"
