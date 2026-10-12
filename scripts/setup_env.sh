#!/usr/bin/env bash
# One-shot environment setup (no sudo needed): uv, Python deps (+CUDA torch),
# and a static ffmpeg build with libvmaf in ~/.local/opt/ffmpeg-vmaf.
# Requires a system ffmpeg/ffprobe with libx264/libx265/libsvtav1 (apt install ffmpeg).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

"$UV" sync --extra torch

VMAF_DIR="$HOME/.local/opt/ffmpeg-vmaf"
if [ ! -x "$VMAF_DIR/bin/ffmpeg" ]; then
  tmp="$(mktemp -d)"
  curl -LsSf -o "$tmp/ffmpeg.tar.xz" \
    https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
  tar -C "$tmp" -xf "$tmp/ffmpeg.tar.xz"
  mkdir -p "$(dirname "$VMAF_DIR")"
  mv "$tmp/ffmpeg-master-latest-linux64-gpl" "$VMAF_DIR"
  rm -rf "$tmp"
fi

"$UV" run vreg doctor
