#!/bin/bash
# =============================================================
#  JesterClaw — Model Downloader
#  Team Lapanic / EmolOrbit
#  Downloads Qwen2.5-Omni-3B into ./Model Files/
# =============================================================

set -e

MODEL_ID="Qwen/Qwen2.5-Omni-3B"
MODEL_DIR="./Model Files/Qwen2.5-Omni-3B"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║       JesterClaw — Model Downloader          ║"
echo "║       Team Lapanic / EmolOrbit               ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

mkdir -p "$MODEL_DIR"

echo "[*] Model: $MODEL_ID"
echo "[*] Destination: $MODEL_DIR"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "[*] Installing huggingface_hub CLI..."
    pip install huggingface_hub
fi

echo "[*] Starting download (this may take 10-30 mins depending on connection)..."
huggingface-cli download "$MODEL_ID" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False \
    --repo-type model

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║     Model downloaded successfully!           ║"
echo "║     Start server: python Main.py             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
