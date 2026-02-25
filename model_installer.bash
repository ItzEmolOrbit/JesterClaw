#!/bin/bash
# =============================================================
#  JesterClaw — Model Downloader (llama.cpp GGUF)
#  Team Lapanic / EmolOrbit
#
#  Downloads:
#   1. GGUF model: yuhong123/Qwen2.5-Omni-3B-Q4_K_M-GGUF
#   2. Whisper model for STT (faster-whisper small.en or base)
# =============================================================

set -e

GGUF_REPO="yuhong123/Qwen2.5-Omni-3B-Q4_K_M-GGUF"
GGUF_FILE="qwen2.5-omni-3b-q4_k_m.gguf"
MODEL_DIR="./Model Files"
GGUF_PATH="$MODEL_DIR/$GGUF_FILE"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║    JesterClaw — GGUF Model Downloader        ║"
echo "║    Team Lapanic / EmolOrbit                  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

mkdir -p "$MODEL_DIR"

# --- Check huggingface-cli ---
if ! command -v huggingface-cli &> /dev/null; then
    echo "[*] Installing huggingface_hub..."
    pip install huggingface_hub
fi

# --- Download GGUF model ---
echo "[*] Downloading GGUF: $GGUF_REPO / $GGUF_FILE"
echo "[*] Destination: $GGUF_PATH"
echo "[*] Size: ~2.0 GB (Q4_K_M quantized)"
huggingface-cli download "$GGUF_REPO" "$GGUF_FILE" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "[*] GGUF model ready at: $GGUF_PATH"

# --- Download Whisper model for STT ---
# faster-whisper downloads automatically on first use,
# but we pre-download it here to avoid runtime delays.
echo ""
echo "[*] Pre-downloading Whisper 'base' model for STT..."
python3 -c "
from faster_whisper import WhisperModel
print('Downloading Whisper base model...')
WhisperModel('base', device='cpu', compute_type='int8')
print('Whisper model ready.')
"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║    All models downloaded successfully!       ║"
echo "║    GGUF: Model Files/$GGUF_FILE"
echo "║    STT:  Whisper (base)                      ║"
echo "║    Start: python Main.py                     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

