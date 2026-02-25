#!/bin/bash
# =============================================================
#  JesterClaw — Server Dependency Installer (Linux / Ubuntu)
#  Team Lapanic / EmolOrbit
#
#  The server runs on Linux Ubuntu with a CUDA GPU.
#  It only runs the AI model — no Windows automation here.
#  All OS actions are executed on the Windows client side.
#
#  Run this ONCE on your GPU Linux server.
# =============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║    JesterClaw — Server Dependency Setup      ║"
echo "║    Team Lapanic / EmolOrbit (Linux / GPU)    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# --- Python version check ---
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[*] Python version: $PYTHON_VERSION"

# --- Upgrade pip ---
echo "[*] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# --- Core Web Framework ---
echo "[*] Installing FastAPI & Uvicorn..."
pip install "fastapi>=0.110.0" "uvicorn[standard]>=0.29.0" websockets python-multipart

# --- PyTorch (CUDA 12.1 — change cu121→cu118 for CUDA 11.8) ---
echo "[*] Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- HuggingFace stack ---
echo "[*] Installing Transformers & Accelerate..."
pip install "transformers>=4.51.0" accelerate sentencepiece tokenizers

# --- Flash Attention 2 (Ampere+ GPU, requires gcc) ---
echo "[*] Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation || echo "[!] flash-attn failed — falling back to eager (still works)"

# --- Qwen Omni utilities ---
echo "[*] Installing qwen-omni-utils..."
pip install qwen-omni-utils

# --- Audio (for TTS/STT on the server) ---
echo "[*] Installing audio libraries..."
pip install soundfile librosa numpy scipy

# --- Image (for processing screenshots sent from client) ---
echo "[*] Installing image tools..."
pip install Pillow

# --- Utilities ---
echo "[*] Installing utilities..."
pip install psutil python-dotenv aiofiles httpx rich

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║    Dependencies installed successfully!      ║"
echo "║    Next: bash model_installer.bash           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
