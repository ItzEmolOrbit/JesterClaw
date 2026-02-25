"""
JesterClaw — Model Loader (llama-cpp-python GGUF)
Team Lapanic / EmolOrbit

Loads yuhong123/Qwen2.5-Omni-3B-Q4_K_M-GGUF using llama-cpp-python.
Text-only inference. No TTS. Voice input is handled client-side.

Singleton pattern — import get_llm() from anywhere in the server.
"""

import os
import logging
from pathlib import Path
from llama_cpp import Llama

logger = logging.getLogger("jesterclaw.model")

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR   = Path(__file__).parent
_MODEL_DIR  = _BASE_DIR / "Model Files"
_GGUF_FILE  = "qwen2.5-omni-3b-q4_k_m.gguf"
_GGUF_PATH  = _MODEL_DIR / _GGUF_FILE

# HuggingFace repo (used if local file not found)
_HF_REPO    = "yuhong123/Qwen2.5-Omni-3B-Q4_K_M-GGUF"
_HF_FILE    = _GGUF_FILE

# ── Config (overridable via env) ───────────────────────────────────────────────
N_GPU_LAYERS = int(os.getenv("JESTERCLAW_GPU_LAYERS", "-1"))   # -1 = all layers on GPU
N_CTX        = int(os.getenv("JESTERCLAW_CTX", "4096"))         # context window
N_THREADS    = int(os.getenv("JESTERCLAW_THREADS", "4"))         # CPU threads (for prefill)

# ── Singleton ──────────────────────────────────────────────────────────────────
_llm: Llama | None = None


def load_model() -> Llama:
    """
    Load the GGUF model into GPU memory.
    Called once at server startup. Subsequent calls are no-ops.
    """
    global _llm

    if _llm is not None:
        return _llm

    # Prefer local file; fall back to HF hub download
    if _GGUF_PATH.exists():
        model_path = str(_GGUF_PATH)
        logger.info("Loading GGUF from local file: %s", model_path)
    else:
        logger.info("Local GGUF not found — downloading from HuggingFace: %s / %s", _HF_REPO, _HF_FILE)
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_HF_FILE,
            local_dir=str(_MODEL_DIR),
        )
        logger.info("Downloaded to: %s", model_path)

    logger.info(
        "Initialising Llama: ctx=%d  gpu_layers=%s  threads=%d",
        N_CTX, N_GPU_LAYERS, N_THREADS,
    )

    _llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,   # -1 offloads everything to GPU
        n_threads=N_THREADS,
        verbose=False,               # set True to see llama.cpp logs
        chat_format="chatml",        # Qwen uses ChatML format
    )

    logger.info("Qwen2.5-Omni-3B-Q4_K_M loaded and ready.")
    return _llm


def get_llm() -> Llama:
    if _llm is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _llm


def model_info() -> dict:
    """Return basic model metadata for the /health endpoint."""
    return {
        "repo":   _HF_REPO,
        "file":   _HF_FILE,
        "ctx":    N_CTX,
        "gpu_layers": N_GPU_LAYERS,
        "loaded": _llm is not None,
    }

