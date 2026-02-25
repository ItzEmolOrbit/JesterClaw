"""
JesterClaw — Model Loader
Team Lapanic / EmolOrbit

Singleton loader for Qwen2.5-Omni-3B.
Import get_model() and get_processor() anywhere in the server.
"""

import os
import logging
import torch
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

logger = logging.getLogger("jesterclaw.model")

# ── Path resolution ────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
MODEL_LOCAL_PATH = _BASE_DIR / "Model Files" / "Qwen2.5-Omni-3B"
MODEL_HF_ID      = "Qwen/Qwen2.5-Omni-3B"

# Use local path if downloaded, otherwise fall back to HF hub streaming
MODEL_PATH = str(MODEL_LOCAL_PATH) if MODEL_LOCAL_PATH.exists() else MODEL_HF_ID

# ── Singletons ─────────────────────────────────────────────────────────────────
_model: Qwen2_5OmniForConditionalGeneration | None = None
_processor: Qwen2_5OmniProcessor | None = None


def load_model() -> tuple[Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor]:
    """
    Load Qwen2.5-Omni-3B into GPU memory.
    Called once at server startup — subsequent calls are no-ops.
    """
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    logger.info("Loading Qwen2.5-Omni-3B from: %s", MODEL_PATH)

    # Try flash_attention_2 first (requires flash-attn package + Ampere+ GPU)
    try:
        _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        logger.info("Loaded with flash_attention_2 ✓")
    except Exception as flash_err:
        logger.warning("flash_attention_2 unavailable (%s). Falling back to eager.", flash_err)
        _model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info("Loaded with eager attention ✓")

    _processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

    # Log VRAM usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used  = torch.cuda.memory_allocated(i) / 1024 ** 3
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            logger.info("GPU %d: %.2f / %.2f GB VRAM used", i, used, total)

    logger.info("Qwen2.5-Omni-3B ready.")
    return _model, _processor


def get_model() -> Qwen2_5OmniForConditionalGeneration:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


def get_processor() -> Qwen2_5OmniProcessor:
    if _processor is None:
        raise RuntimeError("Processor not loaded. Call load_model() first.")
    return _processor


def vram_stats() -> dict:
    """Return VRAM usage per GPU as a dict."""
    stats = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used  = torch.cuda.memory_allocated(i) / 1024 ** 3
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            stats[f"gpu{i}"] = {"used_gb": round(used, 2), "total_gb": round(total, 2)}
    return stats
