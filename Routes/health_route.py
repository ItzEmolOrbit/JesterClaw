"""
JesterClaw — Health Route
Team Lapanic / EmolOrbit
"""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from model_loader import vram_stats

logger = logging.getLogger("jesterclaw.health")
health_router = APIRouter()


@health_router.get("/health")
async def health(session_manager=None):
    return JSONResponse({
        "status":  "ok",
        "name":    "JesterClaw",
        "version": "1.0.0",
        "model":   "Qwen/Qwen2.5-Omni-3B",
        "by":      "Team Lapanic / EmolOrbit",
        "vram":    vram_stats(),
    })
