"""
JesterClaw — Health Route
Team Lapanic / EmolOrbit
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from model_loader import model_info

health_router = APIRouter()


@health_router.get("/health")
async def health():
    return JSONResponse({
        "status":  "ok",
        "name":    "JesterClaw",
        "version": "1.0.0",
        "by":      "Team Lapanic / EmolOrbit",
        "model":   model_info(),
    })
