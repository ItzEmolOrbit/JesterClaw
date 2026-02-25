"""
JesterClaw — Main Server Entry Point (Linux / Ubuntu GPU Server)
Team Lapanic / EmolOrbit

Architecture:
  SERVER (this, Linux) → runs Qwen2.5-Omni-3B Q4_K_M GGUF via llama.cpp
                         infers text, dispatches action JSON to client
  CLIENT (Windows)     → executes actions (pyautogui, playwright, mss, etc.)
                         handles voice input (STT) locally

Endpoints:
  GET /health  → server + model status
  WS  /agent   → full-duplex text agent loop
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
import uvicorn

from model_loader import load_model, model_info
from modules.session_manager import SessionManager
from Routes.agent_route import agent_websocket_handler
from Routes.health_route import health_router
from Routes.screenshot_route import screenshot_router
from Safety_Check.safety_filter import is_server_safe_to_start

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jesterclaw")

# ── Config ─────────────────────────────────────────────────────────────────────
HOST         = os.getenv("JESTERCLAW_HOST", "0.0.0.0")
PORT         = int(os.getenv("JESTERCLAW_PORT", "3839"))
SECRET_TOKEN = os.getenv("JESTERCLAW_TOKEN", "jesterclaw-secret-change-me")

# ── Shared session manager ─────────────────────────────────────────────────────
session_manager = SessionManager()

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(r"""
     ___          _                  _____ _
    |_  |        | |                /  __ \ |
      | | ___  __| |_ __ ___  _ __  | /  \/ | __ ___      __
      | |/ _ \/ _` | '__/ _ \| '_ \ | |   | |/ _` \ \ /\ / /
  /\__/ /  __/ (_| | | | (_) | |_) || \__/\ | (_| |\ V  V /
  \____/ \___|\__,_|_|  \___/| .__/  \____/_|\__,_| \_/\_/
                              | |
                              |_|   Team Lapanic / EmolOrbit
         Backend: Qwen2.5-Omni-3B Q4_K_M (llama.cpp GGUF)
    """)

    logger.info("JesterClaw server starting on %s:%d", HOST, PORT)

    if not is_server_safe_to_start():
        raise RuntimeError("Safety pre-check failed.")

    logger.info("Loading GGUF model via llama.cpp...")
    load_model()
    logger.info("Model ready. JesterClaw is live on port %d.", PORT)

    yield

    logger.info("Shutting down JesterClaw.")
    await session_manager.close_all()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="JesterClaw",
    description="Windows AI Agent — Team Lapanic / EmolOrbit",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(screenshot_router)


# ── WebSocket auth ─────────────────────────────────────────────────────────────
@app.websocket("/agent")
async def agent_endpoint(websocket: WebSocket, token: str = ""):
    if token != SECRET_TOKEN:
        await websocket.close(code=4003, reason="Invalid token")
        return
    await agent_websocket_handler(websocket, session_manager)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "Main:app",
        host=HOST,
        port=PORT,
        ws_ping_interval=20,
        ws_ping_timeout=30,
        log_level="info",
        reload=False,
    )
