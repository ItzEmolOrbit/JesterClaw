"""
JesterClaw — Main Server Entry Point (Linux / Ubuntu GPU Server)
Team Lapanic / EmolOrbit

Architecture:
  SERVER (this, Linux) → runs Qwen2.5-Omni-3B, infers, dispatches action JSON
  CLIENT (Windows)     → executes actions (pyautogui, playwright, mss, etc.)

Endpoints:
  GET  /health  → system status + VRAM
  WS   /agent   → full-duplex agent loop (text / audio / vision / actions)
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
import uvicorn

from model_loader import load_model, vram_stats
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

# ── Config (override via environment variables) ────────────────────────────────
HOST          = os.getenv("JESTERCLAW_HOST", "0.0.0.0")
PORT          = int(os.getenv("JESTERCLAW_PORT", "3839"))
SECRET_TOKEN  = os.getenv("JESTERCLAW_TOKEN", "jesterclaw-secret-change-me")

# ── Session manager (shared across routes) ─────────────────────────────────────
session_manager = SessionManager()

# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print(r"""
     ___          _                  _____ _
    |_  |        | |                /  __ \ |
      | | ___  __| |_ __ ___  _ __  | /  \/ | __ ___      __
      | |/ _ \/ _` | '__/ _ \| '_ \ | |   | |/ _` \ \ /\ / /
  /\__/ /  __/ (_| | | | (_) | |_) || \__/\ | (_| |\ V  V /
  \____/ \___|\__,_|_|  \___/| .__/  \____/_|\__,_| \_/\_/
                              | |
                              |_|   Team Lapanic / EmolOrbit
    """)
    logger.info("JesterClaw server starting on %s:%d", HOST, PORT)

    # Safety pre-check
    if not is_server_safe_to_start():
        logger.critical("Safety pre-check failed. Aborting startup.")
        raise RuntimeError("Safety pre-check failed.")

    # Load model into GPU memory once
    logger.info("Loading Qwen2.5-Omni-3B into GPU memory (this may take ~30s)...")
    load_model()
    logger.info("Model ready. JesterClaw is live.")

    yield  # ← server is running

    # --- Shutdown ---
    logger.info("Shutting down JesterClaw. Goodbye.")
    await session_manager.close_all()


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="JesterClaw",
    description="Windows AI Agent — Team Lapanic / EmolOrbit",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

# ── Auth helper ────────────────────────────────────────────────────────────────
def verify_token(x_jesterclaw_token: str = Header(default="")):
    if x_jesterclaw_token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing JesterClaw token.")


# ── Include routers ────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(screenshot_router, dependencies=[Depends(verify_token)])


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/agent")
async def agent_endpoint(
    websocket: WebSocket,
    token: str = "",          # passed as query param: ws://IP:3839/agent?token=...
):
    # Token auth via query param for WebSocket (headers are tricky across clients)
    if token != SECRET_TOKEN:
        await websocket.close(code=4003, reason="Invalid token")
        return

    await agent_websocket_handler(websocket, session_manager)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting JesterClaw on %s:%d", HOST, PORT)
    uvicorn.run(
        "Main:app",
        host=HOST,
        port=PORT,
        ws_ping_interval=20,
        ws_ping_timeout=30,
        log_level="info",
        reload=False,
    )
