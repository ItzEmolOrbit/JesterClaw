"""
JesterClaw — Screenshot Route (Server Side)
Team Lapanic / EmolOrbit

NOTE: The server runs on Linux and cannot capture the Windows client's screen.
This endpoint is intentionally disabled server-side.
Screenshots are captured by the Windows CLIENT using mss, then sent to the server
via WebSocket as base64 JPEG when needed for visual context.

This file is kept as a placeholder. The client-side handles all screen capture.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

screenshot_router = APIRouter()


@screenshot_router.get("/screenshot")
async def get_screenshot():
    return JSONResponse(
        status_code=501,
        content={
            "error": "Screen capture runs on the Windows client, not the server.",
            "hint": "Send {type: 'image', data: '<base64 JPEG>'} over the WebSocket instead.",
        },
    )
