"""
JesterClaw — Agent WebSocket Route (Server Side — Linux)
Team Lapanic / EmolOrbit

Architecture:
  SERVER (Linux/GPU)  → llama-cpp-python GGUF inference, text-only I/O
  CLIENT (Windows)    → STT locally, sends transcribed text OR raw audio
                        executes all OS actions

CLIENT → SERVER:
  {"type": "text",         "data": "open notepad"}          ← text (inc. STT result)
  {"type": "image",        "data": "<base64 JPEG>"}         ← screenshot from client
  {"type": "action_result","action":"...","ok":true,"data":"..."}
  {"type": "confirm"}     ← user confirmed a HIGH-risk action
  {"type": "stop"}

SERVER → CLIENT:
  {"type": "token",        "data": "Hello"}                 ← streamed token (fast)
  {"type": "text",         "data": "Full reply"}            ← complete response
  {"type": "action",       "data": {"action":"...","params":{...}}}  ← Windows DO THIS
  {"type": "request_screenshot"}                            ← ask client for screen
  {"type": "status",       "data": "thinking|done|error|confirm_required:action"}
  {"type": "error",        "data": "message"}
"""

import json
import logging
import asyncio
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from modules.inference_engine import stream_inference, extract_actions
from modules.session_manager import SessionManager
from modules.audio_processor import cleanup_temp
from Safety_Check.safety_filter import check_action_safety
from Safety_Check.command_validator import validate_command, requires_confirmation
from Database.session_db import register_session, close_session, log_action

logger = logging.getLogger("jesterclaw.agent")

# Actions that benefit from a fresh screenshot before execution
VISION_ACTIONS = {"click", "double_click", "right_click", "move_mouse", "browser_click"}


async def _send(ws: WebSocket, type_: str, data=None):
    try:
        payload = {"type": type_}
        if data is not None:
            payload["data"] = data
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass


async def agent_websocket_handler(websocket: WebSocket, session_manager: SessionManager):
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"

    session = await session_manager.create_session(websocket, client_ip)
    register_session(session.session_id, client_ip)
    await _send(websocket, "status", f"connected:{session.session_id}")
    logger.info("Client connected: %s (%s)", session.session_id, client_ip)

    tmp_image_path: Optional[str] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "text")

            # ── Control messages ───────────────────────────────────────────────
            if msg_type == "stop":
                await _send(websocket, "status", "stopped")
                continue

            if msg_type == "action_result":
                ok     = msg.get("ok", True)
                action = msg.get("action", "")
                detail = msg.get("data", "")
                log_action(session.session_id, client_ip, action,
                           "CLIENT_REPORT", f"{'ok' if ok else 'fail'}: {detail}")
                # Feed result back as context for next inference
                session.add_assistant_turn(
                    f"[Action '{action}' result: {'success' if ok else 'failed'} — {detail}]"
                )
                continue

            # ── Input parsing ──────────────────────────────────────────────────
            user_text = None
            image_b64 = None

            if msg_type == "text":
                user_text = str(msg.get("data", "")).strip()
                if not user_text:
                    continue

            elif msg_type == "image":
                # Client sent a screenshot (on request or manually)
                image_b64 = msg.get("data", "")
                user_text = msg.get("text", "Analyze this screen and decide what action to take next.")

            else:
                await _send(websocket, "error", f"Unknown message type: {msg_type}")
                continue

            # ── Inference ──────────────────────────────────────────────────────
            await _send(websocket, "status", "thinking")

            full_text = ""
            async for token in stream_inference(
                session.history,
                user_text=user_text,
                image_b64=image_b64,
            ):
                if token.startswith("__DONE__:"):
                    full_text = token[len("__DONE__:"):]
                    break
                await _send(websocket, "token", token)  # real-time token stream

            # ── Parse response ─────────────────────────────────────────────────
            clean_text, actions = extract_actions(full_text)

            if clean_text:
                await _send(websocket, "text", clean_text)

            # Update history
            session.add_user_turn([{"type": "text", "text": user_text}])
            session.add_assistant_turn(full_text)

            # ── Dispatch actions to Windows client ─────────────────────────────
            for action_dict in actions:
                action = action_dict.get("action", "")
                params = action_dict.get("params", {})

                # Safety gate (server-side, before sending to client)
                safe, reason = check_action_safety(action, params)
                if not safe:
                    logger.warning("BLOCKED: %s — %s", action, reason)
                    log_action(session.session_id, client_ip, action, "BLOCKED", reason)
                    await _send(websocket, "status", f"blocked:{action}:{reason}")
                    continue

                risk = validate_command(action, params)
                log_action(session.session_id, client_ip, action, risk.name, "dispatched")

                # For visual actions, request a fresh screenshot first
                if action in VISION_ACTIONS and image_b64 is None:
                    await _send(websocket, "request_screenshot")
                    try:
                        scr_raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                        scr_msg = json.loads(scr_raw)
                        if scr_msg.get("type") == "image":
                            new_img_b64 = scr_msg["data"]
                            # Re-infer with screenshot for accurate coordinates
                            await _send(websocket, "status", "thinking")
                            full2 = ""
                            async for t in stream_inference(
                                session.history,
                                user_text=f"Screen captured. Now precisely execute: {action} with context: {params}",
                                image_b64=new_img_b64,
                            ):
                                if t.startswith("__DONE__:"):
                                    full2 = t[len("__DONE__:"):]
                                    break
                            _, actions2 = extract_actions(full2)
                            if actions2:
                                action_dict = actions2[0]
                                action = action_dict.get("action", action)
                                params = action_dict.get("params", params)
                    except asyncio.TimeoutError:
                        logger.warning("Screenshot timeout for action: %s", action)

                # HIGH-risk: require explicit user confirmation from client
                if requires_confirmation(risk):
                    await _send(websocket, "status", f"confirm_required:{action}")
                    try:
                        conf_raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        conf_msg = json.loads(conf_raw)
                        if conf_msg.get("type") != "confirm":
                            await _send(websocket, "status", f"cancelled:{action}")
                            log_action(session.session_id, client_ip, action, risk.name, "cancelled")
                            continue
                    except asyncio.TimeoutError:
                        await _send(websocket, "status", f"timeout:{action}")
                        continue

                # ✅ Send action to Windows client for local execution
                await _send(websocket, "action", action_dict)
                logger.info("Action → client: %s %s", action, params)

            await _send(websocket, "status", "done")

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", session.session_id)
    except Exception as e:
        logger.exception("Agent error [%s]: %s", session.session_id, e)
        await _send(websocket, "error", str(e))
    finally:
        cleanup_temp(tmp_image_path)
        close_session(session.session_id)
        await session_manager.remove_session(session.session_id)

