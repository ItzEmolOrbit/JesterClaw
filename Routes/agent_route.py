"""
JesterClaw — Agent WebSocket Route (Server Side — Linux)
Team Lapanic / EmolOrbit

Architecture:
  SERVER (Linux/GPU) → runs Qwen2.5-Omni-3B, infers intent, sends action JSON
  CLIENT (Windows)   → executes actions (pyautogui, playwright, mss, etc.)

Full-duplex WebSocket on ws://IP:3839/agent?token=...

CLIENT → SERVER messages (JSON):
  {"type": "text",       "data": "open notepad"}
  {"type": "audio",      "data": "<base64 WAV>"}          ← voice input
  {"type": "image",      "data": "<base64 JPEG>"}          ← screenshot from client
  {"type": "action_result", "action": "open_app", "ok": true, "data": "Opened notepad"}
  {"type": "stop"}

SERVER → CLIENT messages (JSON):
  {"type": "token",      "data": "Hello"}                  ← streamed token
  {"type": "text",       "data": "Full reply text"}         ← final clean text
  {"type": "audio",      "data": "<base64 WAV>"}           ← TTS voice reply
  {"type": "action",     "data": {"action": "...", "params": {...}}}   ← DO THIS
  {"type": "request_screenshot"}                            ← ask client to send screen
  {"type": "status",     "data": "thinking|done|error|confirm_required:action_name"}
  {"type": "error",      "data": "message"}
"""

import json
import logging
import asyncio
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from modules.inference_engine import stream_inference, extract_actions, text_to_speech_bytes
from modules.session_manager import SessionManager
from modules.audio_processor import (
    base64_to_wav_bytes, wav_bytes_to_base64,
    wav_bytes_to_temp_file, cleanup_temp_file,
)
from Safety_Check.safety_filter import check_action_safety
from Safety_Check.command_validator import validate_command, requires_confirmation
from Database.session_db import register_session, close_session, log_action

logger = logging.getLogger("jesterclaw.agent")

import os
ENABLE_TTS = os.getenv("JESTERCLAW_TTS", "1") == "1"

# Actions where the server should request a fresh screenshot from the client
# before sending the action, so the model has visual context
VISION_ACTIONS = {"click", "double_click", "right_click", "move_mouse", "browser_click"}


async def _send(ws: WebSocket, type_: str, data=None):
    """Send a JSON message to the client."""
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

    tmp_audio_path: Optional[str] = None
    tmp_image_path: Optional[str] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "text")

            # ── Stop signal ────────────────────────────────────────────────────
            if msg_type == "stop":
                await _send(websocket, "status", "stopped")
                continue

            # ── Action result feedback from client ─────────────────────────────
            if msg_type == "action_result":
                # Client tells us whether an action succeeded
                # Add to session context so model knows what happened
                ok     = msg.get("ok", True)
                action = msg.get("action", "")
                detail = msg.get("data", "")
                log_action(session.session_id, client_ip, action,
                           "CLIENT_REPORT", f"{'ok' if ok else 'fail'}: {detail}")
                continue

            # ── Determine multimodal inputs ────────────────────────────────────
            user_text  = None
            audio_path = None
            image_path = None

            if msg_type == "text":
                user_text = str(msg.get("data", "")).strip()
                if not user_text:
                    continue

            elif msg_type == "audio":
                wav_bytes = base64_to_wav_bytes(msg["data"])
                tmp_audio_path = wav_bytes_to_temp_file(wav_bytes)
                audio_path = tmp_audio_path

            elif msg_type == "image":
                # Screenshot sent by client (either on-demand or in response to request_screenshot)
                import base64, tempfile
                img_bytes = base64.b64decode(msg["data"])
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                tmp_image_path = tmp.name
                image_path = tmp_image_path
                # If no text alongside this image, add a descriptive prompt
                user_text = msg.get("text", "What do you see on the screen? Decide the best next action.")

            else:
                await _send(websocket, "error", f"Unknown message type: {msg_type}")
                continue

            # ── Notify client: thinking ────────────────────────────────────────
            await _send(websocket, "status", "thinking")

            # ── Stream inference from model ────────────────────────────────────
            full_text = ""
            async for token in stream_inference(
                session.history,
                user_text=user_text,
                audio_path=audio_path,
                image_path=image_path,
            ):
                if token.startswith("__DONE__:"):
                    full_text = token[len("__DONE__:"):]
                    break
                await _send(websocket, "token", token)  # stream each token instantly

            # ── Parse model output ─────────────────────────────────────────────
            clean_text, actions = extract_actions(full_text)

            # Send final clean response text
            if clean_text:
                await _send(websocket, "text", clean_text)

            # ── TTS for pure conversation (no action) ──────────────────────────
            if ENABLE_TTS and clean_text and not actions:
                wav_bytes = await text_to_speech_bytes(clean_text[:300])
                if wav_bytes:
                    await _send(websocket, "audio", wav_bytes_to_base64(wav_bytes))

            # ── Update conversation history ────────────────────────────────────
            user_content = []
            if audio_path:
                user_content.append({"type": "audio", "audio": audio_path})
            if image_path:
                user_content.append({"type": "image", "image": image_path})
            if user_text:
                user_content.append({"type": "text", "text": user_text})
            session.add_user_turn(user_content)
            session.add_assistant_turn(full_text)

            # ── Dispatch actions to client ─────────────────────────────────────
            for action_dict in actions:
                action = action_dict.get("action", "")
                params = action_dict.get("params", {})

                # Server-side safety check (before sending to client)
                safe, reason = check_action_safety(action, params)
                if not safe:
                    logger.warning("Action BLOCKED [safety]: %s — %s", action, reason)
                    log_action(session.session_id, client_ip, action, "BLOCKED", reason)
                    await _send(websocket, "status", f"blocked:{action}:{reason}")
                    continue

                risk = validate_command(action, params)
                log_action(session.session_id, client_ip, action, risk.name, "dispatched_to_client")

                # For vision-based actions, ask client to send a screenshot first
                if action in VISION_ACTIONS and image_path is None:
                    logger.info("Requesting screenshot from client for visual action: %s", action)
                    await _send(websocket, "request_screenshot")
                    # Wait for client to send image back (with timeout)
                    try:
                        screen_raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                        screen_msg = json.loads(screen_raw)
                        if screen_msg.get("type") == "image":
                            import base64, tempfile
                            img_bytes = base64.b64decode(screen_msg["data"])
                            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                            tmp.write(img_bytes)
                            tmp.close()
                            tmp_image_path = tmp.name
                            # Re-run inference with screenshot for accurate coordinates
                            await _send(websocket, "status", "thinking")
                            full_text2 = ""
                            async for t in stream_inference(
                                session.history,
                                user_text=f"Here is the current screen. Execute: {action} {params}",
                                image_path=tmp.name,
                            ):
                                if t.startswith("__DONE__:"):
                                    full_text2 = t[len("__DONE__:"):]
                                    break
                            _, actions2 = extract_actions(full_text2)
                            if actions2:
                                action_dict = actions2[0]
                                action = action_dict.get("action", action)
                                params = action_dict.get("params", params)
                            cleanup_temp_file(tmp.name)
                    except asyncio.TimeoutError:
                        logger.warning("Screenshot request timed out for action %s", action)

                # High-risk actions require explicit confirmation from client
                if requires_confirmation(risk):
                    await _send(websocket, "status", f"confirm_required:{action}")
                    try:
                        confirm_raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        confirm_msg = json.loads(confirm_raw)
                        if confirm_msg.get("type") != "confirm":
                            await _send(websocket, "status", f"cancelled:{action}")
                            log_action(session.session_id, client_ip, action, risk.name, "cancelled")
                            continue
                    except asyncio.TimeoutError:
                        await _send(websocket, "status", f"timeout:{action}")
                        log_action(session.session_id, client_ip, action, risk.name, "timeout")
                        continue

                # ✅ Send action to Windows client — client does the actual execution
                await _send(websocket, "action", action_dict)
                logger.info("Action dispatched to client: %s %s", action, params)

            await _send(websocket, "status", "done")

            # ── Cleanup temp files ─────────────────────────────────────────────
            cleanup_temp_file(tmp_audio_path)
            cleanup_temp_file(tmp_image_path)
            tmp_audio_path = None
            tmp_image_path = None

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", session.session_id)
    except Exception as e:
        logger.exception("Agent error [%s]: %s", session.session_id, e)
        await _send(websocket, "error", str(e))
    finally:
        cleanup_temp_file(tmp_audio_path)
        cleanup_temp_file(tmp_image_path)
        close_session(session.session_id)
        await session_manager.remove_session(session.session_id)
