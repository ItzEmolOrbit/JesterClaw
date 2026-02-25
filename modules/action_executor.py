"""
JesterClaw — Action Executor
Team Lapanic / EmolOrbit

Executes Windows OS actions dispatched by the inference engine.
Actions are ONLY executed when the model decides the user explicitly
requested them — never on idle conversation.

Every action passes through the safety filter before execution.
"""

import time
import logging
import subprocess
import pyautogui
import pywinauto
from pywinauto import Desktop

from Safety_Check.safety_filter import check_action_safety
from Safety_Check.command_validator import validate_command, RiskLevel
from Database.session_db import log_action

logger = logging.getLogger("jesterclaw.actions")

# Prevent pyautogui from raising on edge movements
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05   # 50ms between actions for stability


def execute_action(action_dict: dict, session_id: str, client_ip: str) -> dict:
    """
    Entry point. Validates safety, checks risk level, executes action.
    Returns {"ok": bool, "result": str}.
    """
    action  = action_dict.get("action", "")
    params  = action_dict.get("params", {})

    # ── Safety check ──────────────────────────────────────────────────────────
    safe, reason = check_action_safety(action, params)
    if not safe:
        logger.warning("Action BLOCKED [safety]: %s — %s", action, reason)
        log_action(session_id, client_ip, action, "BLOCKED", reason)
        return {"ok": False, "result": f"Blocked by safety filter: {reason}"}

    risk = validate_command(action, params)
    log_action(session_id, client_ip, action, risk.name, "pending")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    try:
        result = _dispatch(action, params)
        log_action(session_id, client_ip, action, risk.name, "ok: " + result)
        return {"ok": True, "result": result}
    except Exception as e:
        err = str(e)
        logger.error("Action failed [%s]: %s", action, err)
        log_action(session_id, client_ip, action, risk.name, "error: " + err)
        return {"ok": False, "result": err}


def _dispatch(action: str, params: dict) -> str:
    handlers = {
        "open_app":     _open_app,
        "click":        _click,
        "double_click": _double_click,
        "right_click":  _right_click,
        "scroll":       _scroll,
        "type_text":    _type_text,
        "press_key":    _press_key,
        "move_mouse":   _move_mouse,
        "screenshot":   _screenshot_noop,   # screenshot handled by screen_capture module
    }
    handler = handlers.get(action)
    if handler is None:
        raise ValueError(f"Unknown action: {action}")
    return handler(params)


# ── Individual handlers ─────────────────────────────────────────────────────────

def _open_app(params: dict) -> str:
    app = params.get("app", "")
    if not app:
        raise ValueError("open_app requires 'app' param")

    # Common aliases
    aliases = {
        "file manager": "explorer.exe",
        "explorer":     "explorer.exe",
        "notepad":      "notepad.exe",
        "terminal":     "cmd.exe",
        "cmd":          "cmd.exe",
        "powershell":   "powershell.exe",
        "calculator":   "calc.exe",
        "paint":        "mspaint.exe",
        "task manager": "taskmgr.exe",
        "browser":      "start chrome",
        "chrome":       "start chrome",
        "firefox":      "start firefox",
        "edge":         "start msedge",
    }

    cmd = aliases.get(app.lower(), app)
    subprocess.Popen(cmd, shell=True)
    time.sleep(0.3)
    return f"Opened: {app}"


def _click(params: dict) -> str:
    x, y = int(params["x"]), int(params["y"])
    btn = params.get("button", "left")
    pyautogui.click(x, y, button=btn)
    return f"Clicked ({x}, {y}) with {btn}"


def _double_click(params: dict) -> str:
    x, y = int(params["x"]), int(params["y"])
    pyautogui.doubleClick(x, y)
    return f"Double-clicked ({x}, {y})"


def _right_click(params: dict) -> str:
    x, y = int(params["x"]), int(params["y"])
    pyautogui.rightClick(x, y)
    return f"Right-clicked ({x}, {y})"


def _scroll(params: dict) -> str:
    direction = params.get("direction", "down")
    amount    = int(params.get("amount", 3))
    clicks = amount if direction in ("up", "right") else -amount

    if direction in ("up", "down"):
        pyautogui.scroll(clicks)
    else:
        pyautogui.hscroll(clicks)
    return f"Scrolled {direction} by {amount}"


def _type_text(params: dict) -> str:
    text = params.get("text", "")
    slow = params.get("slow", False)
    interval = 0.05 if slow else 0.01
    pyautogui.typewrite(text, interval=interval)
    return f"Typed: {text[:40]}{'...' if len(text) > 40 else ''}"


def _press_key(params: dict) -> str:
    key = params.get("key", "")
    if not key:
        raise ValueError("press_key requires 'key' param")
    # Handle combos like ctrl+c
    if "+" in key:
        keys = [k.strip() for k in key.split("+")]
        pyautogui.hotkey(*keys)
    else:
        pyautogui.press(key)
    return f"Pressed key: {key}"


def _move_mouse(params: dict) -> str:
    x, y = int(params["x"]), int(params["y"])
    pyautogui.moveTo(x, y, duration=0.15)
    return f"Moved mouse to ({x}, {y})"


def _screenshot_noop(params: dict) -> str:
    # Actual screenshot is handled upstream by the agent route
    return "screenshot_requested"
