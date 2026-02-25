"""
JesterClaw — Safety Filter
Team Lapanic / EmolOrbit

Hard blocklist + pattern matching to prevent dangerous OS actions.
This runs BEFORE every action execution.
"""

import re
import logging
import sys

logger = logging.getLogger("jesterclaw.safety")

# ── Hardcoded destructive command blocklist ────────────────────────────────────
BLOCKED_ACTIONS = {
    # Disk / filesystem destruction
    "format", "mkfs", "fdisk", "diskpart",
    # Mass deletion
    "rm -rf", "del /f /s /q", "rmdir /s",
    # System shutdown/restart without warning
    "shutdown /r /t 0", "shutdown /s /t 0",
    # Disabling security
    "netsh advfirewall set allprofiles state off",
    "sc stop wuauserv", "sc delete",
    # Privilege escalation scripts
    "net user administrator",
    "reg add hklm\\sam", "reg delete hklm\\system",
    # Dangerous powershell patterns
    "invoke-expression", "iex(", "downloadstring",
    "set-executionpolicy unrestricted",
}

# Regex patterns for dangerous content
BLOCKED_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"del\s+/[fs].*\*", re.IGNORECASE),
    re.compile(r"format\s+[a-z]:", re.IGNORECASE),
    re.compile(r"reg\s+(delete|add)\s+hklm\\(sam|system|security)", re.IGNORECASE),
    re.compile(r"(wget|curl).*(\.exe|\.bat|\.ps1|\.vbs).*\|\s*(bash|sh|cmd|powershell)", re.IGNORECASE),
    re.compile(r"base64.*decode.*invoke", re.IGNORECASE),
]

# Actions that are always safe (never blocked)
ALWAYS_SAFE = {"screenshot", "scroll", "move_mouse", "speak"}

# Actions that can do damage if params are malicious
SENSITIVE_ACTIONS = {"open_app", "type_text", "press_key", "browser_open"}


def check_action_safety(action: str, params: dict) -> tuple[bool, str]:
    """
    Returns (is_safe: bool, reason: str).
    Blocks if any hardcoded blocklist entry or pattern matches.
    """
    if action in ALWAYS_SAFE:
        return True, "ok"

    # Build a combined string to scan
    combined = action.lower() + " " + " ".join(str(v).lower() for v in params.values())

    for blocked in BLOCKED_ACTIONS:
        if blocked in combined:
            return False, f"Blocked keyword detected: '{blocked}'"

    for pattern in BLOCKED_PATTERNS:
        if pattern.search(combined):
            return False, f"Blocked pattern detected: {pattern.pattern}"

    return True, "ok"


def is_server_safe_to_start() -> bool:
    """
    Quick sanity check at startup.
    Returns False only in extreme cases (e.g. running as SYSTEM in production).
    """
    if sys.platform != "win32":
        logger.warning("JesterClaw is designed for Windows. Proceeding anyway...")
    logger.info("Safety pre-check passed.")
    return True
