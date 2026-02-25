"""
JesterClaw — Command Validator
Team Lapanic / EmolOrbit

Classifies every action into LOW / MEDIUM / HIGH risk.
HIGH risk actions require an explicit user confirmation token.
"""

import logging
from enum import Enum

logger = logging.getLogger("jesterclaw.validator")


class RiskLevel(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


# Risk classification by action type
RISK_MAP = {
    "screenshot":      RiskLevel.LOW,
    "scroll":          RiskLevel.LOW,
    "move_mouse":      RiskLevel.LOW,
    "speak":           RiskLevel.LOW,
    "click":           RiskLevel.MEDIUM,
    "double_click":    RiskLevel.MEDIUM,
    "right_click":     RiskLevel.MEDIUM,
    "press_key":       RiskLevel.MEDIUM,
    "type_text":       RiskLevel.MEDIUM,
    "open_app":        RiskLevel.MEDIUM,
    "browser_open":    RiskLevel.MEDIUM,
    "browser_click":   RiskLevel.MEDIUM,
    "browser_scroll":  RiskLevel.LOW,
    "browser_type":    RiskLevel.MEDIUM,
    "browser_back":    RiskLevel.LOW,
    "browser_close":   RiskLevel.LOW,
}

# Specific param patterns that elevate risk
HIGH_RISK_APPS = {"cmd.exe", "powershell.exe", "regedit.exe", "taskkill", "wmic"}
HIGH_RISK_KEYS = {"delete", "f4", "win+r"}


def validate_command(action: str, params: dict) -> RiskLevel:
    """
    Determine the risk level of an action + params combo.
    Some actions are elevated to HIGH based on their params.
    """
    base_risk = RISK_MAP.get(action, RiskLevel.MEDIUM)

    # Elevate risk for dangerous apps
    if action == "open_app":
        app = params.get("app", "").lower()
        if any(h in app for h in HIGH_RISK_APPS):
            logger.warning("Elevated to HIGH risk: open_app → %s", app)
            return RiskLevel.HIGH

    # Elevate for dangerous key combos
    if action == "press_key":
        key = params.get("key", "").lower()
        if any(h in key for h in HIGH_RISK_KEYS):
            return RiskLevel.HIGH

    return base_risk


def requires_confirmation(risk: RiskLevel) -> bool:
    """HIGH risk actions must be confirmed by the user before executing."""
    return risk == RiskLevel.HIGH
