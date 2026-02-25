"""
JesterClaw — Audio Processor (Server Side)
Team Lapanic / EmolOrbit

The server receives audio from the Windows client in two ways:
  1. Client does STT locally → sends plain text (preferred, lowest latency)
  2. Client sends raw WAV bytes → server decodes and passes to model if it supports audio

Since we use the GGUF text model, voice is handled client-side.
This module just provides byte-handling utilities for audio received over WebSocket.
"""

import io
import base64
import logging
import tempfile
import os

logger = logging.getLogger("jesterclaw.audio")


def base64_to_bytes(b64: str) -> bytes:
    """Decode base64 string to raw bytes."""
    return base64.b64decode(b64)


def bytes_to_base64(data: bytes) -> str:
    """Encode raw bytes to base64 string for WebSocket JSON."""
    return base64.b64encode(data).decode("utf-8")


def save_to_temp(data: bytes, suffix: str = ".wav") -> str:
    """Save bytes to a temporary file. Returns path. Caller must delete."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def cleanup_temp(path: str):
    """Delete a temp file safely."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning("Could not delete temp file %s: %s", path, e)

