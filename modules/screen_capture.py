"""
JesterClaw — Screen Capture
Team Lapanic / EmolOrbit

Ultra-fast screen capture using mss (~5ms per shot).
Only called when the user explicitly asks to see the screen
or when the agent needs visual context for a task.
"""

import io
import base64
import logging
from PIL import Image
import mss
import mss.tools

logger = logging.getLogger("jesterclaw.screen")

# Default JPEG quality — lower = faster transfer, higher = more detail
JPEG_QUALITY = 75


def capture_screen_jpeg(
    monitor_index: int = 1,      # 1 = primary monitor
    quality: int = JPEG_QUALITY,
    scale: float = 1.0,          # 0.5 = half resolution for speed
) -> bytes:
    """
    Capture the full screen and return JPEG bytes.
    Fast enough for near-real-time visual context.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        screenshot = sct.grab(monitor)

    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    if scale != 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def capture_screen_base64(monitor_index: int = 1, quality: int = JPEG_QUALITY) -> str:
    """Return base64-encoded JPEG for embedding in WebSocket JSON."""
    return base64.b64encode(capture_screen_jpeg(monitor_index, quality)).decode("utf-8")


def capture_to_temp_file(monitor_index: int = 1) -> str:
    """
    Save screenshot to a temp file and return path.
    Used to pass screenshots into the model as image input.
    """
    import tempfile, os
    jpeg_bytes = capture_screen_jpeg(monitor_index)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(jpeg_bytes)
    tmp.close()
    return tmp.name
