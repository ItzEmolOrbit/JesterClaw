"""
JesterClaw — Audio Processor
Team Lapanic / EmolOrbit

Handles audio I/O:
  - PCM/WAV bytes → temp file for model ingestion
  - WAV bytes → base64 for WebSocket transmission
"""

import io
import os
import uuid
import base64
import logging
import tempfile
import numpy as np
import soundfile as sf

logger = logging.getLogger("jesterclaw.audio")

SAMPLE_RATE = 16000   # model expects 16kHz for audio input
TTS_RATE    = 24000   # model outputs 24kHz TTS


def pcm_bytes_to_wav_file(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Save raw PCM bytes (16-bit LE mono) to a temp WAV file.
    Returns file path. Caller is responsible for cleanup.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(tmp.name, audio_np, samplerate=sample_rate)
    tmp.close()
    return tmp.name


def wav_bytes_to_temp_file(wav_bytes: bytes) -> str:
    """
    Save WAV bytes to a temp file. Returns file path.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_bytes)
    tmp.close()
    return tmp.name


def wav_bytes_to_base64(wav_bytes: bytes) -> str:
    """Encode WAV bytes to base64 string for WebSocket JSON transmission."""
    return base64.b64encode(wav_bytes).decode("utf-8")


def base64_to_wav_bytes(b64: str) -> bytes:
    """Decode base64 string back to raw WAV bytes."""
    return base64.b64decode(b64)


def cleanup_temp_file(path: str):
    """Delete temp audio file safely."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning("Could not delete temp file %s: %s", path, e)
