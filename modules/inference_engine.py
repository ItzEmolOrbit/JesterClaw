"""
JesterClaw — Inference Engine
Team Lapanic / EmolOrbit

Handles all Qwen2.5-Omni-3B inference:
  - Text-only conversations (default, fast)
  - Audio input (voice command)
  - Image input (vision on demand)
  - Structured action parsing from model output
"""

import io
import re
import json
import logging
import asyncio
import numpy as np
import soundfile as sf
from typing import Optional, AsyncIterator
from threading import Thread

import torch
from transformers import TextIteratorStreamer
from qwen_omni_utils import process_mm_info

from model_loader import get_model, get_processor

logger = logging.getLogger("jesterclaw.inference")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are JesterClaw, a Windows AI agent created by Team Lapanic (EmolOrbit).
You help users accomplish tasks on their Windows computer through conversation or direct action.

MODES:
- If the user is just chatting or asking a question → respond conversationally in plain text.
- If the user asks you to DO something on their computer (open app, click, scroll, type, take screenshot, control browser, etc.) → respond with a JSON action block AND a brief spoken/text confirmation.

ACTION FORMAT (only when needed):
Wrap computer actions in <ACTION>...</ACTION> tags like this:
<ACTION>{"action": "open_app", "params": {"app": "notepad"}}</ACTION>

Available actions and params:
  open_app        → {"app": "notepad|explorer|chrome|firefox|edge|any_app_name"}
  click           → {"x": int, "y": int, "button": "left|right|middle"}
  double_click    → {"x": int, "y": int}
  right_click     → {"x": int, "y": int}
  scroll          → {"direction": "up|down|left|right", "amount": int}
  type_text       → {"text": "...", "slow": false}
  press_key       → {"key": "enter|escape|tab|win|ctrl+c|alt+f4|..."}
  move_mouse      → {"x": int, "y": int}
  screenshot      → {}  (capture screen to see current state)
  browser_open    → {"url": "https://..."}
  browser_click   → {"selector": "CSS selector or description"}
  browser_scroll  → {"direction": "up|down", "amount": int}
  browser_type    → {"text": "..."}
  browser_back    → {}
  browser_close   → {}
  speak           → {"text": "..."} (say something aloud)

SAFETY: Never perform destructive actions (format drives, delete system files, disable security) without explicit, confirmed user instruction. When unsure, ask first.

Keep responses short and natural. You are JesterClaw — sharp, quick, reliable.
"""

# ── Action extractor ───────────────────────────────────────────────────────────
ACTION_PATTERN = re.compile(r"<ACTION>(.*?)</ACTION>", re.DOTALL)


def extract_actions(text: str) -> tuple[str, list[dict]]:
    """
    Split model output into (clean_text, list_of_action_dicts).
    """
    actions = []
    for match in ACTION_PATTERN.finditer(text):
        try:
            actions.append(json.loads(match.group(1).strip()))
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse action JSON: %s | raw: %s", e, match.group(1))

    clean_text = ACTION_PATTERN.sub("", text).strip()
    return clean_text, actions


# ── Build conversation dict ────────────────────────────────────────────────────
def build_conversation(
    history: list[dict],
    user_text: Optional[str] = None,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> list[dict]:
    """
    Construct the messages list for the model, including history.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }
    ]

    # Replay history (already formatted)
    messages.extend(history)

    # Build user turn
    user_content = []
    if audio_path:
        user_content.append({"type": "audio", "audio": audio_path})
    if image_path:
        user_content.append({"type": "image", "image": image_path})
    if user_text:
        user_content.append({"type": "text", "text": user_text})

    if user_content:
        messages.append({"role": "user", "content": user_content})

    return messages


# ── Streaming text inference ───────────────────────────────────────────────────
async def stream_inference(
    history: list[dict],
    user_text: Optional[str] = None,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Yields text tokens as they are generated (streaming).
    Last yield is a special sentinel: "__DONE__:{full_text}"
    """
    model = get_model()
    processor = get_processor()
    loop = asyncio.get_event_loop()

    conversation = build_conversation(history, user_text, audio_path, image_path)

    # Prepare inputs in executor to avoid blocking the event loop
    def _prepare():
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text_prompt,
            audio=audios if audios else None,
            images=images if images else None,
            videos=None,
            return_tensors="pt",
            padding=True,
        )
        return inputs.to(model.device).to(model.dtype)

    inputs = await loop.run_in_executor(None, _prepare)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_audio=False,  # text-only streaming; TTS handled separately
    )

    # Run generation in a background thread
    thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    full_text = []
    queue: asyncio.Queue[str] = asyncio.Queue()

    def _enqueue():
        for token in streamer:
            loop.call_soon_threadsafe(queue.put_nowait, token)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    enqueue_thread = Thread(target=_enqueue, daemon=True)
    enqueue_thread.start()

    while True:
        token = await queue.get()
        if token is None:
            break
        full_text.append(token)
        yield token

    thread.join()
    yield f"__DONE__:{''.join(full_text)}"


# ── TTS: generate audio from text ─────────────────────────────────────────────
async def text_to_speech_bytes(text: str) -> Optional[bytes]:
    """
    Use Qwen2.5-Omni's built-in TTS to convert text → WAV bytes.
    Returns None if TTS fails.
    """
    model = get_model()
    processor = get_processor()
    loop = asyncio.get_event_loop()

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are JesterClaw, a Windows AI assistant. Speak the following text naturally."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Say: {text}"}],
        },
    ]

    def _run_tts():
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=text_prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.no_grad():
            text_ids, audio = model.generate(**inputs, max_new_tokens=512, return_audio=True)
        buf = io.BytesIO()
        sf.write(buf, audio.reshape(-1).cpu().float().numpy(), samplerate=24000, format="WAV")
        return buf.getvalue()

    try:
        wav_bytes = await loop.run_in_executor(None, _run_tts)
        return wav_bytes
    except Exception as e:
        logger.error("TTS generation failed: %s", e)
        return None
