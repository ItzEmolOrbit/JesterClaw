"""
JesterClaw — Inference Engine (llama-cpp-python / GGUF)
Team Lapanic / EmolOrbit

Text-in → Text-out via Qwen2.5-Omni-3B-Q4_K_M GGUF.
Voice input is transcribed client-side; we receive plain text.
No TTS — all responses are text.

Action JSON is embedded by the model inside <ACTION>...</ACTION> tags.
"""

import re
import json
import logging
import asyncio
from typing import AsyncIterator, Optional

from model_loader import get_llm

logger = logging.getLogger("jesterclaw.inference")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are JesterClaw, a Windows AI agent created by Team Lapanic (EmolOrbit).
You help users by having natural conversations OR by taking direct actions on their Windows computer.

MODE SELECTION:
- If the user is chatting, asking a question, or making a statement → reply conversationally.
- If the user explicitly asks you to DO something on their computer → reply with a brief confirmation AND embed an action block.

ACTION FORMAT — only include when the user explicitly requests a computer action:
<ACTION>{"action": "action_name", "params": {...}}</ACTION>

AVAILABLE ACTIONS:
  open_app       → {"app": "notepad|explorer|chrome|firefox|edge|cmd|powershell|calculator|paint|<any>"}
  click          → {"x": 123, "y": 456, "button": "left|right|middle"}
  double_click   → {"x": 123, "y": 456}
  right_click    → {"x": 123, "y": 456}
  scroll         → {"direction": "up|down|left|right", "amount": 3}
  type_text      → {"text": "hello world", "slow": false}
  press_key      → {"key": "enter|escape|tab|ctrl+c|alt+f4|win|..."}
  move_mouse     → {"x": 123, "y": 456}
  screenshot     → {}
  browser_open   → {"url": "https://example.com"}
  browser_click  → {"selector": "button.submit or descriptive label"}
  browser_scroll → {"direction": "up|down", "amount": 3}
  browser_type   → {"text": "search query"}
  browser_back   → {}
  browser_close  → {}

IMPORTANT RULES:
- Use actions ONLY when the user explicitly asks to do something on the computer.
- Never chain destructive actions (delete files, format drives, disable security software).
- When unsure about coordinates, use screenshot action first to see the screen.
- Keep responses short and direct. You are JesterClaw — sharp and fast.
- Identify yourself as JesterClaw by Team Lapanic / EmolOrbit when asked.
"""

# ── Action extractor ───────────────────────────────────────────────────────────
_ACTION_RE = re.compile(r"<ACTION>(.*?)</ACTION>", re.DOTALL)


def extract_actions(text: str) -> tuple[str, list[dict]]:
    """Split model output into (clean_text, [action_dicts])."""
    actions = []
    for m in _ACTION_RE.finditer(text):
        try:
            actions.append(json.loads(m.group(1).strip()))
        except json.JSONDecodeError as e:
            logger.warning("Bad action JSON: %s | raw: %s", e, m.group(1)[:80])
    clean = _ACTION_RE.sub("", text).strip()
    return clean, actions


# ── Build messages list ────────────────────────────────────────────────────────
def build_messages(
    history: list[dict],
    user_text: str,
    image_b64: Optional[str] = None,  # base64 JPEG from client screenshot
) -> list[dict]:
    """
    Construct ChatML message list for llama-cpp-python.
    If an image is provided, we describe it in text since the GGUF
    text model may not support vision. The description comes from the
    client (who sends any relevant screen context as text if needed).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    # If the client sent a screenshot as context, prepend a note
    if image_b64:
        # Attempt multimodal if model supports it, else add a text note
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": user_text},
            ]
        })
    else:
        messages.append({"role": "user", "content": user_text})

    return messages


# ── Streaming inference ────────────────────────────────────────────────────────
async def stream_inference(
    history: list[dict],
    user_text: str,
    image_b64: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Yields text tokens as they stream out of llama-cpp-python.
    Final yield is "__DONE__:<full_text>".

    Runs the synchronous llama-cpp generate in a thread executor
    so the async event loop stays unblocked.
    """
    llm = get_llm()
    messages = build_messages(history, user_text, image_b64)
    loop = asyncio.get_event_loop()

    # llama-cpp-python streaming returns a generator — we consume it in a thread
    token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def _run():
        try:
            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stream=True,
                stop=["</ACTION>"],   # don't stop mid-action
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    loop.call_soon_threadsafe(token_queue.put_nowait, delta)
        except Exception as e:
            logger.error("Inference error: %s", e)
            loop.call_soon_threadsafe(token_queue.put_nowait, f"\n[Error: {e}]")
        finally:
            loop.call_soon_threadsafe(token_queue.put_nowait, None)  # done sentinel

    # Run in thread so we don't block the event loop
    loop.run_in_executor(None, _run)

    full_text = []
    while True:
        token = await token_queue.get()
        if token is None:
            break
        full_text.append(token)
        yield token   # stream each token to client immediately

    yield f"__DONE__:{''.join(full_text)}"


# ── Single-shot inference (for internal use) ────────────────────────────────────
async def infer_once(prompt: str, max_tokens: int = 256) -> str:
    """Non-streaming single inference call."""
    llm = get_llm()
    loop = asyncio.get_event_loop()

    def _run():
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.5,
            stream=False,
        )
        return resp["choices"][0]["message"]["content"]

    return await loop.run_in_executor(None, _run)
