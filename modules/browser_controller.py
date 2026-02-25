"""
JesterClaw — Browser Controller
Team Lapanic / EmolOrbit

Playwright-based browser agent (visible Chromium).
Only launched when user explicitly asks to control the browser.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("jesterclaw.browser")

# Blocked domains for safety
BLOCKED_DOMAINS = [
    "localhost", "127.0.0.1", "0.0.0.0",
    "192.168.", "10.", "172.16.", "172.17.", "172.18.",  # private ranges
]

_browser = None
_page    = None
_playwright_instance = None


async def _get_page():
    """Lazy-init browser: only opens when first browser action is requested."""
    global _browser, _page, _playwright_instance
    if _page is not None and not _page.is_closed():
        return _page

    from playwright.async_api import async_playwright
    _playwright_instance = await async_playwright().start()
    _browser = await _playwright_instance.chromium.launch(
        headless=False,   # user sees the browser
        args=["--start-maximized"],
    )
    context = await _browser.new_context(no_viewport=True)
    _page = await context.new_page()
    logger.info("Browser (Chromium) launched.")
    return _page


def _is_safe_url(url: str) -> bool:
    for blocked in BLOCKED_DOMAINS:
        if blocked in url:
            return False
    return True


async def browser_open(url: str) -> str:
    if not _is_safe_url(url):
        return f"Blocked: URL contains a private/blocked domain."
    page = await _get_page()
    await page.goto(url, timeout=30000, wait_until="domcontentloaded")
    title = await page.title()
    logger.info("Navigated to: %s (%s)", url, title)
    return f"Opened: {url} — Page: {title}"


async def browser_click(selector: str) -> str:
    page = await _get_page()
    try:
        await page.click(selector, timeout=8000)
        return f"Clicked: {selector}"
    except Exception as e:
        # Fallback: try by text content
        try:
            await page.get_by_text(selector).first.click(timeout=5000)
            return f"Clicked by text: {selector}"
        except:
            raise e


async def browser_scroll(direction: str = "down", amount: int = 3) -> str:
    page = await _get_page()
    pixels = amount * 300
    delta = pixels if direction == "down" else -pixels
    await page.mouse.wheel(0, delta)
    return f"Scrolled {direction} by {amount} units"


async def browser_type(text: str) -> str:
    page = await _get_page()
    await page.keyboard.type(text, delay=20)
    return f"Typed: {text[:40]}"


async def browser_back() -> str:
    page = await _get_page()
    await page.go_back()
    return "Navigated back"


async def browser_get_text() -> str:
    """Extract visible text from current page (for model context)."""
    page = await _get_page()
    text = await page.evaluate("() => document.body.innerText")
    return text[:4000]  # limit to avoid token overload


async def browser_close() -> str:
    global _browser, _page, _playwright_instance
    if _browser:
        await _browser.close()
        _browser = None
        _page    = None
    if _playwright_instance:
        await _playwright_instance.stop()
        _playwright_instance = None
    logger.info("Browser closed.")
    return "Browser closed"


async def execute_browser_action(action: str, params: dict) -> str:
    """Unified dispatcher for browser actions."""
    dispatch = {
        "browser_open":   lambda: browser_open(params.get("url", "")),
        "browser_click":  lambda: browser_click(params.get("selector", "")),
        "browser_scroll": lambda: browser_scroll(params.get("direction", "down"), int(params.get("amount", 3))),
        "browser_type":   lambda: browser_type(params.get("text", "")),
        "browser_back":   browser_back,
        "browser_close":  browser_close,
    }
    handler = dispatch.get(action)
    if handler is None:
        raise ValueError(f"Unknown browser action: {action}")
    return await handler()
