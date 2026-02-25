"""
JesterClaw — Session Manager
Team Lapanic / EmolOrbit

Tracks connected clients and their conversation history.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional
from fastapi import WebSocket

logger = logging.getLogger("jesterclaw.sessions")

MAX_HISTORY_TURNS = 20   # keep last 20 user+assistant pairs


@dataclass
class SessionState:
    session_id: str
    client_ip: str
    websocket: WebSocket
    history: list[dict] = field(default_factory=list)
    active: bool = True

    def add_user_turn(self, content: list[dict]):
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_turn(self, text: str):
        self.history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        })
        self._trim()

    def _trim(self):
        """Keep at most MAX_HISTORY_TURNS pairs (2 messages per turn)."""
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, websocket: WebSocket, client_ip: str) -> SessionState:
        session_id = str(uuid.uuid4())
        state = SessionState(
            session_id=session_id,
            client_ip=client_ip,
            websocket=websocket,
        )
        async with self._lock:
            self._sessions[session_id] = state
        logger.info("Session created: %s from %s", session_id, client_ip)
        return state

    async def remove_session(self, session_id: str):
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Session removed: %s", session_id)

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def session_count(self) -> int:
        return len(self._sessions)

    async def close_all(self):
        async with self._lock:
            for state in self._sessions.values():
                try:
                    await state.websocket.close()
                except Exception:
                    pass
            self._sessions.clear()
        logger.info("All sessions closed.")
