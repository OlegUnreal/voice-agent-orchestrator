from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..providers.base import ChatMessage


@dataclass
class Session:
    session_id: str
    messages: list[ChatMessage] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    """In-memory session store. Swap for Redis/Postgres without touching the agent."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        return self._sessions.setdefault(session_id, Session(session_id=session_id))

    def get_messages(self, session_id: str) -> list[ChatMessage]:
        return list(self.get_or_create(session_id).messages)

    def append(self, session_id: str, *messages: ChatMessage) -> None:
        self.get_or_create(session_id).messages.extend(messages)
