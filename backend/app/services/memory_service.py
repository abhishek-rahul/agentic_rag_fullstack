from collections import defaultdict, deque
from typing import Deque, Dict, List

from app.core.config import get_settings
from app.infrastructure.sqlite_store import SQLiteMemoryStore


class MemoryService:
    """Combines simple in-process short-term memory and SQLite long-term memory."""

    def __init__(self):
        self.settings = get_settings()
        self.store = SQLiteMemoryStore()
        self._short_term: Dict[str, Deque[dict]] = defaultdict(
            lambda: deque(maxlen=self.settings.short_term_turns * 2)
        )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        message = {"role": role, "content": content}
        self._short_term[session_id].append(message)
        self.store.add_message(session_id=session_id, role=role, content=content)

    def load_memory(self, session_id: str) -> List[dict]:
        long_term_messages = self.store.get_messages(session_id, limit=20)
        short_term_messages = list(self._short_term[session_id])

        # Prefer short-term messages when available; otherwise use persisted history.
        if short_term_messages:
            return short_term_messages

        return [{"role": m["role"], "content": m["content"]} for m in long_term_messages]

    def format_for_prompt(self, messages: List[dict]) -> str:
        if not messages:
            return "No previous conversation."

        return "\n".join(
            f"{message['role'].capitalize()}: {message['content']}" for message in messages
        )
