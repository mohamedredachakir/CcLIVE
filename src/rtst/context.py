"""Rolling conversation context.

Keeps the last few finalized segments (bounded by both count and wall-clock age)
so the translator can disambiguate pronouns, topic shifts and short utterances.
The translator decides how to use it; this module only stores and trims.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class _Entry:
    text: str
    timestamp: float


class ConversationContext:
    def __init__(self, max_segments: int = 6, max_age_seconds: float = 30.0) -> None:
        if max_segments < 0:
            raise ValueError("max_segments must be >= 0")
        self.max_segments = max_segments
        self.max_age_seconds = max_age_seconds
        self._entries: deque[_Entry] = deque(maxlen=max_segments or 1)

    def add(self, text: str, *, now: float | None = None) -> None:
        text = text.strip()
        if not text or self.max_segments == 0:
            return
        self._entries.append(_Entry(text, now if now is not None else time.monotonic()))

    def _trim_age(self, now: float) -> None:
        cutoff = now - self.max_age_seconds
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()

    def recent(self, *, now: float | None = None) -> list[str]:
        now = now if now is not None else time.monotonic()
        self._trim_age(now)
        return [entry.text for entry in self._entries]

    def as_prompt(self, *, separator: str = " ", now: float | None = None) -> str:
        """Render the live context as a single string for prompt-style backends."""
        return separator.join(self.recent(now=now))

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
