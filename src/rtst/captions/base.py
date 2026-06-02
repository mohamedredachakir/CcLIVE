"""Caption sink interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rtst.types import Caption


class CaptionSink(ABC):
    """Renders captions. Implementations update in place rather than spamming
    new lines: a partial caption is repeatedly overwritten until it is final.
    """

    @abstractmethod
    def update(self, caption: Caption) -> None:
        """Show/replace the current caption (partial or final)."""

    def status(self, message: str) -> None:  # noqa: B027  # optional hook
        """Show a transient status line (e.g. ``Listening…``)."""

    def close(self) -> None:  # pragma: no cover - optional
        return None
