"""Translator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Translator(ABC):
    """Translates text from a source language to a target language.

    ``context`` carries recent finalized utterances so backends that support it
    can produce more coherent, meaning-preserving output instead of literal
    word-by-word translations. Backends that cannot use context simply ignore it.
    """

    @abstractmethod
    def translate(
        self,
        text: str,
        *,
        source: str,
        target: str,
        context: Sequence[str] = (),
    ) -> str: ...

    def warmup(self) -> None:  # pragma: no cover - optional hook
        return None
