"""A no-op translator used for testing and same-language passthrough."""

from __future__ import annotations

from collections.abc import Sequence

from rtst.translate.base import Translator


class IdentityTranslator(Translator):
    """Returns the input unchanged. Useful for tests and ``source == target``."""

    def translate(
        self,
        text: str,
        *,
        source: str,
        target: str,
        context: Sequence[str] = (),
    ) -> str:
        return text
