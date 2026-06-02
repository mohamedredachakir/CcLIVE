"""ASR backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rtst.types import TranscriptSegment


class ASRBackend(ABC):
    """Transcribes a chunk of audio into a :class:`TranscriptSegment`.

    Implementations should be stateless with respect to a single ``transcribe``
    call: the segmenter is responsible for deciding what audio constitutes a
    partial vs final segment. ``is_final`` is forwarded so backends may choose
    to use cheaper settings for partial hypotheses.
    """

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        is_final: bool = False,
        language: str | None = None,
    ) -> TranscriptSegment: ...

    def warmup(self) -> None:  # pragma: no cover - optional hook
        """Optionally pre-load weights so the first real call is not slow."""
        return None
