"""Core data structures shared across the pipeline layers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class AudioChunk:
    """A short window of mono PCM audio flowing through the pipeline.

    Attributes:
        samples: 1-D float32 array in the range [-1.0, 1.0].
        sample_rate: Sample rate of ``samples`` in Hz.
        timestamp: Monotonic capture time (seconds) of the first sample.
        is_speech: Whether the VAD considers this chunk to contain speech.
    """

    samples: np.ndarray
    sample_rate: int
    timestamp: float = field(default_factory=time.monotonic)
    is_speech: bool = True

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / float(self.sample_rate)


@dataclass(slots=True)
class TranscriptSegment:
    """A transcription hypothesis for a contiguous speech segment.

    A segment moves from ``partial`` (still being spoken / refined) to final.
    Partial hypotheses are emitted continuously for low latency; the final
    hypothesis is what gets committed to the conversation context.
    """

    text: str
    language: str | None = None
    is_final: bool = False
    start: float = 0.0
    end: float = 0.0
    confidence: float | None = None

    @property
    def is_partial(self) -> bool:
        return not self.is_final


@dataclass(slots=True)
class Caption:
    """A caption ready to be rendered on screen."""

    translated: str
    original: str = ""
    source_language: str | None = None
    target_language: str | None = None
    is_final: bool = False
    created_at: float = field(default_factory=time.monotonic)
