"""Shared test fakes for the lightweight (no-model) test-suite."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from rtst.asr.base import ASRBackend
from rtst.captions.base import CaptionSink
from rtst.translate.base import Translator
from rtst.types import Caption, TranscriptSegment


class FakeASR(ASRBackend):
    """Returns text proportional to the audio length so partials < finals."""

    def __init__(self, words: list[str] | None = None) -> None:
        self.words = words or ["hello", "world", "this", "is", "a", "test"]
        self.calls: list[bool] = []

    def transcribe(self, audio, sample_rate, *, is_final=False, language=None):  # noqa: ANN001
        self.calls.append(is_final)
        # Roughly one word per 0.3s of audio, so longer audio -> more words.
        seconds = len(np.asarray(audio).reshape(-1)) / max(1, sample_rate)
        n = max(1, min(len(self.words), int(seconds / 0.3) + 1))
        return TranscriptSegment(
            text=" ".join(self.words[:n]),
            language=language or "en",
            is_final=is_final,
        )


class FakeTranslator(Translator):
    """Records context seen and prefixes text so output differs from input."""

    def __init__(self) -> None:
        self.seen_context: list[Sequence[str]] = []
        self.calls: list[str] = []

    def translate(self, text, *, source, target, context=()):  # noqa: ANN001
        self.seen_context.append(list(context))
        self.calls.append(text)
        return f"[{target}] {text}"


class RecordingSink(CaptionSink):
    def __init__(self) -> None:
        self.captions: list[Caption] = []
        self.statuses: list[str] = []
        self.closed = False

    def update(self, caption: Caption) -> None:
        self.captions.append(caption)

    def status(self, message: str) -> None:
        self.statuses.append(message)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def conftest_fakes():
    """Convenience fixture returning the (ASR, Translator, Sink) fake classes."""
    return FakeASR, FakeTranslator, RecordingSink
