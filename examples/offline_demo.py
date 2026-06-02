"""Offline demo of the pipeline — no models, no microphone required.

Feeds synthetic "speech bursts" through the real segmenter, a scripted ASR that
returns canned text (so you can see partial -> final caption behaviour), and the
identity translator. Useful to see the caption output format and latency policy
without downloading multi-GB ASR/translation models.

Run:  python examples/offline_demo.py
"""

from __future__ import annotations

import time

import numpy as np

from rtst.asr.base import ASRBackend
from rtst.captions.console import ConsoleCaptionSink
from rtst.config import ASRConfig, AudioConfig, Config, TranslationConfig
from rtst.pipeline import Pipeline
from rtst.translate.identity import IdentityTranslator
from rtst.types import TranscriptSegment

SR = 16_000
FRAME = int(SR * 0.03)

# A canned utterance revealed progressively as more audio arrives.
SCRIPT = "the quarterly numbers look strong and we should ship next week"


class ScriptedASR(ASRBackend):
    """Reveals more of SCRIPT as the audio segment grows (simulates streaming)."""

    def transcribe(self, audio, sample_rate, *, is_final=False, language=None):  # noqa: ANN001
        seconds = len(np.asarray(audio).reshape(-1)) / max(1, sample_rate)
        words = SCRIPT.split()
        n = len(words) if is_final else max(1, min(len(words), int(seconds / 0.25)))
        return TranscriptSegment(" ".join(words[:n]), language="en", is_final=is_final)


def _tone(n_frames: int) -> np.ndarray:
    t = np.arange(n_frames * FRAME) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(n_frames: int) -> np.ndarray:
    return np.zeros(n_frames * FRAME, dtype=np.float32)


def main() -> None:
    config = Config(
        source_language="en",
        target_language="fr",
        audio=AudioConfig(silence_ms=300, partial_interval_ms=300),
        asr=ASRConfig(),
        translation=TranslationConfig(backend="identity", min_partial_delta=1),
    )
    pipeline = Pipeline(
        config, ScriptedASR(), IdentityTranslator(), ConsoleCaptionSink(mode="dual")
    )

    print("Simulating a spoken sentence (partial captions refine, then finalize):\n")
    # 30 frames of speech delivered in small chunks, then a pause to finalize.
    for _ in range(15):
        pipeline.feed(_tone(2))
        time.sleep(0.08)  # pace it so the in-place updates are visible
    pipeline.feed(_silence(15))
    pipeline.flush()
    print("\nDone.")


if __name__ == "__main__":
    main()
