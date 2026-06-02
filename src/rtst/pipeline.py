"""Pipeline orchestration: audio → ASR → translation → captions.

This module wires the swappable layers together and owns the live-translation
policy (partial throttling, de-duplication, context management). It contains no
hard dependency on audio/ASR/translation/GUI libraries: backends are injected,
which lets the whole policy be unit-tested with fakes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import numpy as np

from rtst.asr.base import ASRBackend
from rtst.audio.vad import EventKind, SegmentEvent, StreamingSegmenter
from rtst.captions.base import CaptionSink
from rtst.config import Config
from rtst.context import ConversationContext
from rtst.translate.base import Translator
from rtst.types import Caption, TranscriptSegment

logger = logging.getLogger("rtst.pipeline")


class Pipeline:
    def __init__(
        self,
        config: Config,
        asr: ASRBackend,
        translator: Translator,
        captions: CaptionSink,
        segmenter: StreamingSegmenter | None = None,
        context: ConversationContext | None = None,
    ) -> None:
        self.config = config
        self.asr = asr
        self.translator = translator
        self.captions = captions
        self.segmenter = segmenter or _segmenter_from_config(config)
        self.context = context or ConversationContext(
            max_segments=config.translation.context_segments
        )

        # State used to suppress redundant work between consecutive events.
        self._last_partial_source = ""
        self._last_emitted_translation = ""

    # -- public API ---------------------------------------------------------

    def feed(self, samples: np.ndarray) -> list[Caption]:
        """Push raw audio; return the captions emitted as a result."""
        captions: list[Caption] = []
        for event in self.segmenter.accept(samples):
            caption = self._handle_event(event)
            if caption is not None:
                captions.append(caption)
        return captions

    def flush(self) -> list[Caption]:
        """Finalize any in-progress segment (call when the audio stream ends)."""
        captions: list[Caption] = []
        for event in self.segmenter.flush():
            caption = self._handle_event(event)
            if caption is not None:
                captions.append(caption)
        return captions

    def run(self, frames: Iterable[np.ndarray]) -> Iterator[Caption]:
        """Consume a stream of audio frames, yielding captions as they appear."""
        self.captions.status("Listening…")
        for frame in frames:
            for caption in self.feed(frame):
                yield caption
        for caption in self.flush():
            yield caption
        self.captions.close()

    # -- internals ----------------------------------------------------------

    def _handle_event(self, event: SegmentEvent) -> Caption | None:
        transcript = self._transcribe(event)
        if transcript is None or not transcript.text:
            return None

        if event.kind is EventKind.PARTIAL:
            return self._handle_partial(transcript)
        return self._handle_final(transcript)

    def _transcribe(self, event: SegmentEvent) -> TranscriptSegment | None:
        source = self.config.source_language
        try:
            return self.asr.transcribe(
                event.audio,
                event.sample_rate,
                is_final=event.is_final,
                language=None if source == "auto" else source,
            )
        except Exception:  # pragma: no cover - backend specific
            logger.exception("ASR failed for a %s segment", event.kind.value)
            return None

    def _handle_partial(self, transcript: TranscriptSegment) -> Caption | None:
        text = transcript.text
        delta = abs(len(text) - len(self._last_partial_source))
        if text == self._last_partial_source:
            return None
        if delta < self.config.translation.min_partial_delta:
            # Too small a change to be worth re-translating mid-utterance.
            return None
        self._last_partial_source = text
        translated = self._translate(text, transcript.language, final=False)
        caption = self._make_caption(transcript, translated, is_final=False)
        self.captions.update(caption)
        return caption

    def _handle_final(self, transcript: TranscriptSegment) -> Caption | None:
        text = transcript.text
        translated = self._translate(text, transcript.language, final=True)
        # Commit the source utterance to context *after* translating it.
        self.context.add(text)
        self._last_partial_source = ""
        self._last_emitted_translation = translated
        caption = self._make_caption(transcript, translated, is_final=True)
        self.captions.update(caption)
        return caption

    def _translate(self, text: str, source_lang: str | None, *, final: bool) -> str:
        source = source_lang or self.config.source_language
        target = self.config.target_language
        if source not in ("", "auto") and source == target:
            return text  # nothing to do
        try:
            return self.translator.translate(
                text,
                source=source or "auto",
                target=target,
                context=self.context.recent(),
            )
        except Exception:  # pragma: no cover - backend specific
            logger.exception("Translation failed; falling back to source text")
            return text

    def _make_caption(
        self, transcript: TranscriptSegment, translated: str, *, is_final: bool
    ) -> Caption:
        return Caption(
            translated=translated,
            original=transcript.text,
            source_language=transcript.language or self.config.source_language,
            target_language=self.config.target_language,
            is_final=is_final,
        )


def _segmenter_from_config(config: Config) -> StreamingSegmenter:
    audio = config.audio
    return StreamingSegmenter(
        sample_rate=audio.sample_rate,
        frame_ms=audio.frame_ms,
        silence_ms=audio.silence_ms,
        max_segment_ms=audio.max_segment_ms,
        partial_interval_ms=audio.partial_interval_ms,
        aggressiveness=audio.vad_aggressiveness,
    )
