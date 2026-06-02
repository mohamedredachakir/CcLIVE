"""Streaming voice-activity detection and segmentation.

The segmenter consumes a continuous stream of audio frames and emits events as
speech starts, continues (partial) and ends. It is deliberately backend
agnostic and pure-Python so it can be unit-tested without any audio hardware:

* If ``webrtcvad`` is installed it is used for robust speech detection.
* Otherwise a short-term energy gate is used as a fallback.

Segmentation rules (all configurable via :class:`rtst.config.AudioConfig`):

* A segment opens on the first speech frame.
* While speech continues, a ``partial`` event is emitted every
  ``partial_interval_ms`` so downstream layers can show low-latency captions.
* A segment finalizes after ``silence_ms`` of trailing silence, or when it
  exceeds ``max_segment_ms`` (force flush so captions keep moving).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EventKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


@dataclass(slots=True)
class SegmentEvent:
    kind: EventKind
    audio: np.ndarray  # float32 mono, the accumulated segment so far
    sample_rate: int
    start_time: float
    end_time: float

    @property
    def is_final(self) -> bool:
        return self.kind is EventKind.FINAL


def _energy_is_speech(frame: np.ndarray, threshold: float) -> bool:
    if frame.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return rms >= threshold


class _WebrtcGate:
    """Thin wrapper around webrtcvad operating on float frames."""

    def __init__(self, aggressiveness: int, sample_rate: int) -> None:
        import webrtcvad  # imported lazily; optional dependency

        self._vad = webrtcvad.Vad(aggressiveness)
        self._sample_rate = sample_rate

    def is_speech(self, frame: np.ndarray) -> bool:
        pcm16 = np.clip(frame, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2").tobytes()
        try:
            return self._vad.is_speech(pcm16, self._sample_rate)
        except Exception:
            return False


class StreamingSegmenter:
    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
        silence_ms: int = 600,
        max_segment_ms: int = 6_000,
        partial_interval_ms: int = 700,
        aggressiveness: int = 2,
        energy_threshold: float = 0.012,
        use_webrtc: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000))
        self.silence_frames = max(1, int(silence_ms / frame_ms))
        self.max_segment_frames = max(1, int(max_segment_ms / frame_ms))
        self.partial_interval_frames = max(1, int(partial_interval_ms / frame_ms))
        self.energy_threshold = energy_threshold

        self._gate: _WebrtcGate | None = None
        if use_webrtc:
            try:
                self._gate = _WebrtcGate(aggressiveness, sample_rate)
            except Exception:
                self._gate = None  # fall back to energy gate

        self._buffer = np.empty(0, dtype=np.float32)  # leftover < one frame
        self._segment: list[np.ndarray] = []
        self._in_speech = False
        self._silence_run = 0
        self._frames_since_partial = 0
        self._segment_frames = 0
        self._elapsed_frames = 0  # global frame counter, for timestamps

    @property
    def uses_webrtc(self) -> bool:
        return self._gate is not None

    def _frame_is_speech(self, frame: np.ndarray) -> bool:
        if self._gate is not None:
            return self._gate.is_speech(frame)
        return _energy_is_speech(frame, self.energy_threshold)

    def _segment_audio(self) -> np.ndarray:
        if not self._segment:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._segment)

    def _frame_time(self, frame_index: int) -> float:
        return frame_index * self.frame_samples / self.sample_rate

    def _start_time(self) -> float:
        return self._frame_time(self._elapsed_frames - self._segment_frames)

    def accept(self, samples: np.ndarray) -> list[SegmentEvent]:
        """Feed an arbitrary-length audio buffer; return any emitted events."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._buffer.size:
            samples = np.concatenate([self._buffer, samples])

        events: list[SegmentEvent] = []
        n_frames = samples.size // self.frame_samples
        for i in range(n_frames):
            frame = samples[i * self.frame_samples : (i + 1) * self.frame_samples]
            events.extend(self._process_frame(frame))

        self._buffer = samples[n_frames * self.frame_samples :].copy()
        return events

    def _process_frame(self, frame: np.ndarray) -> list[SegmentEvent]:
        self._elapsed_frames += 1
        events: list[SegmentEvent] = []
        speech = self._frame_is_speech(frame)

        if not self._in_speech:
            if speech:
                self._in_speech = True
                self._segment = [frame]
                self._segment_frames = 1
                self._silence_run = 0
                self._frames_since_partial = 0
            return events

        # Currently inside a segment.
        self._segment.append(frame)
        self._segment_frames += 1
        self._frames_since_partial += 1
        self._silence_run = 0 if speech else self._silence_run + 1

        end_of_speech = self._silence_run >= self.silence_frames
        too_long = self._segment_frames >= self.max_segment_frames

        if end_of_speech or too_long:
            events.append(
                SegmentEvent(
                    EventKind.FINAL,
                    self._segment_audio(),
                    self.sample_rate,
                    self._start_time(),
                    self._frame_time(self._elapsed_frames),
                )
            )
            self._reset_segment()
            return events

        if self._frames_since_partial >= self.partial_interval_frames:
            self._frames_since_partial = 0
            events.append(
                SegmentEvent(
                    EventKind.PARTIAL,
                    self._segment_audio(),
                    self.sample_rate,
                    self._start_time(),
                    self._frame_time(self._elapsed_frames),
                )
            )
        return events

    def _reset_segment(self) -> None:
        self._segment = []
        self._in_speech = False
        self._silence_run = 0
        self._frames_since_partial = 0
        self._segment_frames = 0

    def flush(self) -> list[SegmentEvent]:
        """Emit any in-progress segment as final (call when the stream ends)."""
        if not self._in_speech or not self._segment:
            self._buffer = np.empty(0, dtype=np.float32)
            return []
        event = SegmentEvent(
            EventKind.FINAL,
            self._segment_audio(),
            self.sample_rate,
            self._start_time(),
            self._frame_time(self._elapsed_frames),
        )
        self._reset_segment()
        self._buffer = np.empty(0, dtype=np.float32)
        return [event]
