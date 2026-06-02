"""faster-whisper (CTranslate2) streaming ASR backend.

``faster-whisper`` is a re-implementation of OpenAI Whisper that runs several
times faster on CPU and GPU, which makes it a good fit for the low-latency
streaming use-case. The model is loaded lazily on first use so importing this
module never triggers a multi-GB download.
"""

from __future__ import annotations

import numpy as np

from rtst.asr.base import ASRBackend
from rtst.types import TranscriptSegment


def _resolve_device(device: str) -> tuple[str, str]:
    """Resolve ('auto'|'cpu'|'cuda') to a concrete (device, default_compute)."""
    if device and device != "auto":
        return device, ("float16" if device == "cuda" else "int8")
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


class FasterWhisperASR(ASRBackend):
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 1,
        language: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.beam_size = beam_size
        self.language = language
        self._device, default_compute = _resolve_device(device)
        self.compute_type = default_compute if compute_type == "auto" else compute_type
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self._device,
                compute_type=self.compute_type,
            )
        return self._model

    def warmup(self) -> None:  # pragma: no cover - requires model download
        model = self._ensure_model()
        model.transcribe(np.zeros(1600, dtype=np.float32), beam_size=1)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        is_final: bool = False,
        language: str | None = None,
    ) -> TranscriptSegment:
        model = self._ensure_model()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16_000:
            audio = _resample(audio, sample_rate, 16_000)

        lang = language or self.language
        if lang == "auto":
            lang = None

        segments, info = model.transcribe(
            audio,
            language=lang,
            beam_size=self.beam_size if is_final else 1,
            # Partial hypotheses favour speed; final ones favour quality.
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscriptSegment(
            text=text,
            language=getattr(info, "language", None) or lang,
            is_final=is_final,
            confidence=getattr(info, "language_probability", None),
        )


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.size / src_rate
    dst_len = int(round(duration * dst_rate))
    if dst_len <= 0:
        return np.empty(0, dtype=np.float32)
    src_idx = np.linspace(0, audio.size - 1, num=dst_len)
    return np.interp(src_idx, np.arange(audio.size), audio).astype(np.float32)
