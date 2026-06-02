"""Runtime configuration for the engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from rtst import languages

# Whisper / faster-whisper expects 16 kHz mono audio.
DEFAULT_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    # Frame size handed to the VAD (10/20/30 ms are the values WebRTC VAD allows).
    frame_ms: int = 30
    # Emit a segment after this much trailing silence (speech considered ended).
    silence_ms: int = 600
    # Force-flush a segment that runs longer than this, so captions keep moving.
    max_segment_ms: int = 6_000
    # Emit a partial hypothesis at most this often while speech continues.
    partial_interval_ms: int = 700
    # 0-3 for WebRTC VAD; higher = more aggressive at filtering non-speech.
    vad_aggressiveness: int = 2
    # Loopback (system audio) vs microphone. Loopback needs an OS monitor device.
    use_loopback: bool = False
    device: str | int | None = None


@dataclass(slots=True)
class ASRConfig:
    backend: str = "faster-whisper"
    model_size: str = "small"  # tiny/base/small/medium/large-v3
    device: str = "auto"  # auto/cpu/cuda
    compute_type: str = "auto"  # e.g. int8, float16
    beam_size: int = 1  # 1 == greedy, lowest latency
    # If set, ASR assumes this source language instead of auto-detecting.
    language: str | None = None


@dataclass(slots=True)
class TranslationConfig:
    backend: str = "nllb"  # nllb/argos/identity
    model: str = "facebook/nllb-200-distilled-600M"
    device: str = "auto"
    # How many recent finalized segments to feed as context to the translator.
    context_segments: int = 6
    # Don't re-translate a partial unless it grew by at least this many chars.
    min_partial_delta: int = 4
    # Allow sending text to a cloud API. Off by default (privacy mode).
    allow_cloud: bool = False


@dataclass(slots=True)
class CaptionConfig:
    backend: str = "console"  # console/overlay
    mode: str = "compact"  # compact (translation only) / dual (original + translation)
    max_lines: int = 2
    # Remove a finalized caption after this many seconds of inactivity.
    linger_seconds: float = 4.0
    font_point_size: int = 22
    opacity: float = 0.85


@dataclass(slots=True)
class Config:
    source_language: str = "auto"
    target_language: str = "en"
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)

    def __post_init__(self) -> None:
        self.source_language = languages.normalize(self.source_language)
        self.target_language = languages.normalize(self.target_language)
        self.validate()

    def validate(self) -> None:
        if self.target_language in ("", "auto"):
            raise ValueError("target_language must be a concrete language, not 'auto'")
        if not 0 <= self.audio.vad_aggressiveness <= 3:
            raise ValueError("vad_aggressiveness must be in [0, 3]")
        if self.audio.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20 or 30 (WebRTC VAD constraint)")
        if self.audio.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 <= self.caption.opacity <= 1.0:
            raise ValueError("opacity must be in [0.0, 1.0]")

    def to_dict(self) -> dict:
        return asdict(self)
