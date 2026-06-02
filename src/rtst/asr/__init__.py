"""Speech-to-text backends."""

from rtst.asr.base import ASRBackend

__all__ = ["ASRBackend", "create_asr"]


def create_asr(config) -> ASRBackend:  # noqa: ANN001
    """Instantiate an ASR backend from an :class:`rtst.config.ASRConfig`."""
    backend = (config.backend or "faster-whisper").lower()
    if backend in ("faster-whisper", "whisper", "faster_whisper"):
        from rtst.asr.whisper_asr import FasterWhisperASR

        return FasterWhisperASR(
            model_size=config.model_size,
            device=config.device,
            compute_type=config.compute_type,
            beam_size=config.beam_size,
            language=config.language,
        )
    raise ValueError(f"Unknown ASR backend: {config.backend!r}")
