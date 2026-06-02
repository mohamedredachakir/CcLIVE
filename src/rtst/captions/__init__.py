"""Caption rendering backends."""

from rtst.captions.base import CaptionSink
from rtst.captions.console import ConsoleCaptionSink

__all__ = ["CaptionSink", "ConsoleCaptionSink", "create_caption_sink"]


def create_caption_sink(config) -> CaptionSink:  # noqa: ANN001
    """Instantiate a caption sink from a :class:`rtst.config.CaptionConfig`."""
    backend = (config.backend or "console").lower()
    if backend == "console":
        return ConsoleCaptionSink(mode=config.mode)
    if backend == "overlay":
        from rtst.captions.overlay import OverlayCaptionSink

        return OverlayCaptionSink(
            mode=config.mode,
            max_lines=config.max_lines,
            linger_seconds=config.linger_seconds,
            font_point_size=config.font_point_size,
            opacity=config.opacity,
        )
    raise ValueError(f"Unknown caption backend: {config.backend!r}")
