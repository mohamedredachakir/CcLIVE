"""Real-time speech translation engine.

A local-first pipeline that captures live audio, transcribes it with a streaming
ASR backend, translates the text with a context-aware translator, and renders
low-latency captions suitable for a meeting overlay.

The package is organised into swappable layers so that the orchestration logic
(:mod:`rtst.pipeline`) can be unit-tested with lightweight fakes without pulling
in the heavy ASR / translation / GUI dependencies.
"""

from rtst.types import AudioChunk, Caption, TranscriptSegment

__all__ = ["AudioChunk", "Caption", "TranscriptSegment", "__version__"]

__version__ = "0.1.0"
