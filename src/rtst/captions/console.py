"""Console caption sink.

Renders captions to a terminal, overwriting the current line in place (carriage
return) so partial hypotheses refine smoothly instead of scrolling. Finalized
captions are committed with a newline. This backend has no GUI dependency and
is the default, which makes the engine usable over SSH and easy to test.
"""

from __future__ import annotations

import sys

from rtst.captions.base import CaptionSink
from rtst.types import Caption


class ConsoleCaptionSink(CaptionSink):
    def __init__(self, mode: str = "compact", stream=None) -> None:  # noqa: ANN001
        self.mode = mode
        self._stream = stream if stream is not None else sys.stdout
        self._last_len = 0

    def _format(self, caption: Caption) -> str:
        if self.mode == "dual" and caption.original:
            return f"{caption.original}  →  {caption.translated}"
        return caption.translated

    def _write_line(self, text: str, *, final: bool) -> None:
        # Pad with spaces to clear any leftover characters from a longer line.
        padding = max(0, self._last_len - len(text))
        self._stream.write("\r" + text + " " * padding)
        if final:
            self._stream.write("\n")
            self._last_len = 0
        else:
            self._last_len = len(text)
        self._stream.flush()

    def update(self, caption: Caption) -> None:
        self._write_line(self._format(caption), final=caption.is_final)

    def status(self, message: str) -> None:
        self._write_line(f"[{message}]", final=False)

    def close(self) -> None:
        if self._last_len:
            self._stream.write("\n")
            self._stream.flush()
            self._last_len = 0
