"""Live audio capture via ``sounddevice``.

Provides a blocking generator of float32 mono frames at the configured sample
rate. Both microphone input and system-audio loopback are supported; loopback
relies on an OS monitor/loopback device (e.g. a PulseAudio ``.monitor`` source
or a virtual audio cable) being selected via ``device``.

This module imports ``sounddevice`` lazily so the rest of the package (and the
test-suite) does not require the PortAudio native library to be installed.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator

import numpy as np


def list_devices() -> list[dict]:
    """Return the available audio input devices (name, channels, sample rate)."""
    import sounddevice as sd

    devices = []
    for index, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append(
                {
                    "index": index,
                    "name": dev["name"],
                    "max_input_channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                }
            )
    return devices


def _to_mono(block: np.ndarray) -> np.ndarray:
    if block.ndim == 2 and block.shape[1] > 1:
        return block.mean(axis=1)
    return block.reshape(-1)


class MicrophoneStream:
    """Context manager yielding mono float32 frames from an input device."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        block_ms: int = 30,
        device: str | int | None = None,
        channels: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = max(1, int(sample_rate * block_ms / 1000))
        self.device = device
        self.channels = channels
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            # Overflows are expected occasionally under load; keep streaming.
            pass
        self._queue.put(_to_mono(np.asarray(indata, dtype=np.float32)).copy())

    def __enter__(self) -> MicrophoneStream:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def frames(self) -> Iterator[np.ndarray]:
        """Yield captured frames until the stream is closed."""
        while True:
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stream is None:
                    return
                continue
