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


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample (cheap; good enough before 16 kHz ASR)."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    dst_len = int(round(audio.size * dst_rate / src_rate))
    if dst_len <= 0:
        return np.empty(0, dtype=np.float32)
    src_idx = np.linspace(0, audio.size - 1, num=dst_len)
    return np.interp(src_idx, np.arange(audio.size), audio).astype(np.float32)


def resolve_input_device(sd, device, use_loopback, target_rate):  # noqa: ANN001
    """Resolve which device/host-api to capture from.

    Returns ``(device, channels, samplerate, extra_settings)``.

    For microphone capture this is just the requested device at the target
    rate. For ``--loopback`` (capturing what the *meeting* plays through your
    speakers, which is what you want for Teams/Meet/Zoom) it resolves a
    system-audio source per platform:

    * **Windows** – WASAPI loopback on the default output device (no extra
      drivers needed).
    * **Linux** – the PulseAudio/PipeWire ``.monitor`` source of the output.
    * **macOS** – requires a virtual device (e.g. BlackHole); pass ``--device``.
    """
    if not use_loopback:
        return device, 1, target_rate, None

    # 1) Windows WASAPI loopback on the default output device.
    try:
        for api in sd.query_hostapis():
            if "WASAPI" in api.get("name", ""):
                out_idx = api.get("default_output_device", -1)
                if out_idx is not None and out_idx >= 0:
                    dev = sd.query_devices(out_idx)
                    ch = max(1, int(dev.get("max_output_channels", 2)))
                    rate = int(dev.get("default_samplerate") or target_rate)
                    return out_idx, ch, rate, sd.WasapiSettings(loopback=True)
    except Exception:
        pass

    # 2) An explicit monitor/loopback device was named alongside --loopback.
    if device is not None:
        dev = sd.query_devices(device)
        ch = max(1, int(dev.get("max_input_channels", 1) or 1))
        rate = int(dev.get("default_samplerate") or target_rate)
        return device, ch, rate, None

    # 3) Linux PulseAudio/PipeWire: pick the output's ".monitor" source.
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0 and "monitor" in dev.get("name", "").lower():
            ch = max(1, int(dev["max_input_channels"]))
            rate = int(dev.get("default_samplerate") or target_rate)
            return idx, ch, rate, None

    raise RuntimeError(
        "No system-audio loopback device found. On Windows this uses WASAPI "
        "loopback automatically; on Linux select your output's '.monitor' source "
        "with --device; on macOS install a virtual loopback device (e.g. BlackHole) "
        "and pass it with --device."
    )


class MicrophoneStream:
    """Context manager yielding mono float32 frames from an input device.

    When ``use_loopback`` is set the stream captures system output audio
    instead of the microphone, downmixing to mono and resampling to the
    configured ``sample_rate`` (loopback devices commonly run at 44.1/48 kHz).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        block_ms: int = 30,
        device: str | int | None = None,
        channels: int = 1,
        use_loopback: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.block_size = max(1, int(sample_rate * block_ms / 1000))
        self.device = device
        self.channels = channels
        self.use_loopback = use_loopback
        self._actual_rate = sample_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            # Overflows are expected occasionally under load; keep streaming.
            pass
        mono = _to_mono(np.asarray(indata, dtype=np.float32))
        if self._actual_rate != self.sample_rate:
            mono = _resample(mono, self._actual_rate, self.sample_rate)
        self._queue.put(mono.copy())

    def __enter__(self) -> MicrophoneStream:
        import sounddevice as sd

        device, channels, rate, extra = resolve_input_device(
            sd, self.device, self.use_loopback, self.sample_rate
        )
        self._actual_rate = rate
        # Block size is in frames at the *device* rate so each block ~= block_ms.
        block_size = max(1, int(rate * self.block_ms / 1000))
        self._stream = sd.InputStream(
            samplerate=rate,
            blocksize=block_size,
            device=device,
            channels=channels,
            dtype="float32",
            callback=self._callback,
            extra_settings=extra,
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
