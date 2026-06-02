"""Tests for audio capture device resolution and frame post-processing.

These exercise the loopback logic without any real audio hardware by passing a
fake ``sounddevice``-like object into :func:`resolve_input_device`.
"""

from __future__ import annotations

import numpy as np
import pytest

from rtst.audio.capture import _resample, _to_mono, resolve_input_device


class _WasapiSettings:
    def __init__(self, loopback: bool = False) -> None:
        self.loopback = loopback


class _FakeSD:
    def __init__(self, hostapis, devices) -> None:
        self._hostapis = hostapis
        self._devices = devices
        self.WasapiSettings = _WasapiSettings

    def query_hostapis(self):
        return self._hostapis

    def query_devices(self, index=None):
        if index is None:
            return self._devices
        return self._devices[index]


def test_microphone_path_is_passthrough():
    sd = _FakeSD([], [])
    device, channels, rate, extra = resolve_input_device(sd, None, False, 16_000)
    assert (device, channels, rate, extra) == (None, 1, 16_000, None)


def test_windows_wasapi_loopback_uses_default_output():
    devices = [
        {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0,
         "default_samplerate": 16_000},
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": 48_000},
    ]
    hostapis = [{"name": "Windows WASAPI", "default_output_device": 1}]
    sd = _FakeSD(hostapis, devices)

    device, channels, rate, extra = resolve_input_device(sd, None, True, 16_000)

    assert device == 1  # the speakers (output) device
    assert channels == 2
    assert rate == 48_000  # device native rate; capture layer resamples to 16k
    assert isinstance(extra, _WasapiSettings) and extra.loopback is True


def test_linux_monitor_source_is_selected():
    devices = [
        {"name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0,
         "default_samplerate": 44_100},
        {"name": "Monitor of Built-in Audio", "max_input_channels": 2, "max_output_channels": 0,
         "default_samplerate": 44_100},
    ]
    # No WASAPI host api on Linux.
    sd = _FakeSD([{"name": "ALSA", "default_output_device": -1}], devices)

    device, channels, rate, extra = resolve_input_device(sd, None, True, 16_000)

    assert device == 1  # the ".monitor" source
    assert channels == 2
    assert rate == 44_100
    assert extra is None


def test_explicit_device_with_loopback_is_respected():
    devices = [
        {"name": "Some Monitor", "max_input_channels": 2, "max_output_channels": 0,
         "default_samplerate": 48_000},
    ]
    sd = _FakeSD([{"name": "ALSA", "default_output_device": -1}], devices)

    device, channels, rate, extra = resolve_input_device(sd, 0, True, 16_000)

    assert device == 0
    assert channels == 2
    assert rate == 48_000
    assert extra is None


def test_loopback_without_any_source_raises():
    devices = [
        {"name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0,
         "default_samplerate": 44_100},
    ]
    sd = _FakeSD([{"name": "ALSA", "default_output_device": -1}], devices)

    with pytest.raises(RuntimeError, match="loopback"):
        resolve_input_device(sd, None, True, 16_000)


def test_to_mono_downmixes_stereo():
    stereo = np.array([[0.0, 1.0], [0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)
    mono = _to_mono(stereo)
    assert mono.shape == (3,)
    np.testing.assert_allclose(mono, [0.5, 0.5, 0.0])


def test_resample_changes_length_and_is_noop_when_equal():
    audio = np.linspace(-1.0, 1.0, num=48_000, dtype=np.float32)
    down = _resample(audio, 48_000, 16_000)
    assert down.size == 16_000
    assert down.dtype == np.float32
    same = _resample(audio, 16_000, 16_000)
    assert same is audio  # unchanged object when rates match
