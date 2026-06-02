import pytest

from rtst.config import AudioConfig, CaptionConfig, Config


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.target_language == "en"
    assert cfg.audio.sample_rate == 16_000


def test_languages_are_normalized():
    cfg = Config(source_language="French", target_language="Arabic")
    assert cfg.source_language == "fr"
    assert cfg.target_language == "ar"


def test_target_cannot_be_auto():
    with pytest.raises(ValueError):
        Config(target_language="auto")


def test_invalid_frame_ms_rejected():
    with pytest.raises(ValueError):
        Config(audio=AudioConfig(frame_ms=25))


def test_invalid_vad_aggressiveness_rejected():
    with pytest.raises(ValueError):
        Config(audio=AudioConfig(vad_aggressiveness=5))


def test_invalid_opacity_rejected():
    with pytest.raises(ValueError):
        Config(caption=CaptionConfig(opacity=1.5))


def test_to_dict_roundtrips_nested():
    cfg = Config(target_language="es")
    data = cfg.to_dict()
    assert data["target_language"] == "es"
    assert data["audio"]["sample_rate"] == 16_000
