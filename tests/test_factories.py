import pytest

from rtst.asr import create_asr
from rtst.captions import ConsoleCaptionSink, create_caption_sink
from rtst.config import ASRConfig, CaptionConfig, TranslationConfig
from rtst.translate import IdentityTranslator, create_translator


def test_create_translator_identity():
    t = create_translator(TranslationConfig(backend="identity"))
    assert isinstance(t, IdentityTranslator)
    assert t.translate("hi", source="en", target="fr") == "hi"


def test_create_translator_unknown_raises():
    with pytest.raises(ValueError):
        create_translator(TranslationConfig(backend="nope"))


def test_create_caption_sink_console():
    sink = create_caption_sink(CaptionConfig(backend="console"))
    assert isinstance(sink, ConsoleCaptionSink)


def test_create_caption_sink_unknown_raises():
    with pytest.raises(ValueError):
        create_caption_sink(CaptionConfig(backend="nope"))


def test_create_asr_unknown_raises():
    with pytest.raises(ValueError):
        create_asr(ASRConfig(backend="nope"))
