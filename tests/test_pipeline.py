import numpy as np

from rtst.config import ASRConfig, AudioConfig, Config, TranslationConfig
from rtst.pipeline import Pipeline
from rtst.types import Caption

SR = 16_000
FRAME = int(SR * 30 / 1000)


def _config(**kw) -> Config:
    base = dict(
        source_language="en",
        target_language="fr",
        audio=AudioConfig(silence_ms=300, partial_interval_ms=300),
        asr=ASRConfig(),
        translation=TranslationConfig(backend="identity", min_partial_delta=1),
    )
    base.update(kw)
    return Config(**base)


def _tone(n_frames: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(n_frames * FRAME) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(n_frames: int) -> np.ndarray:
    return np.zeros(n_frames * FRAME, dtype=np.float32)


def _make_pipeline(config, FakeASR, FakeTranslator, RecordingSink):
    from rtst.audio.vad import StreamingSegmenter

    seg = StreamingSegmenter(
        sample_rate=SR, frame_ms=30, silence_ms=300, partial_interval_ms=300,
        max_segment_ms=6000, use_webrtc=False, energy_threshold=0.01,
    )
    sink = RecordingSink()
    asr = FakeASR()
    translator = FakeTranslator()
    pipe = Pipeline(config, asr, translator, sink, segmenter=seg)
    return pipe, asr, translator, sink


def test_emits_final_caption(conftest_fakes):
    FakeASR, FakeTranslator, RecordingSink = conftest_fakes
    pipe, asr, translator, sink = _make_pipeline(_config(), FakeASR, FakeTranslator, RecordingSink)
    audio = np.concatenate([_silence(3), _tone(25), _silence(15)])
    pipe.feed(audio)
    finals = [c for c in sink.captions if c.is_final]
    assert len(finals) == 1
    assert finals[0].translated.startswith("[fr]")
    assert finals[0].original  # original text preserved


def test_partial_then_final_ordering(conftest_fakes):
    FakeASR, FakeTranslator, RecordingSink = conftest_fakes
    pipe, asr, translator, sink = _make_pipeline(_config(), FakeASR, FakeTranslator, RecordingSink)
    pipe.feed(np.concatenate([_tone(25), _silence(15)]))
    kinds = [c.is_final for c in sink.captions]
    assert kinds[-1] is True
    assert any(k is False for k in kinds)  # at least one partial before final


def test_final_text_added_to_context(conftest_fakes):
    FakeASR, FakeTranslator, RecordingSink = conftest_fakes
    config = _config()
    pipe, asr, translator, sink = _make_pipeline(config, FakeASR, FakeTranslator, RecordingSink)
    # Two separate utterances.
    pipe.feed(np.concatenate([_tone(20), _silence(15)]))
    pipe.feed(np.concatenate([_tone(20), _silence(15)]))
    # On the second utterance the translator should have received context.
    assert any(len(c) > 0 for c in translator.seen_context)


def test_same_language_skips_translation(conftest_fakes):
    FakeASR, FakeTranslator, RecordingSink = conftest_fakes
    pipe, asr, translator, sink = _make_pipeline(
        _config(source_language="en", target_language="en"),
        FakeASR, FakeTranslator, RecordingSink,
    )
    pipe.feed(np.concatenate([_tone(20), _silence(15)]))
    assert translator.calls == []  # identity short-circuit, translator untouched
    assert any(c.is_final for c in sink.captions)


def test_partial_dedup_respects_min_delta(conftest_fakes):
    FakeASR, FakeTranslator, RecordingSink = conftest_fakes
    config = _config(translation=TranslationConfig(backend="identity", min_partial_delta=1000))
    pipe, asr, translator, sink = _make_pipeline(config, FakeASR, FakeTranslator, RecordingSink)
    pipe.feed(np.concatenate([_tone(25), _silence(15)]))
    partials = [c for c in sink.captions if not c.is_final]
    assert partials == []  # huge min delta suppresses all partials


def test_run_streams_and_closes(conftest_fakes):
    FakeASR, FakeTranslator, RecordingSink = conftest_fakes
    pipe, asr, translator, sink = _make_pipeline(_config(), FakeASR, FakeTranslator, RecordingSink)
    frames = [_tone(2) for _ in range(15)] + [_silence(2) for _ in range(15)]
    captions = list(pipe.run(iter(frames)))
    assert sink.closed is True
    assert "Listening…" in sink.statuses
    assert all(isinstance(c, Caption) for c in captions)
