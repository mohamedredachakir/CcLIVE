import numpy as np

from rtst.audio.vad import EventKind, StreamingSegmenter

SR = 16_000
FRAME_MS = 30
FRAME = int(SR * FRAME_MS / 1000)  # 480


def _silence(n_frames: int) -> np.ndarray:
    return np.zeros(n_frames * FRAME, dtype=np.float32)


def _tone(n_frames: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(n_frames * FRAME) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _segmenter(**kw):
    defaults = dict(
        sample_rate=SR,
        frame_ms=FRAME_MS,
        silence_ms=300,  # 10 frames
        max_segment_ms=6_000,
        partial_interval_ms=300,  # 10 frames
        use_webrtc=False,
        energy_threshold=0.01,
    )
    defaults.update(kw)
    return StreamingSegmenter(**defaults)


def test_uses_energy_gate_when_webrtc_disabled():
    seg = _segmenter()
    assert seg.uses_webrtc is False


def test_detects_speech_segment_with_trailing_silence():
    seg = _segmenter()
    audio = np.concatenate([_silence(5), _tone(25), _silence(15)])
    events = seg.accept(audio)
    finals = [e for e in events if e.kind is EventKind.FINAL]
    partials = [e for e in events if e.kind is EventKind.PARTIAL]
    assert len(finals) == 1
    assert len(partials) >= 1
    # Final segment = speech frames + the trailing silence that closed it.
    assert finals[0].audio.size > 25 * FRAME * 0.9


def test_pure_silence_emits_nothing():
    seg = _segmenter()
    assert seg.accept(_silence(40)) == []


def test_flush_finalizes_in_progress_segment():
    seg = _segmenter()
    # Speech that never ends with silence -> nothing final until flush.
    events = seg.accept(_tone(8))
    assert all(e.kind is not EventKind.FINAL for e in events)
    flushed = seg.flush()
    assert len(flushed) == 1
    assert flushed[0].kind is EventKind.FINAL


def test_max_segment_force_flush():
    # Long continuous speech should be force-flushed even without silence.
    seg = _segmenter(max_segment_ms=300)  # 10 frames max
    events = seg.accept(_tone(40))
    assert any(e.kind is EventKind.FINAL for e in events)


def test_handles_unaligned_buffer_lengths():
    seg = _segmenter()
    # Feed half-frame chunks; segmenter should buffer the remainder.
    tone = _tone(25)
    chunk = FRAME // 2 + 7
    events = []
    for i in range(0, tone.size, chunk):
        events += seg.accept(tone[i : i + chunk])
    events += seg.accept(_silence(15))
    assert any(e.kind is EventKind.FINAL for e in events)
