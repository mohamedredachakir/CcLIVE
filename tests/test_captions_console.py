import io

from rtst.captions.console import ConsoleCaptionSink
from rtst.types import Caption


def test_partial_updates_overwrite_in_place():
    buf = io.StringIO()
    sink = ConsoleCaptionSink(mode="compact", stream=buf)
    sink.update(Caption(translated="bonjour", is_final=False))
    sink.update(Caption(translated="bonjour le", is_final=False))
    out = buf.getvalue()
    # Carriage returns rewrite the same line rather than printing new lines.
    assert "\r" in out
    assert out.count("\n") == 0


def test_final_commits_with_newline():
    buf = io.StringIO()
    sink = ConsoleCaptionSink(mode="compact", stream=buf)
    sink.update(Caption(translated="bonjour", is_final=True))
    assert buf.getvalue().endswith("\n")


def test_dual_mode_shows_original_and_translation():
    buf = io.StringIO()
    sink = ConsoleCaptionSink(mode="dual", stream=buf)
    sink.update(Caption(translated="hello", original="bonjour", is_final=True))
    out = buf.getvalue()
    assert "bonjour" in out and "hello" in out


def test_status_line():
    buf = io.StringIO()
    sink = ConsoleCaptionSink(stream=buf)
    sink.status("Listening…")
    assert "Listening…" in buf.getvalue()
