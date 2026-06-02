import pytest

from rtst import cli


def test_list_languages_command(capsys):
    rc = cli.main(["list-languages"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "French" in out and "Arabic" in out


def test_no_command_prints_help(capsys):
    rc = cli.main([])
    assert rc == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_config_from_args_normalizes_and_parses_device():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["run", "-s", "French", "-t", "English", "--device", "3", "--captions", "console"]
    )
    config = cli._config_from_args(args)
    assert config.source_language == "fr"
    assert config.target_language == "en"
    assert config.audio.device == 3


def test_translation_backend_choice_enforced():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--translation-backend", "bogus"])


def test_defaults_without_preset():
    parser = cli.build_parser()
    config = cli._config_from_args(parser.parse_args(["run", "-t", "fr"]))
    assert config.asr.model_size == "small"
    assert config.translation.backend == "nllb"
    assert config.audio.partial_interval_ms == 700


def test_preset_fast_tunes_for_realtime():
    parser = cli.build_parser()
    config = cli._config_from_args(parser.parse_args(["run", "-t", "fr", "--preset", "fast"]))
    assert config.asr.model_size == "base"
    assert config.translation.backend == "argos"
    assert config.audio.partial_interval_ms == 900
    assert config.translation.min_partial_delta == 8


def test_explicit_flags_override_preset():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["run", "-t", "fr", "--preset", "fast", "--model", "small",
         "--translation-backend", "nllb"]
    )
    config = cli._config_from_args(args)
    assert config.asr.model_size == "small"
    assert config.translation.backend == "nllb"


def test_preset_choice_enforced():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--preset", "turbo"])
