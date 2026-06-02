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
