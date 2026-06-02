from rtst import languages


def test_normalize_codes_names_and_aliases():
    assert languages.normalize("en") == "en"
    assert languages.normalize("French") == "fr"
    assert languages.normalize("FRANÇAIS") == "fr"
    assert languages.normalize("darija") == "ar"
    assert languages.normalize("en-US") == "en"
    assert languages.normalize("zh_Hans") == "zh"
    assert languages.normalize("") == "auto"


def test_normalize_unknown_passes_through_base():
    assert languages.normalize("sw-KE") == "sw"


def test_to_nllb_mapping():
    assert languages.to_nllb("en") == "eng_Latn"
    assert languages.to_nllb("ar") == "arb_Arab"
    # Unknown code falls back to a Latin-script guess.
    assert languages.to_nllb("xx") == "xx_Latn"


def test_supported_excludes_auto():
    codes = {lang.code for lang in languages.supported()}
    assert "auto" not in codes
    assert {"en", "fr", "ar", "es", "de", "it"} <= codes


def test_display_name_and_is_known():
    assert languages.display_name("fr") == "French"
    assert languages.display_name("xx") == "xx"
    assert languages.is_known("Spanish")
    assert not languages.is_known("klingon")
