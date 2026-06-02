import pytest

from rtst.translate.nllb import _target_bos_id


class _LangMapTokenizer:
    """Tokenizer exposing the authoritative lang_code_to_id map."""

    lang_code_to_id = {"fra_Latn": 256057, "eng_Latn": 256047}


class _ConvertTokenizer:
    """Tokenizer that only exposes convert_tokens_to_ids (+ an UNK id)."""

    unk_token_id = 3

    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def convert_tokens_to_ids(self, token: str) -> int:
        # Mirrors HF behaviour: unknown tokens resolve to the UNK id.
        return self._vocab.get(token, self.unk_token_id)


def test_uses_lang_code_to_id_when_available():
    assert _target_bos_id(_LangMapTokenizer(), "fra_Latn") == 256057


def test_convert_returns_known_token():
    tok = _ConvertTokenizer({"fra_Latn": 1234})
    assert _target_bos_id(tok, "fra_Latn") == 1234


def test_unknown_code_raises_instead_of_returning_unk():
    tok = _ConvertTokenizer({"fra_Latn": 1234})
    # "xx_Latn" is not in the vocab -> convert returns unk_token_id (3).
    # That must raise, not silently return the UNK id.
    with pytest.raises(ValueError):
        _target_bos_id(tok, "xx_Latn")


def test_missing_from_lang_map_falls_through_to_raise():
    with pytest.raises(ValueError):
        _target_bos_id(_LangMapTokenizer(), "zzz_Latn")
