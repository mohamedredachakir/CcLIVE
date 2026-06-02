"""Language registry and code normalisation.

The engine speaks in plain ISO-639-1 codes (``en``, ``fr``, ``ar`` ...) at the
API boundary and maps them to backend-specific codes where needed (e.g. NLLB's
``eng_Latn`` / ``arb_Arab``). This keeps the public surface stable regardless of
which translation backend is active.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Language:
    code: str  # ISO-639-1
    name: str
    flag: str
    nllb: str  # NLLB-200 code (FLORES-200 style)


# A curated set covering the explicitly requested languages plus common ones.
# Any language supported by the underlying model can still be passed through by
# code; this table just powers the UI picker and nicer error messages.
_LANGUAGES: tuple[Language, ...] = (
    Language("auto", "Auto-detect", "🌐", "auto"),
    Language("en", "English", "🇬🇧", "eng_Latn"),
    Language("fr", "French", "🇫🇷", "fra_Latn"),
    Language("ar", "Arabic", "🇸🇦", "arb_Arab"),
    Language("es", "Spanish", "🇪🇸", "spa_Latn"),
    Language("de", "German", "🇩🇪", "deu_Latn"),
    Language("it", "Italian", "🇮🇹", "ita_Latn"),
    Language("pt", "Portuguese", "🇵🇹", "por_Latn"),
    Language("nl", "Dutch", "🇳🇱", "nld_Latn"),
    Language("ru", "Russian", "🇷🇺", "rus_Cyrl"),
    Language("zh", "Chinese", "🇨🇳", "zho_Hans"),
    Language("ja", "Japanese", "🇯🇵", "jpn_Jpan"),
    Language("ko", "Korean", "🇰🇷", "kor_Hang"),
    Language("tr", "Turkish", "🇹🇷", "tur_Latn"),
    Language("hi", "Hindi", "🇮🇳", "hin_Deva"),
)

_BY_CODE: dict[str, Language] = {lang.code: lang for lang in _LANGUAGES}

# Common aliases people type for the same language.
_ALIASES: dict[str, str] = {
    "english": "en",
    "french": "fr",
    "francais": "fr",
    "français": "fr",
    "arabic": "ar",
    "arabe": "ar",
    "darija": "ar",
    "spanish": "es",
    "espanol": "es",
    "español": "es",
    "german": "de",
    "deutsch": "de",
    "italian": "it",
    "italiano": "it",
}


def normalize(code: str) -> str:
    """Normalise a user-supplied language string to an ISO-639-1 code.

    Accepts names ("French"), aliases ("darija"), region tags ("en-US") and
    raw codes. Unknown values are lower-cased and returned as-is so that codes
    supported by the model but absent from the registry still work.
    """

    if not code:
        return "auto"
    raw = code.strip().lower()
    if raw in _BY_CODE:
        return raw
    if raw in _ALIASES:
        return _ALIASES[raw]
    # Strip region/script suffixes like "en-US" / "zh_hans".
    base = raw.replace("_", "-").split("-", 1)[0]
    if base in _BY_CODE:
        return base
    if base in _ALIASES:
        return _ALIASES[base]
    return base


def get(code: str) -> Language | None:
    return _BY_CODE.get(normalize(code))


def display_name(code: str) -> str:
    lang = get(code)
    return lang.name if lang else code


def to_nllb(code: str) -> str:
    """Map an ISO code to an NLLB-200 code, defaulting to a Latin-script guess."""
    lang = get(code)
    if lang:
        return lang.nllb
    return f"{normalize(code)}_Latn"


def supported() -> list[Language]:
    """All languages in the registry except the synthetic ``auto`` entry."""
    return [lang for lang in _LANGUAGES if lang.code != "auto"]


def is_known(code: str) -> bool:
    return normalize(code) in _BY_CODE
