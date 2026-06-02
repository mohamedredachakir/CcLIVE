"""Translation backends."""

from rtst.translate.base import Translator
from rtst.translate.identity import IdentityTranslator

__all__ = ["Translator", "IdentityTranslator", "create_translator"]


def create_translator(config) -> Translator:  # noqa: ANN001
    """Instantiate a translator from an :class:`rtst.config.TranslationConfig`."""
    backend = (config.backend or "nllb").lower()
    if backend == "identity":
        return IdentityTranslator()
    if backend == "nllb":
        from rtst.translate.nllb import NLLBTranslator

        return NLLBTranslator(model_name=config.model, device=config.device)
    if backend == "argos":
        from rtst.translate.argos import ArgosTranslator

        return ArgosTranslator()
    raise ValueError(f"Unknown translation backend: {config.backend!r}")
