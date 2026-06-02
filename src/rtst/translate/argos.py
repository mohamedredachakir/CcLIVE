"""Argos Translate backend (fully offline, CPU-only).

Argos Translate ships small OpenNMT models that install per language-pair and
run entirely offline. It is a lighter-weight alternative to NLLB when only a
few language pairs are needed and no GPU is available.
"""

from __future__ import annotations

from collections.abc import Sequence

from rtst import languages
from rtst.translate.base import Translator


class ArgosTranslator(Translator):
    def __init__(self, auto_install: bool = True) -> None:
        self.auto_install = auto_install
        self._ready = False

    def _ensure_package(self, source: str, target: str) -> None:
        import argostranslate.package
        import argostranslate.translate

        installed = {
            (lang.code) for lang in argostranslate.translate.get_installed_languages()
        }
        if source in installed and target in installed:
            return
        if not self.auto_install:
            return
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()
        for pkg in available:
            if pkg.from_code == source and pkg.to_code == target:
                argostranslate.package.install_from_path(pkg.download())
                break

    def translate(
        self,
        text: str,
        *,
        source: str,
        target: str,
        context: Sequence[str] = (),
    ) -> str:
        text = text.strip()
        if not text:
            return ""

        import argostranslate.translate

        src = languages.normalize(source)
        tgt = languages.normalize(target)
        if src in ("", "auto"):
            src = "en"  # Argos needs a concrete source language.
        self._ensure_package(src, tgt)
        return argostranslate.translate.translate(text, src, tgt)
