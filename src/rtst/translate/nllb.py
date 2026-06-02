"""NLLB-200 translation backend (offline, local).

Uses Hugging Face ``transformers`` with Meta's No Language Left Behind models.
The distilled 600M model is a good speed/quality trade-off for live captions
and runs on CPU. Weights are downloaded once and cached locally; nothing is
sent to the cloud.
"""

from __future__ import annotations

from collections.abc import Sequence

from rtst import languages
from rtst.translate.base import Translator


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class NLLBTranslator(Translator):
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        max_new_tokens: int = 128,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        return self._model, self._tokenizer

    def warmup(self) -> None:  # pragma: no cover - requires model download
        self.translate("hello", source="en", target="fr")

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

        import torch

        model, tokenizer = self._ensure_model()
        src_code = languages.to_nllb(source) if source not in ("", "auto") else None
        tgt_code = languages.to_nllb(target)

        if src_code:
            tokenizer.src_lang = src_code

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        bos = _target_bos_id(tokenizer, tgt_code)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=bos,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,  # greedy for lowest latency
            )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _target_bos_id(tokenizer, tgt_code: str) -> int:
    """Resolve the forced BOS token id for the target language across versions."""
    # `lang_code_to_id` is the authoritative map on tokenizers that expose it.
    lang_map = getattr(tokenizer, "lang_code_to_id", None)
    if lang_map and tgt_code in lang_map:
        return lang_map[tgt_code]
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        token_id = convert(tgt_code)
        # convert_tokens_to_ids returns the (non-negative) UNK id for unknown
        # tokens, so an UNK result means the language code is not in the vocab.
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is not None and token_id >= 0 and token_id != unk_id:
            return token_id
    raise ValueError(f"Could not resolve NLLB target token for {tgt_code!r}")
