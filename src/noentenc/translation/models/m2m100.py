import json
from pathlib import Path
from typing import ClassVar

from tokenizers import Tokenizer

from noentenc.languages import Language, LanguageSchema
from noentenc.translation.models._seq2seq import Precision, Seq2SeqModel

# M2M100 language code for each supported language (token `__{code}__`).
M2M100_CODES: dict[Language, str] = {
    **{
        Language(code): code
        for code in (  # noqa: SIM905 - a word list reads better than 100 quoted strings
            "af am ar ast az ba be bg bn br bs ca ceb cs cy da de el en es et fa ff fi fr "
            "fy ga gd gl gu ha he hi hr ht hu hy id ig ilo is it ja jv ka kk km kn ko lb lg "
            "ln lo lt lv mg mk ml mn mr ms my ne nl no oc or pa pl ps pt ro ru sd si sk sl "
            "so sq sr ss su sv sw ta th tl tn tr uk ur uz vi wo xh yi yo zh zu"
        ).split()
    },
    Language.NORTHERN_SOTHO: "ns",
}

_LANGUAGES = frozenset(M2M100_CODES)


class M2M100Model(Seq2SeqModel):
    """Facebook M2M100-418M: one MIT-licensed model for any pair of 100 languages."""

    schema = LanguageSchema(source=_LANGUAGES, target=_LANGUAGES)
    default_model: ClassVar[str] = "Xenova/m2m100_418M"
    default_precision: ClassVar[Precision] = Precision.Q4
    onnx_files: ClassVar[dict[Precision, tuple[str, str]]] = {
        Precision.FP32: ("onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"),
        Precision.INT8: (
            "onnx/encoder_model_quantized.onnx",
            "onnx/decoder_model_merged_quantized.onnx",
        ),
        Precision.Q4: (
            "onnx/encoder_model_q4.onnx",
            "onnx/decoder_model_merged_q4.onnx",
        ),
    }

    def _load_tokenizer(self, files: dict[str, Path]) -> Tokenizer:
        try:
            return super()._load_tokenizer(files)
        except Exception:  # tokenizers raises a bare Exception on invalid files
            spec = json.loads(files["tokenizer.json"].read_text())
            return Tokenizer.from_str(json.dumps(_drop_invalid_merges(spec)))

    def _language_id(self, language: Language) -> int:
        return self._token_id(f"__{M2M100_CODES[language]}__")

    def _frame(
        self, ids: list[int], source: Language | None, target: Language
    ) -> list[int]:
        if source is None:  # already rejected by `schema`; narrows the type
            raise ValueError("M2M100 requires a source language")
        return [self._language_id(source), *ids, self.config["eos_token_id"]]

    def _forced_first_id(self, target: Language) -> int | None:
        return self._language_id(target)


def _drop_invalid_merges(spec: dict) -> dict:
    """Remove BPE merges whose parts or result are missing from the vocab.

    Xenova's M2M100 tokenizer.json has ~1k such merges (left over from the
    SentencePiece conversion); `tokenizers` refuses to load them. They can never
    produce an in-vocab token, so dropping them doesn't change tokenization.
    """
    vocab = spec["model"]["vocab"]

    def valid(merge: str | list[str]) -> bool:
        left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
        return left in vocab and right in vocab and left + right in vocab

    spec["model"]["merges"] = [m for m in spec["model"]["merges"] if valid(m)]
    return spec
