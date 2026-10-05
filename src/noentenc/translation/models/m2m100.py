from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

from noentenc.languages import Language, LanguageSchema
from noentenc.translation.models._seq2seq import (
    Precision,
    Seq2SeqModel,
    tokenizer_class,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tokenizers import Tokenizer

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
    """Facebook M2M100-418M: one MIT-licensed model for any pair of 100 languages.

    About as accurate as SMaLL-100 overall: better into Chinese and Japanese, worse on
    low-resource languages such as Swahili and Tamil, and about 4x slower.
    """

    schema = LanguageSchema(source=_LANGUAGES, target=_LANGUAGES)
    default_model: ClassVar[str] = "Xenova/m2m100_418M"
    default_revision: ClassVar[str] = "9c374f0b7aca709787cea97b047bfbbd1559d177"
    # The q4 export scores up to 26 chrF++ lower on FLORES-200 and is 2x larger.
    default_precision: ClassVar[Precision] = Precision.INT8

    def _load_tokenizer(self, files: dict[str, Path]) -> Tokenizer:
        try:
            return super()._load_tokenizer(files)
        except Exception:  # tokenizers raises a bare Exception on invalid files
            spec = json.loads(files["tokenizer.json"].read_text())
            return tokenizer_class().from_str(json.dumps(_drop_invalid_merges(spec)))

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
