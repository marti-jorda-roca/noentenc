"""langid.py backend: a pure-Python/numpy naive Bayes classifier (97 languages). Benchmark baseline."""

from collections.abc import Iterable

from noentenc.language_detection._optional import require
from noentenc.language_detection.models.base import BaseModel


class LangidModel(BaseModel):
    """Wraps ``langid`` (``uv add 'noentenc[langid]'``), with normalised probabilities.

    ``languages`` restricts the candidates (ISO 639-1 codes, as langid uses).
    """

    def __init__(
        self,
        model: str = "langid",
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
        languages: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        langid = require("langid.langid", "langid")
        self._identifier = langid.LanguageIdentifier.from_modelstring(
            langid.model, norm_probs=True
        )
        if languages is not None:
            self._identifier.set_languages(list(languages))
        self._mapper = self._label_mapper(self._identifier.nb_classes)

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        return [self._label(self._identifier.classify(text)[0]) for text in texts]

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        return [
            self._normalize_scores(
                {code: float(p) for code, p in self._identifier.rank(text)[:top_k]}
            )
            for text in texts
        ]
