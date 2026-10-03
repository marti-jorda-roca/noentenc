"""lingua-py backend: n-gram models, the most accurate on short texts (75 languages)."""

from collections.abc import Iterable
from typing import Any

from noentenc.language_detection._optional import require
from noentenc.language_detection.labels import UNDETERMINED
from noentenc.language_detection.models.base import BaseModel


class LinguaModel(BaseModel):
    """Wraps ``lingua-language-detector`` (``uv add 'noentenc[lingua]'``).

    ``languages`` restricts the candidates to the given ISO 639-3 codes, which is faster and more
    accurate when you know the possible languages. ``low_accuracy`` uses only trigrams: much
    faster and lighter, but noticeably worse on texts under ~120 characters. ``preload`` loads
    every language model up front instead of lazily on first use.
    """

    def __init__(
        self,
        model: str = "lingua",
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
        languages: Iterable[str] | None = None,
        low_accuracy: bool = False,
        preload: bool = False,
        minimum_relative_distance: float = 0.0,
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        lingua = require("lingua", "lingua")
        if languages is None:
            builder = lingua.LanguageDetectorBuilder.from_all_languages()
            candidates = list(lingua.Language.all())
        else:
            iso = [lingua.IsoCode639_3.from_str(code) for code in languages]
            builder = lingua.LanguageDetectorBuilder.from_iso_codes_639_3(*iso)
            candidates = [lingua.Language.from_iso_code_639_3(code) for code in iso]
        if low_accuracy:
            builder = builder.with_low_accuracy_mode()
        if preload:
            builder = builder.with_preloaded_language_models()
        if minimum_relative_distance:
            builder = builder.with_minimum_relative_distance(minimum_relative_distance)
        self._detector = builder.build()
        self._mapper = self._label_mapper([_code(lang) for lang in candidates])

    def _language_label(self, language: Any) -> str:  # noqa: ANN401 - lingua.Language, imported lazily
        return UNDETERMINED if language is None else self._label(_code(language))

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        if len(texts) == 1:
            return [self._language_label(self._detector.detect_language_of(texts[0]))]
        return [
            self._language_label(lang)
            for lang in self._detector.detect_languages_in_parallel_of(texts)
        ]

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        if len(texts) == 1:
            batches = [self._detector.compute_language_confidence_values(texts[0])]
        else:
            batches = self._detector.compute_language_confidence_values_in_parallel(
                texts
            )
        results = []
        for values in batches:
            scores: dict[str, float] = {}
            for value in values:  # already sorted descending
                if value.value <= 0.0:
                    break
                key = self._language_label(value.language)
                scores[key] = scores.get(key, 0.0) + value.value
            results.append(
                dict(list(scores.items())[:top_k]) if scores else {UNDETERMINED: 1.0}
            )
        return results


def _code(language: Any) -> str:  # noqa: ANN401 - lingua.Language
    return language.iso_code_639_3.name.lower()
