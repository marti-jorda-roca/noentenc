from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from tqdm import tqdm

from noentenc._batching import check_batch_size, check_texts
from noentenc._dataframe import column_values, with_column
from noentenc.language_detection.models.base import BaseModel
from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.profiles import Profile

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

DEFAULT_TOP_K = 5


def _profile_model(profile: Profile) -> BaseModel:
    match profile:
        case Profile.SPEED:
            return FastTextModel()
        case Profile.BALANCE:
            return FastTextModel("openlid-v3")
        case Profile.QUALITY:
            return FastTextModel("glotlid")


class LanguageDetector:
    """Detect the language of texts with any ``BaseModel`` backend.

    ``model`` is a backend, or a ``Profile`` that picks one (``None`` is ``"speed"``):

    - ``speed``: fastText ``lid176`` (quantized, 0.9 MB, 176 languages).
    - ``balance``: fastText ``openlid-v3`` (1.2 GB, 195 languages, GPL-3.0).
    - ``quality``: fastText ``glotlid`` (1.7 GB, 2102 labels).

    Labels are ISO 639-3 codes; ``"und"`` means undetermined. With ``with_score``, only the
    ``top_k`` highest-scoring languages are returned (``None`` for all).
    """

    def __init__(self, model: BaseModel | Profile | str | None = None) -> None:
        if not isinstance(model, BaseModel):
            model = _profile_model(Profile(model or Profile.SPEED))
        self.model = model

    @overload
    def detect(
        self,
        text: str,
        with_score: Literal[False] = False,
        top_k: int | None = DEFAULT_TOP_K,
    ) -> str: ...

    @overload
    def detect(
        self, text: str, with_score: Literal[True], top_k: int | None = DEFAULT_TOP_K
    ) -> dict[str, float]: ...

    def detect(
        self, text: str, with_score: bool = False, top_k: int | None = DEFAULT_TOP_K
    ) -> str | dict[str, float]:
        """The language of ``text``, or with ``with_score`` its ``top_k`` highest-scoring languages."""
        _check_top_k(top_k)
        if with_score:
            return self.model.predict_score(text, top_k=top_k)
        return self.model.predict(text)

    @overload
    def detect_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        with_score: Literal[False] = False,
        top_k: int | None = DEFAULT_TOP_K,
    ) -> list[str]: ...

    @overload
    def detect_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        *,
        with_score: Literal[True],
        top_k: int | None = DEFAULT_TOP_K,
    ) -> list[dict[str, float]]: ...

    def detect_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        with_score: bool = False,
        top_k: int | None = DEFAULT_TOP_K,
    ) -> list[str] | list[dict[str, float]]:
        """The language of each of ``texts``, like ``detect``."""
        check_texts(texts)
        check_batch_size(batch_size)
        _check_top_k(top_k)
        if with_score:
            return self.model.predict_batch_score(
                texts, batch_size=batch_size, top_k=top_k
            )
        return self.model.predict_batch(texts, batch_size=batch_size)

    @overload
    def detect_dataset(
        self,
        dataset: pl.DataFrame,
        target_column: str,
        result_column: str,
        batch_size: int = 32,
        with_score: bool = False,
        top_k: int | None = DEFAULT_TOP_K,
        show_progress: bool = True,
    ) -> pl.DataFrame: ...

    @overload
    def detect_dataset(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        result_column: str,
        batch_size: int = 32,
        with_score: bool = False,
        top_k: int | None = DEFAULT_TOP_K,
        show_progress: bool = True,
    ) -> pd.DataFrame: ...

    def detect_dataset(
        self,
        dataset: pl.DataFrame | pd.DataFrame,
        target_column: str,
        result_column: str,
        batch_size: int = 32,
        with_score: bool = False,
        top_k: int | None = DEFAULT_TOP_K,
        show_progress: bool = True,
    ) -> pl.DataFrame | pd.DataFrame:
        """Return ``dataset`` with a ``result_column`` holding the language of ``target_column``.

        With ``with_score=True`` each cell is a list of ``{"language", "score"}`` records
        sorted by score, which gives polars a fixed schema. Null texts get ``"und"``.
        With ``show_progress`` a tqdm bar tracks the rows inferred so far.
        """
        check_batch_size(batch_size)
        _check_top_k(top_k)
        texts = [
            text if isinstance(text, str) else ""
            for text in column_values(dataset, target_column)
        ]
        results: list[Any] = []
        with tqdm(
            total=len(texts), desc="Detecting language", disable=not show_progress
        ) as progress:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                results.extend(
                    self.detect_batch(
                        chunk, batch_size=batch_size, with_score=with_score, top_k=top_k
                    )
                )
                progress.update(len(chunk))
        if with_score:
            results = [
                [{"language": lang, "score": score} for lang, score in scores.items()]
                for scores in results
            ]
        return with_column(dataset, result_column, results)


def _check_top_k(top_k: object) -> None:
    if top_k is None:
        return
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an int or None, got {type(top_k).__name__}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k}")
