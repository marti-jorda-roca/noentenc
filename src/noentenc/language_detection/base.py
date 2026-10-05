from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from tqdm import tqdm

from noentenc._batching import check_batch_size, check_texts
from noentenc._dataframe import column_values, with_column
from noentenc.language_detection._conservative import Detection, Policy
from noentenc.language_detection.models.base import BaseModel
from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.profiles import Profile

if TYPE_CHECKING:
    from collections.abc import Iterable

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

    Labels are ISO 639-3 codes. Empty text is ``"und"`` (undetermined), and text without
    letters outside URLs and email addresses (digits, emoji, punctuation, a bare link) is
    ``"zxx"`` (no linguistic content); neither runs the model. With ``with_score``, only the
    ``top_k`` highest-scoring languages are returned (``None`` for all).

    By default every other text gets the model's top label. To abstain instead, returning
    ``"und"``, set any of:

    - ``min_letters``: fewer letters than this (URLs and emails don't count) is
      insufficient text. The model doesn't run.
    - ``min_score``: a top score below this is low confidence.
    - ``min_margin``: a top score less than this above the second is ambiguous.

    Scores are backend-specific and not calibrated, so thresholds don't carry over from one
    backend to another; docs/benchmarks.md lists tested settings per backend.

    ``candidates`` restricts the answer to the given languages (ISO 639-3 or 639-1 codes):
    the best-scoring candidate wins, with its own score, which is not renormalised. It needs
    a backend that scores every language; CLD3 and heliport don't.

    ``detailed=True`` returns a ``Detection`` with the status that explains the label.
    """

    def __init__(
        self,
        model: BaseModel | Profile | str | None = None,
        *,
        min_score: float | None = None,
        min_margin: float | None = None,
        min_letters: int | None = None,
        candidates: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(model, BaseModel):
            model = _profile_model(Profile(model or Profile.SPEED))
        self.model = model
        self.policy = Policy(
            model,
            min_score=min_score,
            min_margin=min_margin,
            min_letters=min_letters,
            candidates=candidates,
        )

    @overload
    def detect(
        self,
        text: str,
        with_score: Literal[False] = False,
        top_k: int | None = DEFAULT_TOP_K,
        *,
        detailed: Literal[False] = False,
    ) -> str: ...

    @overload
    def detect(
        self,
        text: str,
        with_score: Literal[True],
        top_k: int | None = DEFAULT_TOP_K,
        *,
        detailed: Literal[False] = False,
    ) -> dict[str, float]: ...

    @overload
    def detect(
        self,
        text: str,
        with_score: Literal[False] = False,
        top_k: int | None = DEFAULT_TOP_K,
        *,
        detailed: Literal[True],
    ) -> Detection: ...

    def detect(
        self,
        text: str,
        with_score: bool = False,
        top_k: int | None = DEFAULT_TOP_K,
        *,
        detailed: bool = False,
    ) -> str | dict[str, float] | Detection:
        """The language of ``text``, its ``top_k`` highest-scoring languages, or a ``Detection``."""
        (result,) = self._detect([text], 32, with_score, top_k, detailed)
        return result

    @overload
    def detect_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        with_score: Literal[False] = False,
        top_k: int | None = DEFAULT_TOP_K,
        *,
        detailed: Literal[False] = False,
    ) -> list[str]: ...

    @overload
    def detect_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        *,
        with_score: Literal[True],
        top_k: int | None = DEFAULT_TOP_K,
        detailed: Literal[False] = False,
    ) -> list[dict[str, float]]: ...

    @overload
    def detect_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        with_score: Literal[False] = False,
        top_k: int | None = DEFAULT_TOP_K,
        *,
        detailed: Literal[True],
    ) -> list[Detection]: ...

    def detect_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        with_score: bool = False,
        top_k: int | None = DEFAULT_TOP_K,
        *,
        detailed: bool = False,
    ) -> list[str] | list[dict[str, float]] | list[Detection]:
        """The language of each of ``texts``, like ``detect``."""
        check_texts(texts)
        return self._detect(texts, batch_size, with_score, top_k, detailed)

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
        *,
        status_column: str | None = None,
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
        *,
        status_column: str | None = None,
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
        *,
        status_column: str | None = None,
    ) -> pl.DataFrame | pd.DataFrame:
        """Return ``dataset`` with a ``result_column`` holding the language of ``target_column``.

        With ``with_score=True`` each cell is a list of ``{"language", "score"}`` records
        sorted by score, which gives polars a fixed schema. Null texts get ``"und"``.
        ``status_column`` adds each row's ``DetectionStatus`` (``"empty"`` for null texts).
        With ``show_progress`` a tqdm bar tracks the rows inferred so far.
        """
        check_batch_size(batch_size)
        _check_top_k(top_k)
        texts = [
            text if isinstance(text, str) else ""
            for text in column_values(dataset, target_column)
        ]
        results: list[Any] = []
        statuses: list[str] = []
        with tqdm(
            total=len(texts), desc="Detecting language", disable=not show_progress
        ) as progress:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                if status_column is not None:
                    detections = self._detect_detailed(chunk, batch_size)
                    statuses += [str(d.status) for d in detections]
                if with_score or status_column is None:
                    results += self._detect(
                        chunk, batch_size, with_score, top_k, detailed=False
                    )
                else:
                    results += [d.language for d in detections]
                progress.update(len(chunk))
        if with_score:
            results = [
                [{"language": lang, "score": score} for lang, score in scores.items()]
                for scores in results
            ]
        dataset = with_column(dataset, result_column, results)
        if status_column is not None:
            dataset = with_column(dataset, status_column, statuses, strings=True)
        return dataset

    def _detect(
        self,
        texts: list[str],
        batch_size: int,
        with_score: bool,
        top_k: int | None,
        detailed: bool,
    ) -> Any:  # noqa: ANN401 - the public overloads type each combination
        check_batch_size(batch_size)
        _check_top_k(top_k)
        if with_score and detailed:
            raise ValueError("pass with_score or detailed, not both")
        if detailed:
            return self._detect_detailed(texts, batch_size)
        if with_score:
            return self._scores(texts, batch_size, top_k)
        if self.policy.active:
            return [d.language for d in self._detect_detailed(texts, batch_size)]
        return self.model.predict_batch(texts, batch_size=batch_size)

    def _scores(
        self, texts: list[str], batch_size: int, top_k: int | None
    ) -> list[dict[str, float]]:
        if self.policy.candidates is None:
            return self.model.predict_batch_score(
                texts, batch_size=batch_size, top_k=top_k
            )
        restricted = [
            self.policy.restrict(scores)
            for scores in self.model.predict_batch_score(
                texts, batch_size=batch_size, top_k=None
            )
        ]
        return [dict(list(scores.items())[:top_k]) for scores in restricted]

    def _detect_detailed(self, texts: list[str], batch_size: int) -> list[Detection]:
        results = [self.policy.before_model(text) for text in texts]
        pending = [i for i, result in enumerate(results) if result is None]
        scores = self.model.predict_batch_score(
            [texts[i] for i in pending], batch_size=batch_size, top_k=self.policy.top_k
        )
        for i, text_scores in zip(pending, scores, strict=True):
            results[i] = self.policy.decide(text_scores)
        return results  # ty: ignore[invalid-return-type] - every slot is filled above


def _check_top_k(top_k: object) -> None:
    if top_k is None:
        return
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an int or None, got {type(top_k).__name__}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k}")
