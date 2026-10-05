from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import TypeVar

from noentenc._batching import check_batch_size, is_blank
from noentenc.language_detection._content import has_linguistic_content
from noentenc.language_detection.labels import (
    NO_LINGUISTIC_CONTENT,
    UNDETERMINED,
    LabelMapper,
    normalize_label,
    normalize_scores,
)


class BaseModel(ABC):
    """A language-identification backend.

    Subclasses implement ``_predict_chunk`` and ``_predict_score_chunk``, and either set
    ``self._mapper`` (see ``_label_mapper``) or override ``labels``. This base class handles
    batching and two rules that hold for every backend, without running the model:

    - empty or whitespace-only texts are ``"und"`` (undetermined);
    - texts without a letter outside URLs, domains and email addresses (digits, emoji,
      punctuation, ``https://foo.com``) are ``"zxx"`` (no linguistic content).

    Labels are ISO 639-3 codes unless ``normalize_labels=False``, which returns the model's own
    codes (with fastText's ``__label__`` prefix removed).

    Scores are backend-specific: higher means more likely, but only some backends return a
    probability distribution. Each backend's docstring says what its scores are.
    """

    # Backends that pad each batch to its longest text (transformers) set this, so batches are
    # formed from texts of similar length. Results are still returned in input order.
    sort_batches_by_length: bool = False
    # Whether `predict_score(text, top_k=None)` scores every label, which restricting the
    # candidate languages needs. Backends that report only their top spans or label don't.
    scores_every_label: bool = True
    _mapper: LabelMapper

    def __init__(
        self,
        model: str | Path,
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
    ) -> None:
        self.model = model
        self.only_local_files = only_local_files
        self.normalize_labels = normalize_labels
        self.collapse_macrolanguages = collapse_macrolanguages

    @property
    def labels(self) -> list[str]:
        """Every label this model can return."""
        return list(self._mapper.labels)

    @abstractmethod
    def _predict_chunk(self, texts: list[str]) -> list[str]:
        """Top-1 label for each (non-empty) text."""

    @abstractmethod
    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        """``{label: score}`` for each (non-empty) text, sorted descending."""

    def predict(self, text: str) -> str:
        label = rule_label(text)
        if label is not None:
            return label
        return self._predict_chunk([text])[0]

    def predict_score(self, text: str, top_k: int | None = None) -> dict[str, float]:
        label = rule_label(text)
        if label is not None:
            return {label: 1.0}
        return self._predict_score_chunk([text], top_k)[0]

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[str]:
        return _run_batched(
            texts,
            batch_size,
            self._predict_chunk,
            lambda label: label,
            self.sort_batches_by_length,
        )

    def predict_batch_score(
        self, texts: list[str], batch_size: int = 32, top_k: int | None = None
    ) -> list[dict[str, float]]:
        return _run_batched(
            texts,
            batch_size,
            lambda chunk: self._predict_score_chunk(chunk, top_k),
            lambda label: {label: 1.0},
            self.sort_batches_by_length,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def _label_mapper(self, native_labels: Sequence[str]) -> LabelMapper:
        return LabelMapper(
            native_labels,
            normalize=self.normalize_labels,
            collapse_macrolanguages=self.collapse_macrolanguages,
        )

    def _label(self, native: str) -> str:
        """``native`` normalised as configured, memoised: backends call this once per text."""
        label = self._labels_seen.get(native)
        if label is None:
            label = self._labels_seen[native] = normalize_label(
                native,
                normalize=self.normalize_labels,
                collapse_macrolanguages=self.collapse_macrolanguages,
            )
        return label

    @cached_property
    def _labels_seen(self) -> dict[str, str]:
        return {}

    def _normalize_scores(self, scores: Mapping[str, float]) -> dict[str, float]:
        return normalize_scores(
            scores,
            normalize=self.normalize_labels,
            collapse_macrolanguages=self.collapse_macrolanguages,
        )


def rule_label(text: str) -> str | None:
    """``"und"`` for blank text, ``"zxx"`` for text without linguistic content, else None."""
    if is_blank(text):
        return UNDETERMINED
    if not has_linguistic_content(text):
        return NO_LINGUISTIC_CONTENT
    return None


_T = TypeVar("_T")


def _run_batched(
    texts: list[str],
    batch_size: int,
    run: Callable[[list[str]], list[_T]],
    ruled: Callable[[str], _T],
    sort_by_length: bool = False,
) -> list[_T]:
    """``run`` over the texts the model must see, in batches; ``ruled(label)`` for the others."""
    check_batch_size(batch_size)
    results: list[_T | None] = [None] * len(texts)
    keep: list[int] = []
    for i, text in enumerate(texts):
        label = rule_label(text)
        if label is None:
            keep.append(i)
        else:
            results[i] = ruled(label)
    if sort_by_length:
        keep.sort(key=lambda i: len(texts[i]))
    for start in range(0, len(keep), batch_size):
        idx = keep[start : start + batch_size]
        for i, result in zip(idx, run([texts[i] for i in idx]), strict=True):
            results[i] = result
    return results  # ty: ignore[invalid-return-type] - every slot is filled above
