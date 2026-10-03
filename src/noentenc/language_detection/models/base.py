from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from noentenc.language_detection.labels import UNDETERMINED


class BaseModel(ABC):
    """A language-identification backend.

    Subclasses implement ``_predict_chunk`` and ``_predict_score_chunk``. This base class handles
    batching and the empty-text rule: empty or whitespace-only texts are ``"und"`` for every backend
    and never reach the model.

    Labels are ISO 639-3 codes unless ``normalize_labels=False``, which returns the model's own
    codes (with fastText's ``__label__`` prefix removed).

    Scores are backend-specific: higher means more likely, but only some backends return a
    probability distribution. Each backend's docstring says what its scores are.
    """

    # Backends that pad each batch to its longest text (transformers) set this, so batches are
    # formed from texts of similar length. Results are still returned in input order.
    sort_batches_by_length: bool = False

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
    @abstractmethod
    def labels(self) -> list[str]:
        """Every label this model can return."""

    @abstractmethod
    def _predict_chunk(self, texts: list[str]) -> list[str]:
        """Top-1 label for each (non-empty) text."""

    @abstractmethod
    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        """``{label: score}`` for each (non-empty) text, sorted descending."""

    def predict(self, text: str) -> str:
        return self.predict_batch([text])[0]

    def predict_score(self, text: str, top_k: int | None = None) -> dict[str, float]:
        return self.predict_batch_score([text], top_k=top_k)[0]

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[str]:
        return _run_batched(
            texts,
            batch_size,
            self._predict_chunk,
            lambda: UNDETERMINED,
            self.sort_batches_by_length,
        )

    def predict_batch_score(
        self, texts: list[str], batch_size: int = 32, top_k: int | None = None
    ) -> list[dict[str, float]]:
        return _run_batched(
            texts,
            batch_size,
            lambda chunk: self._predict_score_chunk(chunk, top_k),
            lambda: {UNDETERMINED: 1.0},
            self.sort_batches_by_length,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


def _run_batched[T](
    texts: list[str],
    batch_size: int,
    run: Callable[[list[str]], list[T]],
    empty: Callable[[], T],
    sort_by_length: bool = False,
) -> list[T]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    keep = [i for i, text in enumerate(texts) if text and not text.isspace()]
    if sort_by_length:
        keep.sort(key=lambda i: len(texts[i]))
    results: list[T | None] = [None] * len(texts)
    for start in range(0, len(keep), batch_size):
        idx = keep[start : start + batch_size]
        for i, result in zip(idx, run([texts[i] for i in idx]), strict=True):
            results[i] = result
    return [empty() if result is None else result for result in results]
