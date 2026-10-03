"""Normalise every backend's language labels to ISO 639-3."""

from collections.abc import Mapping, Sequence
from functools import cache

import numpy as np

from noentenc.language_detection._iso639 import (
    INDIVIDUAL_TO_MACRO,
    PART1_TO_PART3,
    RETIRED_TO_CURRENT,
    VALID_PART3,
)

UNDETERMINED = "und"
NO_LINGUISTIC_CONTENT = "zxx"
FASTTEXT_LABEL_PREFIX = "__label__"
_PART1_LENGTH = 2

# Deprecated ISO 639-1 codes and Wikipedia-style codes (used by lid.176 and CLD3) that the
# SIL tables don't resolve, or resolve to something other than what the model means.
LABEL_OVERRIDES: dict[str, str] = {
    "als": "gsw",  # Alemannic Wikipedia, not Tosk Albanian
    "bh": "bho",  # Bhojpuri Wikipedia
    "sh": "hbs",
    "iw": "heb",
    "in": "ind",
    "ji": "yid",
    "jw": "jav",
    "mo": "ron",
    "nah": "nhe",  # Nahuatl Wikipedia; `nah` is a collective (ISO 639-5) code
    "eml": "egl",  # retired by a split; Emilian is the larger part of the Wikipedia
}


def to_iso639_3(label: str, *, collapse_macrolanguages: bool = False) -> str:
    """Map a native model label (``__label__en``, ``eng_Latn``, ``zh-Latn``, ``EN``...) to ISO 639-3."""
    code = (
        label.removeprefix(FASTTEXT_LABEL_PREFIX)
        .replace("-", "_")
        .split("_", 1)[0]
        .lower()
    )
    if code in LABEL_OVERRIDES:
        code = LABEL_OVERRIDES[code]
    elif len(code) == _PART1_LENGTH:
        code = PART1_TO_PART3.get(code, code)
    code = RETIRED_TO_CURRENT.get(code, code)
    if collapse_macrolanguages:
        code = INDIVIDUAL_TO_MACRO.get(code, code)
    return code


def normalize_label(
    label: str, *, normalize: bool = True, collapse_macrolanguages: bool = False
) -> str:
    """``to_iso639_3(label)``, or with ``normalize=False`` the label minus fastText's prefix."""
    if not normalize:
        return label.removeprefix(FASTTEXT_LABEL_PREFIX)
    return to_iso639_3(label, collapse_macrolanguages=collapse_macrolanguages)


@cache
def valid_iso639_3_codes() -> frozenset[str]:
    """Every current ISO 639-3 code, plus ``und`` and ``zxx``."""
    return frozenset(VALID_PART3.split()) | {UNDETERMINED, NO_LINGUISTIC_CONTENT}


class LabelMapper:
    """Map a model's native label axis onto unified ISO 639-3 codes.

    When several native labels collapse to the same code (``zho_Hans`` and ``zho_Hant``),
    their scores are summed with a single ``np.add.reduceat`` over a permuted axis.
    """

    def __init__(
        self,
        native_labels: Sequence[str],
        *,
        normalize: bool = True,
        collapse_macrolanguages: bool = False,
    ) -> None:
        self.native_labels = list(native_labels)
        mapped = [
            normalize_label(
                lab,
                normalize=normalize,
                collapse_macrolanguages=collapse_macrolanguages,
            )
            for lab in native_labels
        ]
        # One-to-one mappings keep the native label order, so scores need no reshuffling.
        self.identity = len(set(mapped)) == len(mapped)
        self.labels: list[str] = mapped if self.identity else sorted(set(mapped))
        self._label_array = np.array(self.labels, dtype=object)
        if not self.identity:
            index = {lab: i for i, lab in enumerate(self.labels)}
            target = np.array([index[lab] for lab in mapped], dtype=np.intp)
            self._order = np.argsort(target, kind="stable")
            self._starts = np.searchsorted(
                target[self._order], np.arange(len(self.labels))
            )

    def reduce(self, scores: np.ndarray) -> np.ndarray:
        """Sum the native-label columns of ``scores`` (shape ``(batch, n_native)``) per unified label."""
        if self.identity:
            return scores
        return np.add.reduceat(scores[:, self._order], self._starts, axis=1)

    def top_label(self, unified_index: np.ndarray) -> list[str]:
        return self._label_array[unified_index].tolist()

    def to_dicts(self, scores: np.ndarray, top_k: int | None) -> list[dict[str, float]]:
        """Convert unified scores to ``{label: score}`` dicts, sorted descending, keeping ``top_k``."""
        n_labels = scores.shape[1]
        k = n_labels if top_k is None else max(1, min(top_k, n_labels))
        if k < n_labels:
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1, kind="stable")
            top = np.take_along_axis(top, order, axis=1)
        else:
            top = np.argsort(-scores, axis=1, kind="stable")
        top_scores = np.take_along_axis(scores, top, axis=1).tolist()
        names = self._label_array[top].tolist()
        return [
            dict(zip(row_names, row_scores, strict=True))
            for row_names, row_scores in zip(names, top_scores, strict=True)
        ]


def normalize_scores(
    scores: Mapping[str, float],
    *,
    normalize: bool = True,
    collapse_macrolanguages: bool = False,
) -> dict[str, float]:
    """Normalise the keys of a ``{native label: score}`` mapping, summing scores of labels that collapse."""
    out: dict[str, float] = {}
    for label, score in scores.items():
        key = normalize_label(
            label, normalize=normalize, collapse_macrolanguages=collapse_macrolanguages
        )
        out[key] = out.get(key, 0.0) + score
    return out
