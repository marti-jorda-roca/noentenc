"""Deciding when a detection is trustworthy: abstention thresholds and candidate languages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from noentenc.language_detection._content import count_letters
from noentenc.language_detection.labels import (
    NO_LINGUISTIC_CONTENT,
    UNDETERMINED,
    normalize_label,
)
from noentenc.language_detection.models.base import BaseModel, rule_label


class DetectionStatus(StrEnum):
    """Why a `Detection` has its label."""

    # The model's top label, which passed every configured threshold.
    DETECTED = "detected"
    # Empty or whitespace-only text: "und", without running the model.
    EMPTY = "empty"
    # Digits, emoji, punctuation, URLs or email addresses only: "zxx", without the model.
    NO_LINGUISTIC_CONTENT = "no_linguistic_content"
    # Fewer letters than `min_letters`: "und", without running the model.
    INSUFFICIENT_TEXT = "insufficient_text"
    # The top score is below `min_score`, or no candidate language scored: "und".
    LOW_CONFIDENCE = "low_confidence"
    # The top two scores are closer than `min_margin`: "und".
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class Detection:
    """A detected language, why it was chosen, and the model's best guess when abstaining."""

    # ISO 639-3 code, "und" (undetermined) or "zxx" (no linguistic content).
    language: str
    status: DetectionStatus = DetectionStatus.DETECTED
    # The model's top label among the candidates and its score, also when `language` is
    # "und" because a threshold wasn't met. None when the model didn't run.
    top_language: str | None = None
    score: float | None = None


_RULE_STATUS = {
    UNDETERMINED: DetectionStatus.EMPTY,
    NO_LINGUISTIC_CONTENT: DetectionStatus.NO_LINGUISTIC_CONTENT,
}


class Policy:
    """When `LanguageDetector` abstains, and which languages it may return."""

    def __init__(
        self,
        model: BaseModel,
        *,
        min_score: float | None = None,
        min_margin: float | None = None,
        min_letters: int | None = None,
        candidates: Iterable[str] | None = None,
    ) -> None:
        self.min_score = _check_threshold("min_score", min_score)
        self.min_margin = _check_threshold("min_margin", min_margin)
        self.min_letters = _check_min_letters(min_letters)
        self.candidates = _check_candidates(model, candidates)

    @property
    def active(self) -> bool:
        """Whether any setting differs from returning the model's top label."""
        return any(
            value is not None
            for value in (
                self.min_score,
                self.min_margin,
                self.min_letters,
                self.candidates,
            )
        )

    @property
    def top_k(self) -> int | None:
        """Scores to request from the model: all of them to restrict candidates, else two."""
        return None if self.candidates is not None else 2

    def before_model(self, text: str) -> Detection | None:
        """The detection of `text` when it needs no model, else None."""
        label = rule_label(text)
        if label is not None:
            return Detection(label, _RULE_STATUS[label])
        if self.min_letters is not None and count_letters(text) < self.min_letters:
            return Detection(UNDETERMINED, DetectionStatus.INSUFFICIENT_TEXT)
        return None

    def restrict(self, scores: dict[str, float]) -> dict[str, float]:
        """`scores` (sorted descending) without the languages that aren't candidates."""
        if self.candidates is None:
            return scores
        return {label: s for label, s in scores.items() if label in self.candidates}

    def decide(self, scores: dict[str, float]) -> Detection:
        """The detection for the model's `scores` of a text, sorted descending."""
        ranked = list(self.restrict(scores).items())
        if not ranked or ranked[0][0] == UNDETERMINED:
            return Detection(UNDETERMINED, DetectionStatus.LOW_CONFIDENCE)
        top, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if self.min_score is not None and score < self.min_score:
            status = DetectionStatus.LOW_CONFIDENCE
        elif self.min_margin is not None and score - second < self.min_margin:
            status = DetectionStatus.AMBIGUOUS
        else:
            return Detection(top, DetectionStatus.DETECTED, top, score)
        return Detection(UNDETERMINED, status, top, score)


def _check_threshold(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number or None, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return float(value)


def _check_min_letters(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"min_letters must be an int or None, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"min_letters must be >= 1, got {value}")
    return value


def _check_candidates(
    model: BaseModel, candidates: Iterable[str] | None
) -> frozenset[str] | None:
    if candidates is None:
        return None
    if isinstance(candidates, str):
        raise TypeError("candidates must be a list of language codes, not a string")
    if not model.scores_every_label:
        raise ValueError(
            f"{type(model).__name__} doesn't score every language, so it can't be "
            "restricted to candidates; use a fastText, ONNX, langid or lingua model"
        )
    codes = frozenset(
        normalize_label(
            code,
            normalize=model.normalize_labels,
            collapse_macrolanguages=model.collapse_macrolanguages,
        )
        for code in candidates
    )
    if not codes:
        raise ValueError("candidates must name at least one language")
    unknown = sorted(codes - set(model.labels))
    if unknown:
        raise ValueError(f"{type(model).__name__} never returns {unknown}")
    return codes
