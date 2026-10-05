"""Nonlinguistic input, abstention thresholds and candidate languages."""

from pathlib import Path

import polars as pl
import pytest

from noentenc.language_detection import Detection, DetectionStatus, LanguageDetector
from noentenc.language_detection._content import count_letters, has_linguistic_content
from noentenc.language_detection._download import CACHE_ENV
from noentenc.language_detection.labels import LabelMapper
from noentenc.language_detection.models.base import BaseModel
from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.language_detection.models.langid_model import LANGID_SDIST, LangidModel
from tests.unit.language_detection.helpers import FASTTEXT_FIXTURES
from tests.unit.language_detection.test_langid import _model_pickle, _write_sdist
from tests.unit.language_detection.test_onnx_classifier import _fake_model

TINY = FASTTEXT_FIXTURES / "tiny-softmax.bin"


class ScriptedModel(BaseModel):
    """Returns fixed scores and records every text that reaches it."""

    def __init__(
        self, scores: dict[str, float], *, scores_every_label: bool = True
    ) -> None:
        super().__init__("scripted")
        self.scores = scores
        self.scores_every_label = scores_every_label
        self._mapper = LabelMapper(list(scores))
        self.seen: list[str] = []

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        self.seen += texts
        return [next(iter(self.scores)) for _ in texts]

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        self.seen += texts
        return [dict(list(self.scores.items())[:top_k]) for _ in texts]


NONLINGUISTIC = [
    "123 456",
    "😀😀😀",
    "!!! ???",
    "https://foo.com",
    "HTTPS://Example.com/reset?id=42",
    "www.example.org",
    "support@acme.io",
    "foo.com/path",
    "(https://x.io).",
    "+34 600 123 456",
    "$49.99 #1234",
]


@pytest.mark.parametrize("text", NONLINGUISTIC)
def test_nonlinguistic_text(text: str) -> None:
    assert not has_linguistic_content(text)
    assert count_letters(text) == 0


@pytest.mark.parametrize(
    ("text", "letters"),
    [
        ("lol", 3),
        ("Visit https://foo.com", 5),
        ("Mr.Smith", 7),
        ("e.g.", 2),
        ("你好", 2),
        ("mail support@acme.io now", 7),
    ],
)
def test_linguistic_text(text: str, letters: int) -> None:
    assert has_linguistic_content(text)
    assert count_letters(text) == letters


def test_nonlinguistic_text_is_zxx_without_inference() -> None:
    model = ScriptedModel({"eng": 0.7, "spa": 0.3})
    detector = LanguageDetector(model)
    texts = [*NONLINGUISTIC, "", "hello"]
    labels = detector.detect_batch(texts)
    assert labels == ["zxx"] * len(NONLINGUISTIC) + ["und", "eng"]
    assert detector.detect("https://foo.com") == "zxx"
    assert detector.detect("👍", with_score=True) == {"zxx": 1.0}
    assert model.predict_batch_score(["123"]) == [{"zxx": 1.0}]
    assert model.seen == ["hello"]


def test_default_returns_the_top_label_for_short_text() -> None:
    model = ScriptedModel({"eng": 0.12, "spa": 0.11})
    detector = LanguageDetector(model)
    assert detector.detect("lol") == "eng"
    assert detector.detect("lol", detailed=True) == Detection(
        "eng", DetectionStatus.DETECTED, "eng", 0.12
    )


def test_detailed_explains_rule_labels() -> None:
    detector = LanguageDetector(ScriptedModel({"eng": 1.0}))
    assert detector.detect_batch(["", "123"], detailed=True) == [
        Detection("und", DetectionStatus.EMPTY),
        Detection("zxx", DetectionStatus.NO_LINGUISTIC_CONTENT),
    ]


def test_min_letters_abstains_without_inference() -> None:
    model = ScriptedModel({"eng": 0.9, "spa": 0.1})
    detector = LanguageDetector(model, min_letters=4)
    assert detector.detect("lol") == "und"
    assert detector.detect("lol https://example.com", detailed=True) == Detection(
        "und", DetectionStatus.INSUFFICIENT_TEXT
    )
    assert detector.detect("hello") == "eng"
    assert model.seen == ["hello"]


def test_min_score_abstains_on_low_confidence() -> None:
    detector = LanguageDetector(ScriptedModel({"eng": 0.12, "spa": 0.1}), min_score=0.5)
    assert detector.detect("lol") == "und"
    assert detector.detect("lol", detailed=True) == Detection(
        "und", DetectionStatus.LOW_CONFIDENCE, "eng", 0.12
    )
    # Scores are still the model's own.
    assert detector.detect("lol", with_score=True) == {"eng": 0.12, "spa": 0.1}


def test_min_margin_abstains_on_ambiguity() -> None:
    model = ScriptedModel({"cat": 0.5, "spa": 0.45})
    assert LanguageDetector(model, min_margin=0.1).detect("hola", detailed=True) == (
        Detection("und", DetectionStatus.AMBIGUOUS, "cat", 0.5)
    )
    assert LanguageDetector(model, min_margin=0.01).detect("hola") == "cat"


def test_candidates_pick_the_best_candidate() -> None:
    model = ScriptedModel({"eng": 0.6, "cat": 0.3, "spa": 0.1})
    detector = LanguageDetector(model, candidates=["ca", "spa"])
    assert detector.detect("bon dia") == "cat"
    assert detector.detect("bon dia", with_score=True) == {"cat": 0.3, "spa": 0.1}
    assert detector.detect("bon dia", with_score=True, top_k=1) == {"cat": 0.3}
    assert detector.detect("bon dia", detailed=True).score == 0.3
    assert LanguageDetector(model, candidates=["spa"], min_score=0.2).detect(
        "bon dia", detailed=True
    ) == Detection("und", DetectionStatus.LOW_CONFIDENCE, "spa", 0.1)


def test_candidates_need_known_labels_and_full_scores() -> None:
    model = ScriptedModel({"eng": 0.6, "cat": 0.4})
    with pytest.raises(ValueError, match=r"never returns \['fra'\]"):
        LanguageDetector(model, candidates=["fra", "eng"])
    with pytest.raises(TypeError, match="not a string"):
        LanguageDetector(model, candidates="eng")
    with pytest.raises(ValueError, match="at least one"):
        LanguageDetector(model, candidates=[])
    partial = ScriptedModel({"eng": 1.0}, scores_every_label=False)
    with pytest.raises(ValueError, match="doesn't score every language"):
        LanguageDetector(partial, candidates=["eng"])


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("min_score", -0.1, ValueError),
        ("min_score", "0.5", TypeError),
        ("min_margin", -1, ValueError),
        ("min_letters", 0, ValueError),
        ("min_letters", 2.5, TypeError),
        ("min_letters", True, TypeError),
    ],
)
def test_invalid_settings(argument: str, value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match=argument):
        LanguageDetector(ScriptedModel({"eng": 1.0}), **{argument: value})  # ty: ignore[invalid-argument-type]


def test_with_score_and_detailed_are_exclusive() -> None:
    detector = LanguageDetector(ScriptedModel({"eng": 1.0}))
    with pytest.raises(ValueError, match="not both"):
        detector.detect("hi", with_score=True, detailed=True)  # ty: ignore[no-matching-overload]


def test_detect_dataset_status_column() -> None:
    detector = LanguageDetector(ScriptedModel({"eng": 0.3, "spa": 0.2}), min_score=0.5)
    frame = pl.DataFrame({"text": ["hello there", None, "123", "hi"]})
    out = detector.detect_dataset(
        frame, "text", "lang", status_column="status", batch_size=2
    )
    assert out["lang"].to_list() == ["und", "und", "zxx", "und"]
    assert out["status"].to_list() == [
        "low_confidence",
        "empty",
        "no_linguistic_content",
        "low_confidence",
    ]
    scored = detector.detect_dataset(
        frame, "text", "scores", with_score=True, status_column="status"
    )
    assert scored["scores"][0][0]["language"] == "eng"


# Candidate restriction on every backend that scores every language.


def test_candidates_fasttext() -> None:
    model = FastTextModel(TINY)
    text = "el gat seia a la catifa"
    assert model.predict(text) == "cat"
    assert LanguageDetector(model, candidates=["spa", "eng"]).detect(text) in {
        "spa",
        "eng",
    }
    scores = LanguageDetector(model, candidates=["spa", "eng"]).detect(
        text, with_score=True, top_k=None
    )
    assert set(scores) == {"spa", "eng"}


def test_candidates_langid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    _write_sdist(tmp_path / LANGID_SDIST.cache_path, _model_pickle())
    model = LangidModel()
    assert model.predict("aab") == "eng"
    assert LanguageDetector(model, candidates=["cat"]).detect("aab") == "cat"


def test_candidates_onnx() -> None:
    model, _ = _fake_model({"input_ids", "attention_mask"})
    # The fake scores long texts as Spanish.
    assert model.predict("a b c d e") == "spa"
    assert LanguageDetector(model, candidates=["eng"]).detect("a b c d e") == "eng"


def test_candidates_lingua() -> None:
    pytest.importorskip("lingua")
    from noentenc.language_detection.models.lingua_model import LinguaModel

    model = LinguaModel(languages=["spa", "cat", "eng"])
    detector = LanguageDetector(model, candidates=["spa", "eng"])
    assert detector.detect("Bon dia a tothom, com esteu?") in {"spa", "eng"}


@pytest.mark.parametrize("backend", ["cld3", "heliport"])
def test_candidates_unsupported_backends(backend: str) -> None:
    if backend == "cld3":
        pytest.importorskip("gcld3")
        from noentenc.language_detection.models.cld3_model import Cld3Model as Model
    else:
        pytest.importorskip("heliport")
        from noentenc.language_detection.models.heliport_model import (
            HeliportModel as Model,
        )
    with pytest.raises(ValueError, match="doesn't score every language"):
        LanguageDetector(Model(), candidates=["eng"])
