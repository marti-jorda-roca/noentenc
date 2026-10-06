from itertools import islice

import pandas as pd
import polars as pl
import pytest

from noentenc.language_detection import LanguageDetector
from noentenc.language_detection import base as detector_module
from noentenc.language_detection.models.base import BaseModel
from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.profiles import Profile
from tests.unit.helpers import CountingTexts
from tests.unit.language_detection.helpers import FASTTEXT_FIXTURES

TINY = FASTTEXT_FIXTURES / "tiny-softmax.bin"


@pytest.fixture
def detector() -> LanguageDetector:
    return LanguageDetector(FastTextModel(TINY))


def test_profiles_pick_fasttext_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class FakeFastText(FastTextModel):
        def __init__(
            self,
            model: str = "lid176",
            only_local_files: bool = False,
            *,
            cache_dir: object = None,
        ) -> None:
            created.append(model)
            super().__init__(TINY)

    monkeypatch.setattr(detector_module, "FastTextModel", FakeFastText)
    detector = LanguageDetector()
    assert created == ["lid176"]
    assert isinstance(detector.model, BaseModel)
    for profile in Profile:
        LanguageDetector(profile)
    LanguageDetector("quality")
    assert created == ["lid176", "lid176", "openlid-v3", "glotlid", "glotlid"]


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="'fast' is not a valid Profile"):
        LanguageDetector("fast")


def test_detect_and_batch(detector: LanguageDetector) -> None:
    assert detector.detect("der hund lief im park") == "deu"
    scores = detector.detect("der hund lief im park", with_score=True, top_k=2)
    assert isinstance(scores, dict) and len(scores) == 2 and next(iter(scores)) == "deu"
    assert detector.detect_batch(
        ["el perro corrió", "", "кошка сидела"], batch_size=2
    ) == ["spa", "und", "rus"]
    batch_scores = detector.detect_batch(
        ["el perro corrió"], with_score=True, top_k=None
    )
    assert len(batch_scores[0]) == 6
    assert sum(batch_scores[0].values()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("batch_size", [0, -1, 2.5, "32", True])
def test_invalid_batch_size(detector: LanguageDetector, batch_size: object) -> None:
    frame = pl.DataFrame({"text": ["a"]})
    for texts in (["a"], []):
        with pytest.raises((TypeError, ValueError), match="batch_size"):
            detector.detect_batch(texts, batch_size=batch_size)  # ty: ignore[invalid-argument-type]
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        detector.detect_dataset(frame, "text", "lang", batch_size=batch_size)  # ty: ignore[no-matching-overload]
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        detector.model.predict_batch(["a"], batch_size=batch_size)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("top_k", [0, -1, 1.5])
def test_invalid_top_k(detector: LanguageDetector, top_k: object) -> None:
    frame = pl.DataFrame({"text": ["a"]})
    with pytest.raises((TypeError, ValueError), match="top_k"):
        detector.detect("a", with_score=True, top_k=top_k)  # ty: ignore[no-matching-overload]
    with pytest.raises((TypeError, ValueError), match="top_k"):
        detector.detect_batch(["a"], with_score=True, top_k=top_k)  # ty: ignore[no-matching-overload]
    with pytest.raises((TypeError, ValueError), match="top_k"):
        detector.detect_dataset(frame, "text", "lang", with_score=True, top_k=top_k)  # ty: ignore[no-matching-overload]


def test_invalid_texts(detector: LanguageDetector) -> None:
    with pytest.raises(TypeError, match="texts must be a list"):
        detector.detect_batch("hello")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"texts\[1\] must be a string, got int"):
        detector.detect_batch(["a", 1])  # ty: ignore[invalid-argument-type]


def test_empty_and_blank_batches(detector: LanguageDetector) -> None:
    assert detector.detect_batch([]) == []
    assert detector.detect_batch([], with_score=True) == []
    assert detector.detect_batch(["", " \n"]) == ["und", "und"]
    empty = pl.DataFrame({"text": []}, schema={"text": pl.String})
    assert detector.detect_dataset(empty, "text", "lang")["lang"].to_list() == []


def test_batch_size_none_uses_the_backend_default(
    detector: LanguageDetector, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert FastTextModel.default_batch_size == 256
    model = detector.model
    sizes: list[int] = []
    run = model._predict_chunk

    def recording(texts: list[str]) -> list[str]:
        sizes.append(len(texts))
        return run(texts)

    monkeypatch.setattr(model, "_predict_chunk", recording)
    monkeypatch.setattr(model, "default_batch_size", 2)
    detector.detect_batch(["el perro corrió"] * 5)
    assert sizes == [2, 2, 1]
    detector.detect_batch(["el perro corrió"] * 5, batch_size=4)
    assert sizes[3:] == [4, 1]


def test_batching_does_not_change_results(
    detector: LanguageDetector, sentences: list[str]
) -> None:
    assert detector.detect_batch(sentences, batch_size=1) == detector.detect_batch(
        sentences, batch_size=1000
    )


def test_detect_dataset_polars(detector: LanguageDetector) -> None:
    df = pl.DataFrame({"text": ["the dog ran in the park", None, "кошка сидела"]})
    out = detector.detect_dataset(df, "text", "lang")
    assert out["lang"].to_list() == ["eng", "und", "rus"]
    with_score = detector.detect_dataset(df, "text", "scores", with_score=True, top_k=2)
    assert with_score.schema["scores"] == pl.List(
        pl.Struct({"language": pl.String, "score": pl.Float64})
    )
    assert with_score["scores"][0][0]["language"] == "eng"


def test_detect_dataset_pandas(detector: LanguageDetector) -> None:
    df = pd.DataFrame({"text": ["the dog ran in the park", None]})
    out = detector.detect_dataset(df, "text", "lang", with_score=False)
    assert out["lang"].tolist() == ["eng", "und"]
    assert "lang" not in df.columns  # input is not mutated
    with_score = detector.detect_dataset(df, "text", "lang", with_score=True, top_k=1)
    assert with_score["lang"].iloc[0] == [
        {
            "language": "eng",
            "score": pytest.approx(with_score["lang"].iloc[0][0]["score"]),
        }
    ]


def test_detect_dataset_rejects_other_types(detector: LanguageDetector) -> None:
    with pytest.raises(TypeError, match="polars or pandas"):
        detector.detect_dataset({"text": ["a"]}, "text", "lang")  # ty: ignore[no-matching-overload]


def test_detect_stream_matches_detect_batch(
    detector: LanguageDetector, sentences: list[str]
) -> None:
    assert list(detector.detect_stream(iter(sentences), chunk_size=7)) == (
        detector.detect_batch(sentences)
    )
    # Scores move in the last float digits with batch composition; labels don't.
    streamed = list(detector.detect_stream(sentences, chunk_size=7, detailed=True))
    batch = detector.detect_batch(sentences, detailed=True)
    assert [(d.language, d.status) for d in streamed] == [
        (d.language, d.status) for d in batch
    ]
    assert [d.score for d in streamed] == pytest.approx([d.score for d in batch])
    scores = detector.detect_stream(sentences, with_score=True, top_k=2, chunk_size=7)
    batch_scores = detector.detect_batch(sentences, with_score=True, top_k=2)
    for streamed_scores, expected in zip(scores, batch_scores, strict=True):
        assert streamed_scores == pytest.approx(expected)


def test_detect_stream_reads_one_chunk_at_a_time(detector: LanguageDetector) -> None:
    texts = CountingTexts(["el perro corrió"] * 5)
    stream = detector.detect_stream(texts, chunk_size=2)
    assert texts.read == 0
    labels: list[str] = []
    for label in stream:
        labels.append(label)
        # Never more than one chunk read ahead of what was returned.
        assert texts.read - len(labels) < 2
    assert labels == ["spa"] * 5
    texts = CountingTexts(["el perro corrió"] * 5)
    assert list(islice(detector.detect_stream(texts, chunk_size=2), 3)) == ["spa"] * 3
    assert texts.read == 4


def test_detect_stream_checks_arguments_before_reading(
    detector: LanguageDetector,
) -> None:
    texts = CountingTexts(["hello"])
    with pytest.raises(ValueError, match="batch_size"):
        detector.detect_stream(texts, batch_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        detector.detect_stream(texts, chunk_size=0)
    with pytest.raises(ValueError, match="top_k"):
        detector.detect_stream(texts, top_k=0)
    with pytest.raises(ValueError, match="not both"):
        detector.detect_stream(texts, with_score=True, detailed=True)  # ty: ignore[no-matching-overload]
    with pytest.raises(TypeError, match="iterable of strings, got str"):
        detector.detect_stream("hello")
    assert texts.read == 0
    stream = detector.detect_stream(["el perro corrió", 1], chunk_size=1)  # ty: ignore[invalid-argument-type]
    assert next(stream) == "spa"
    with pytest.raises(TypeError, match=r"texts\[1\] must be a string, got int"):
        next(stream)
