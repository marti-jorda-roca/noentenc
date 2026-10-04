import pandas as pd
import polars as pl
import pytest

from noentenc.language_detection import LanguageDetector
from noentenc.language_detection import base as detector_module
from noentenc.language_detection.models.base import BaseModel
from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.profiles import Profile
from tests.unit.language_detection.helpers import FASTTEXT_FIXTURES

TINY = FASTTEXT_FIXTURES / "tiny-softmax.bin"


@pytest.fixture
def detector() -> LanguageDetector:
    return LanguageDetector(FastTextModel(TINY))


def test_profiles_pick_fasttext_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class FakeFastText(FastTextModel):
        def __init__(self, model: str = "lid176") -> None:
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


def test_invalid_batch_size(detector: LanguageDetector) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        detector.detect_batch(["a"], batch_size=0)


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
