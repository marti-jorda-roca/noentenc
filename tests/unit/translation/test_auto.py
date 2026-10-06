"""`source_language="auto"`: detect, group by language and translate in one call."""

from itertools import islice
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import noentenc
from noentenc import Language, LanguageSchema
from noentenc.language_detection import LanguageDetector
from noentenc.language_detection import _download as download_module
from noentenc.language_detection import base as detector_module
from noentenc.language_detection.labels import LabelMapper
from noentenc.language_detection.models.base import BaseModel as DetectionModel
from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.translation import Translation, TranslationStatus, Translator
from noentenc.translation.base import AUTO_DETECTION_SETTINGS, SourceLanguageError
from tests.unit.helpers import CountingTexts
from tests.unit.language_detection.helpers import FASTTEXT_FIXTURES
from tests.unit.translation.test_translator import (  # noqa: F401 - fixture
    FakeNLLB,
    FakeOpus,
    FakeSeq2Seq,
    FakeSmall100,
    UpperModel,
    fake_models,
)

EN, ES, DE, FR, CA = (
    Language.ENGLISH,
    Language.SPANISH,
    Language.GERMAN,
    Language.FRENCH,
    Language.CATALAN,
)

# What the scripted detector answers for each text: its label and score.
LABELS = {
    "Guten Morgen": ("deu", 0.95),
    "Buenos días": ("spa", 0.9),
    "Bonjour à tous": ("fra", 0.9),
    "Good morning": ("eng", 0.97),
    "Bon dia a tothom": ("cat", 0.8),
    "Egun on denoi": ("eus", 0.8),
    "Grüezi mitenand": ("gsw", 0.9),
    "Hans Müller": ("deu", 0.2),
}


class ScriptedDetector(DetectionModel):
    def __init__(self) -> None:
        super().__init__("scripted")
        labels = sorted({label for label, _ in LABELS.values()} | {"eng"})
        self._mapper = LabelMapper(labels)
        self.seen: list[str] = []

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        return [next(iter(scores)) for scores in self._predict_score_chunk(texts, 1)]

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        self.seen += texts
        return [dict([LABELS.get(text, ("eng", 0.3))]) for text in texts]


@pytest.fixture
def detector() -> LanguageDetector:
    return LanguageDetector(ScriptedDetector(), **AUTO_DETECTION_SETTINGS)  # ty: ignore[invalid-argument-type]


@pytest.fixture
def translator(fake_models: None, detector: LanguageDetector) -> Translator:  # noqa: ARG001, F811
    return Translator(detector=detector)


INBOX = [
    "Guten Morgen",
    "Buenos días",
    "Good morning",
    "👍 https://example.com",
    "",
    "lol",
    "Bonjour à tous",
    "Guten Morgen",
]


def test_mixed_inbox_in_one_call(translator: Translator) -> None:
    results = translator.translate_batch(INBOX, EN, "auto", detailed=True)
    assert [r.text for r in results] == [
        "GUTEN MORGEN:en",
        "BUENOS DÍAS:en",
        "Good morning",
        "👍 https://example.com",
        "",
        "lol",
        "BONJOUR À TOUS:en",
        "GUTEN MORGEN:en",
    ]
    assert [r.status for r in results] == [
        TranslationStatus.TRANSLATED,
        TranslationStatus.TRANSLATED,
        TranslationStatus.UNCHANGED,
        TranslationStatus.UNCHANGED,
        TranslationStatus.UNCHANGED,
        TranslationStatus.UNKNOWN_SOURCE,
        TranslationStatus.TRANSLATED,
        TranslationStatus.TRANSLATED,
    ]
    assert results[0] == Translation(
        "GUTEN MORGEN:en",
        source_language=DE,
        detected_language="deu",
        detection_score=0.95,
        model="FakeOpus(upper)",
    )
    assert results[2].source_language == EN
    assert results[3].detected_language == "zxx"
    assert results[5].detected_language == "und"
    assert results[4].detected_language is None  # blank: not detected
    # One model per source pair, loaded once; the repeated German text is batched.
    assert FakeOpus.created == [(DE, EN), (ES, EN), (FR, EN)]


def test_pairs_without_opus_use_the_fallback(translator: Translator) -> None:
    results = translator.translate_batch(
        ["Bon dia a tothom", "Egun on denoi", "Buenos días"], EN, "auto", detailed=True
    )
    assert [r.source_language for r in results] == [CA, Language.BASQUE, ES]
    assert [r.model for r in results] == [
        "FakeSmall100(upper)",
        "FakeSmall100(upper)",
        "FakeOpus(upper)",
    ]
    assert FakeSmall100.created == 1


def test_target_language_texts_pass_through(translator: Translator) -> None:
    assert translator.translate("Guten Morgen", "de", "auto") == "Guten Morgen"
    assert FakeSeq2Seq.loads == []


def test_unknown_source_policies(
    translator: Translator, detector: LanguageDetector
) -> None:
    texts = ["Hans Müller", "Guten Morgen"]
    kept = translator.translate_batch(texts, EN, "auto", detailed=True)
    assert kept[0].status is TranslationStatus.UNKNOWN_SOURCE
    assert kept[0].text == "Hans Müller"
    assert kept[0].detection_score == 0.2  # the guess it didn't trust
    fallback = translator.translate_batch(
        texts, EN, "auto", detailed=True, unknown_source="fallback"
    )
    assert fallback[0].text == "HANS MÜLLER:en"
    assert fallback[0].source_language is None
    assert fallback[0].model == "FakeSmall100(upper)"
    loads = len(FakeSeq2Seq.loads)
    with pytest.raises(SourceLanguageError, match=r"texts\[0\] \(unknown_source"):
        Translator(detector=detector).translate_batch(
            texts, EN, "auto", unknown_source="raise"
        )
    assert len(FakeSeq2Seq.loads) == loads  # nothing loaded before failing


class EnglishToSpanish(UpperModel):
    schema = LanguageSchema(source=frozenset({EN}), target=frozenset({ES}))


def test_unsupported_detected_languages(
    translator: Translator, detector: LanguageDetector
) -> None:
    (swiss,) = translator.translate_batch(
        ["Grüezi mitenand"], EN, "auto", detailed=True
    )
    assert swiss.status is TranslationStatus.UNSUPPORTED_SOURCE
    assert swiss.detected_language == "gsw"
    # A detected language the explicit model doesn't read.
    english_only = Translator(EnglishToSpanish(), detector=detector)
    results = english_only.translate_batch(
        ["Good morning", "Guten Morgen"], ES, "auto", detailed=True
    )
    assert [r.status for r in results] == [
        TranslationStatus.TRANSLATED,
        TranslationStatus.UNSUPPORTED_SOURCE,
    ]


@pytest.mark.usefixtures("fake_models")
def test_auto_respects_the_loaded_model_limit(
    detector: LanguageDetector,
) -> None:
    translator = Translator(max_loaded_models=1, detector=detector)
    texts = ["Guten Morgen", "Buenos días", "Bonjour à tous"] * 3
    results = translator.translate_batch(texts, EN, "auto")
    assert (
        results
        == [
            "GUTEN MORGEN:en",
            "BUENOS DÍAS:en",
            "BONJOUR À TOUS:en",
        ]
        * 3
    )
    assert FakeOpus.created == [(DE, EN), (ES, EN), (FR, EN)]
    assert len(translator.loaded_models) == 1


@pytest.mark.usefixtures("fake_models")
def test_dataset_auto_keeps_rows_aligned(
    detector: LanguageDetector,
) -> None:
    translator = Translator(max_loaded_models=1, detector=detector)
    frame = pl.DataFrame(
        {
            "text": [
                "Guten Morgen",
                None,
                "Buenos días",
                "lol",
                "Guten Morgen",
                "Good morning",
            ]
        }
    )
    out = translator.translate_dataset(
        frame,
        "text",
        "en",
        EN,
        "auto",
        batch_size=2,
        status_column="status",
        source_column="source",
    )
    assert out["en"].to_list() == [
        "GUTEN MORGEN:en",
        None,
        "BUENOS DÍAS:en",
        "lol",
        "GUTEN MORGEN:en",
        "Good morning",
    ]
    assert out["status"].to_list() == [
        "translated",
        None,
        "translated",
        "unknown_source",
        "translated",
        "unchanged",
    ]
    assert out["source"].to_list() == ["de", None, "es", None, "de", "en"]
    # Grouped over the whole column, so chunks don't reload models.
    assert FakeOpus.created == [(DE, EN), (ES, EN)]


def test_dataset_auto_pandas(translator: Translator) -> None:
    frame = pd.DataFrame({"text": ["Buenos días", None]})
    out = translator.translate_dataset(
        frame, "text", "en", "en", "auto", status_column="s"
    )
    assert out["en"].iloc[0] == "BUENOS DÍAS:en"
    assert pd.isna(out["en"].iloc[1])
    assert out["s"].iloc[0] == "translated"


def test_languages_can_be_codes_and_names(translator: Translator) -> None:
    assert translator.translate("hi", "spanish", "en") == "HI:es"
    assert translator.translate("hi", "es-ES", "eng") == "HI:es"
    assert FakeOpus.created == [(EN, ES)]


def test_invalid_unknown_source_policy(translator: Translator) -> None:
    with pytest.raises(ValueError, match="unknown_source must be"):
        translator.translate("hi", EN, "auto", unknown_source="drop")  # ty: ignore[invalid-argument-type]


@pytest.mark.usefixtures("fake_models")
def test_default_detector_is_the_profile_detector_with_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    class TinyFastText(FastTextModel):
        def __init__(self, model: str, *args: object, **kwargs: object) -> None:
            built.append(model)
            super().__init__(FASTTEXT_FIXTURES / "tiny-softmax.bin")

    monkeypatch.setattr(detector_module, "FastTextModel", TinyFastText)
    translator = Translator("balance")
    assert built == []  # not until "auto" is used
    assert translator.translate("el perro corrió en el parque", EN, "auto")
    assert built == ["openlid-v3"]
    assert translator.detector.policy.min_letters == 4
    assert translator.detector.policy.min_score == 0.5


@pytest.mark.usefixtures("fake_models")
def test_prepare_auto_downloads_every_route_and_the_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetched: list[str] = []

    def fake_fetch(remote: download_module.RemoteFile, **_kwargs: object) -> Path:
        fetched.append(remote.cache_path)
        return tmp_path / remote.cache_path

    monkeypatch.setattr(download_module, "fetch", fake_fetch)
    noentenc.prepare("speed", detection=False, translation=[("auto", "en")])
    assert fetched == ["fasttext/lid.176.ftz"]
    downloads = [load["class"] for load in FakeSeq2Seq.loads]
    assert downloads.count("FakeOpus") == 25
    assert downloads.count("FakeSmall100") == 1
    assert "FakeNLLB" not in downloads
    FakeSeq2Seq.loads = []
    noentenc.prepare("balance", detection=False, translation=[("auto", "en")])
    assert [load["class"] for load in FakeSeq2Seq.loads].count("FakeNLLB") == 1
    assert FakeNLLB.created == []  # downloaded, never loaded


@pytest.mark.usefixtures("fake_models")
def test_stream_detects_and_groups_each_chunk(detector: LanguageDetector) -> None:
    texts = CountingTexts(INBOX)
    stream = Translator(detector=detector).translate_stream(
        texts, EN, "auto", detailed=True, chunk_size=3
    )
    first = next(stream)
    assert first.source_language == DE
    # Only the first chunk has been read and detected.
    assert texts.read == 3
    assert isinstance(detector.model, ScriptedDetector)
    assert detector.model.seen == INBOX[:3]
    batch = Translator(detector=detector).translate_batch(
        INBOX, EN, "auto", detailed=True
    )
    assert [first, *stream] == batch


def test_stream_unknown_source_raises_at_its_chunk(translator: Translator) -> None:
    texts = ["Guten Morgen", "Buenos días", "Bonjour à tous", "Hans Müller"]
    stream = translator.translate_stream(
        texts, EN, "auto", unknown_source="raise", chunk_size=2
    )
    assert list(islice(stream, 2)) == ["GUTEN MORGEN:en", "BUENOS DÍAS:en"]
    # The index counts from the start of the stream, not of the chunk.
    with pytest.raises(SourceLanguageError, match=r"texts\[3\] \(unknown_source"):
        next(stream)
