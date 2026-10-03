import pandas as pd
import polars as pl
import pytest

from noentenc.languages import (
    ANY_LANGUAGE,
    Language,
    LanguageSchema,
    UnsupportedLanguageError,
)
from noentenc.translation import base as translator_module
from noentenc.translation.base import Translator
from noentenc.translation.models.base import BaseModel
from noentenc.translation.models.small100 import SMaLL100Model

EN, ES = Language.ENGLISH, Language.SPANISH


class UpperModel(BaseModel):
    schema = LanguageSchema(source=ANY_LANGUAGE, target=ANY_LANGUAGE)

    def __init__(self) -> None:
        super().__init__("upper")
        self.batches: list[list[str]] = []

    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        self.batches.append(texts)
        return [f"{text.upper()}:{target_language}" for text in texts]


@pytest.fixture
def translator() -> Translator:
    return Translator(UpperModel())


def test_translate_and_batch(translator: Translator) -> None:
    assert translator.translate("hola", EN) == "HOLA:en"
    assert translator.translate_batch(["a", "b"], ES) == ["A:es", "B:es"]


def test_translate_dataset_polars_keeps_nulls_and_order(translator: Translator) -> None:
    frame = pl.DataFrame({"text": ["a", None, "c"]})
    result = translator.translate_dataset(frame, "text", "translated", EN)
    assert isinstance(result, pl.DataFrame)
    assert result["translated"].to_list() == ["A:en", None, "C:en"]
    assert result.schema["translated"] == pl.String
    assert "translated" not in frame.columns


def test_translate_dataset_pandas_keeps_nulls_and_order(translator: Translator) -> None:
    frame = pd.DataFrame({"text": ["a", None, "c"]})
    result = translator.translate_dataset(frame, "text", "translated", EN)
    assert isinstance(result, pd.DataFrame)
    assert result["translated"].tolist()[0::2] == ["A:en", "C:en"]
    assert pd.isna(result["translated"].iloc[1])
    assert "translated" not in frame.columns


def test_translate_dataset_skips_nulls_in_model_calls(translator: Translator) -> None:
    model = translator.model
    assert isinstance(model, UpperModel)
    translator.translate_dataset(pl.DataFrame({"text": [None, "b"]}), "text", "out", EN)
    assert model.batches == [["b"]]


def test_translate_dataset_rejects_other_types(translator: Translator) -> None:
    with pytest.raises(TypeError, match="polars or pandas"):
        translator.translate_dataset({"text": ["a"]}, "text", "out", EN)  # ty: ignore[no-matching-overload]


class FakeOpus(UpperModel):
    created: list[tuple[Language, Language]] = []

    @classmethod
    def from_pair(cls, source: Language, target: Language) -> FakeOpus:
        cls.created.append((source, target))
        return cls()


class FakeSmall100(UpperModel):
    schema = SMaLL100Model.schema
    created = 0

    def __init__(self) -> None:
        FakeSmall100.created += 1
        super().__init__()


@pytest.fixture
def default_translator(monkeypatch: pytest.MonkeyPatch) -> Translator:
    FakeOpus.created = []
    FakeSmall100.created = 0
    monkeypatch.setattr(translator_module, "OpusMTModel", FakeOpus)
    monkeypatch.setattr(translator_module, "SMaLL100Model", FakeSmall100)
    return Translator()


def test_default_uses_opus_for_direct_pairs(default_translator: Translator) -> None:
    default_translator.translate("hello", ES, EN)
    default_translator.translate("again", ES, EN)
    assert FakeOpus.created == [(EN, ES)]
    assert FakeSmall100.created == 0


def test_default_falls_back_to_small100(default_translator: Translator) -> None:
    default_translator.translate("bon dia", Language.JAPANESE, Language.CATALAN)
    default_translator.translate("hello", ES)  # no source: Opus-MT cannot be chosen
    assert FakeOpus.created == []
    assert FakeSmall100.created == 1


def test_default_rejects_pairs_no_model_supports(
    default_translator: Translator,
) -> None:
    with pytest.raises(UnsupportedLanguageError):
        # Acehnese is NLLB-only, and NLLB is never picked by default.
        default_translator.translate("hello", Language.ACEHNESE, EN)


def test_default_rejection_without_source_names_only_the_target(
    default_translator: Translator,
) -> None:
    with pytest.raises(UnsupportedLanguageError, match="translates into ace;"):
        default_translator.translate("hello", Language.ACEHNESE)
