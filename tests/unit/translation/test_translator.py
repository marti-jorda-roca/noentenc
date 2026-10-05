from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import polars as pl
import pytest

from noentenc.languages import (
    ANY_LANGUAGE,
    Language,
    LanguageSchema,
    UnsupportedLanguageError,
)
from noentenc.profiles import Profile
from noentenc.translation import base as translator_module
from noentenc.translation.base import Translator
from noentenc.translation.models.base import (
    BaseModel,
    InputTooLongError,
    Translation,
    TranslationStatus,
)
from noentenc.translation.models.nllb import NLLBModel
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


def test_detailed_results_default_to_nothing_missing(translator: Translator) -> None:
    assert translator.translate("hola", EN, detailed=True) == Translation("HOLA:en")
    assert translator.translate_batch(["a"], ES, detailed=True) == [Translation("A:es")]


class TruncatingModel(UpperModel):
    """Reports every text as truncated when asked to truncate."""

    def predict_batch_detailed(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
    ) -> list[Translation]:
        return [
            Translation(text, input_truncated=truncate)
            for text in self.predict_batch(texts, target_language, source_language)
        ]


def test_truncate_reaches_the_model() -> None:
    translator = Translator(TruncatingModel())
    assert translator.translate("a", EN) == "A:en"
    assert translator.translate("a", EN, detailed=True).input_truncated is False
    assert translator.translate("a", EN, truncate=True, detailed=True).input_truncated
    (result,) = translator.translate_batch(["a"], EN, truncate=True, detailed=True)
    assert result.input_truncated
    frame = pl.DataFrame({"text": ["a"]})
    out = translator.translate_dataset(frame, "text", "out", EN, truncate=True)
    assert out["out"].to_list() == ["A:en"]


class FakeOpus(UpperModel):
    created: list[tuple[Language, Language]] = []

    @classmethod
    def from_pair(cls, source: Language, target: Language) -> FakeOpus:
        cls.created.append((source, target))
        return cls()


class FakeSmall100(UpperModel):
    schema = SMaLL100Model.schema
    created = 0

    def __init__(self, precision: str | None = None) -> None:
        FakeSmall100.created += 1
        super().__init__()


class FakeNLLB(UpperModel):
    schema = NLLBModel.schema
    created: list[str | None] = []

    def __init__(self, precision: str | None = None) -> None:
        FakeNLLB.created.append(precision)
        super().__init__()


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeOpus.created = []
    FakeSmall100.created = 0
    FakeNLLB.created = []
    monkeypatch.setattr(translator_module, "OpusMTModel", FakeOpus)
    monkeypatch.setattr(translator_module, "SMaLL100Model", FakeSmall100)
    monkeypatch.setattr(translator_module, "NLLBModel", FakeNLLB)


@pytest.fixture
def default_translator(fake_models: None) -> Translator:  # noqa: ARG001
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
    assert FakeNLLB.created == []


def test_default_rejects_pairs_no_model_supports(
    default_translator: Translator,
) -> None:
    with pytest.raises(UnsupportedLanguageError):
        # Acehnese is NLLB-only, and the speed profile never picks NLLB.
        default_translator.translate("hello", Language.ACEHNESE, EN)


def test_default_rejection_without_source_names_only_the_target(
    default_translator: Translator,
) -> None:
    with pytest.raises(UnsupportedLanguageError, match="translates into ace;"):
        default_translator.translate("hello", Language.ACEHNESE)


def test_profile_accepts_strings_and_rejects_unknown_names() -> None:
    assert Translator("quality").profile is Profile.QUALITY
    assert Translator(Profile.BALANCE).profile is Profile.BALANCE
    assert Translator().profile is Profile.SPEED
    with pytest.raises(ValueError, match="'fast' is not a valid Profile"):
        Translator("fast")


@pytest.mark.usefixtures("fake_models")
def test_explicit_model_ignores_profiles() -> None:
    translator = Translator(UpperModel())
    translator.translate("hello", ES, EN)
    assert FakeOpus.created == []


@pytest.mark.usefixtures("fake_models")
@pytest.mark.parametrize(
    ("profile", "precision"), [(Profile.BALANCE, "int8"), (Profile.QUALITY, "fp32")]
)
def test_profiles_prefer_opus_then_nllb(profile: Profile, precision: str) -> None:
    translator = Translator(profile)
    translator.translate("hello", ES, EN)
    translator.translate("bon dia", Language.JAPANESE, Language.CATALAN)
    translator.translate("bon dia", Language.ACEHNESE, Language.CATALAN)
    assert FakeOpus.created == [(EN, ES)]
    assert FakeNLLB.created == [precision]
    assert FakeSmall100.created == 0


@pytest.mark.usefixtures("fake_models")
@pytest.mark.parametrize("profile", list(Profile))
def test_every_profile_falls_back_to_small100_without_source(profile: Profile) -> None:
    Translator(profile).translate("hello", ES)
    assert FakeSmall100.created == 1
    assert FakeNLLB.created == []


@pytest.mark.usefixtures("fake_models")
def test_rejection_names_the_profile() -> None:
    with pytest.raises(UnsupportedLanguageError, match="No quality model translates"):
        Translator("quality").translate("hello", Language.ACEHNESE)


class FailingModel(UpperModel):
    """Fails any batch that holds a text starting with "bad"."""

    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        if any(text.startswith("bad") for text in texts):
            self.batches.append(texts)
            raise InputTooLongError(f"too long: {texts}")
        return super().predict_batch(texts, target_language, source_language)


@pytest.mark.usefixtures("fake_models")
@pytest.mark.parametrize("batch_size", [0, -1, 2.5, "32", True])
def test_invalid_batch_size_fails_before_loading_a_model(batch_size: object) -> None:
    translator = Translator()
    frame = pl.DataFrame({"text": ["hello"]})
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        translator.translate_batch(["hello"], ES, EN, batch_size=batch_size)  # ty: ignore[invalid-argument-type]
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        translator.translate_dataset(frame, "text", "out", ES, EN, batch_size)  # ty: ignore[no-matching-overload]
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        translator.translate_batch([], ES, EN, batch_size=batch_size)  # ty: ignore[invalid-argument-type]
    assert FakeOpus.created == []


@pytest.mark.usefixtures("fake_models")
def test_invalid_texts_and_errors_fail_before_loading_a_model() -> None:
    translator = Translator()
    with pytest.raises(TypeError, match="texts must be a list"):
        translator.translate_batch("hello", ES, EN)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"texts\[1\] must be a string, got NoneType"):
        translator.translate_batch(["a", None], ES, EN)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="errors must be"):
        translator.translate_batch(["a"], ES, EN, errors="ignore")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="is not a valid Language"):
        translator.translate_batch([], "xx", EN)  # ty: ignore[invalid-argument-type]
    assert FakeOpus.created == []


@pytest.mark.usefixtures("fake_models")
def test_trivial_inputs_load_no_model() -> None:
    translator = Translator()
    assert translator.translate_batch([], ES, EN) == []
    assert translator.translate_batch(["", "  \n"], ES) == ["", "  \n"]
    assert translator.translate("hello", EN, EN) == "hello"
    # Same language needs no model even where no model supports it.
    assert translator.translate("hello", Language.ACEHNESE, Language.ACEHNESE)
    frame = pl.DataFrame({"text": [None, "", "hi"]})
    out = translator.translate_dataset(frame, "text", "out", EN, EN)
    assert out["out"].to_list() == [None, "", "hi"]
    assert translator.translate_dataset(
        pl.DataFrame({"text": [None]}), "text", "out", ES
    )["out"].to_list() == [None]
    assert FakeOpus.created == []
    assert FakeSmall100.created == 0


def test_blank_and_same_language_results_are_unchanged() -> None:
    model = UpperModel()
    translator = Translator(model)
    results = translator.translate_batch(["a", " ", "b"], ES, EN, detailed=True)
    assert results == [
        Translation("A:es"),
        Translation(" ", status=TranslationStatus.UNCHANGED),
        Translation("B:es"),
    ]
    assert model.batches == [["a", "b"]]
    (same,) = translator.translate_batch(["a"], EN, EN, detailed=True)
    assert same == Translation("a", status=TranslationStatus.UNCHANGED)


def test_explicit_model_rejects_unsupported_pairs_before_translating() -> None:
    class EnglishOnly(UpperModel):
        schema = LanguageSchema(source=ANY_LANGUAGE, target=frozenset({EN}))

    model = EnglishOnly()
    with pytest.raises(UnsupportedLanguageError, match="cannot translate into"):
        Translator(model).translate_batch(["a"], ES, errors="record")
    assert model.batches == []


def test_strict_mode_raises_the_model_error() -> None:
    translator = Translator(FailingModel())
    with pytest.raises(InputTooLongError, match="too long"):
        translator.translate_batch(["ok", "bad"], EN)
    with pytest.raises(InputTooLongError):
        translator.translate_dataset(pl.DataFrame({"text": ["bad"]}), "text", "o", EN)


def test_record_mode_isolates_failing_rows() -> None:
    model = FailingModel()
    translator = Translator(model)
    texts = ["a", "bad1", "b", "c", "d", "e", "bad2", "f"]
    results = translator.translate_batch(texts, EN, detailed=True, errors="record")
    assert [r.text for r in results] == [
        "A:en", "bad1", "B:en", "C:en", "D:en", "E:en", "bad2", "F:en"
    ]  # fmt: skip
    failed = {i: r.error for i, r in enumerate(results) if r.error is not None}
    assert list(failed) == [1, 6]
    assert failed[1] == "InputTooLongError: too long: ['bad1']"
    assert {r.status for i, r in enumerate(results) if i not in failed} == {
        TranslationStatus.TRANSLATED
    }
    assert results[1].status is TranslationStatus.FAILED
    # Failing batches are halved, so the good texts next to them stay batched.
    assert ["b", "c"] in model.batches
    plain = translator.translate_batch(["bad", "a"], EN, errors="record")
    assert plain == ["bad", "A:en"]


@pytest.mark.parametrize("frame_type", [pl.DataFrame, pd.DataFrame])
def test_record_mode_in_datasets_keeps_nulls_and_alignment(
    frame_type: type[pl.DataFrame] | type[pd.DataFrame],
) -> None:
    translator = Translator(FailingModel())
    frame = frame_type({"text": ["a", None, "bad", "", "b"]})
    out = translator.translate_dataset(
        frame,
        "text",
        "out",
        EN,
        batch_size=2,
        show_progress=False,
        errors="record",
        error_column="error",
    )
    translated = [None if pd.isna(v) else v for v in list(out["out"])]
    errors = [None if pd.isna(v) else v for v in list(out["error"])]
    assert translated == ["A:en", None, None, "", "B:en"]
    assert errors == [None, None, "InputTooLongError: too long: ['bad']", None, None]
    assert list(out["text"]) == list(frame["text"])


# Bounded model cache.

FR, DE = Language.FRENCH, Language.GERMAN


@pytest.mark.usefixtures("fake_models")
def test_loaded_models_stay_within_the_limit() -> None:
    translator = Translator(max_loaded_models=2)
    for target in (ES, DE, FR):
        assert translator.translate("hi", target, EN) == f"HI:{target}"
    assert FakeOpus.created == [(EN, ES), (EN, DE), (EN, FR)]
    assert len(translator.loaded_models) == 2


@pytest.mark.usefixtures("fake_models")
def test_least_recently_used_model_is_evicted_and_reloaded() -> None:
    translator = Translator(max_loaded_models=2)
    translator.translate("hi", ES, EN)
    translator.translate("hi", DE, EN)
    translator.translate("hi", ES, EN)  # en->es is now the most recent
    translator.translate("hi", FR, EN)  # evicts en->de
    translator.translate("hi", ES, EN)  # still loaded
    assert FakeOpus.created == [(EN, ES), (EN, DE), (EN, FR)]
    translator.translate("hi", DE, EN)  # reloaded after eviction
    assert FakeOpus.created[-1] == (EN, DE)
    assert len(FakeOpus.created) == 4


@pytest.mark.usefixtures("fake_models")
def test_no_limit_keeps_every_model() -> None:
    translator = Translator(max_loaded_models=None)
    for target in (ES, DE, FR):
        translator.translate("hi", target, EN)
    translator.translate("hi", ES)  # SMaLL-100
    assert len(translator.loaded_models) == 4


@pytest.mark.usefixtures("fake_models")
def test_unload_releases_models_and_reloads_on_demand() -> None:
    translator = Translator()
    translator.translate("hi", ES, EN)
    (model,) = translator.loaded_models
    released = weakref.ref(model)
    del model
    translator.unload()
    assert translator.loaded_models == []
    gc.collect()
    assert released() is None
    assert translator.translate("hi", ES, EN) == "HI:es"
    assert FakeOpus.created == [(EN, ES), (EN, ES)]
    translator.unload()
    translator.unload()


def test_explicit_model_is_not_cached_or_unloaded(translator: Translator) -> None:
    assert translator.loaded_models == []
    translator.unload()
    assert translator.translate("hola", EN) == "HOLA:en"


@pytest.mark.parametrize(
    ("value", "error"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_invalid_max_loaded_models(value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="max_loaded_models"):
        Translator(max_loaded_models=value)  # ty: ignore[invalid-argument-type]


class UnloadingOpus(UpperModel):
    """Unloads its translator in the middle of translating, like a concurrent eviction."""

    translator: Translator

    @classmethod
    def from_pair(cls, source: Language, target: Language) -> UnloadingOpus:
        return cls()

    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        UnloadingOpus.translator.unload()
        return super().predict_batch(texts, target_language, source_language)


def test_eviction_does_not_break_a_call_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(translator_module, "OpusMTModel", UnloadingOpus)
    translator = UnloadingOpus.translator = Translator(max_loaded_models=1)
    assert translator.translate_batch(["a", "b"], ES, EN) == ["A:es", "B:es"]
    assert translator.loaded_models == []


@pytest.mark.usefixtures("fake_models")
def test_concurrent_calls_share_the_bounded_cache() -> None:
    translator = Translator(max_loaded_models=1)
    targets = [ES, DE, FR] * 10

    def translate(target: Language) -> str:
        return translator.translate("hi", target, EN)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(translate, targets))
    assert results == [f"HI:{target}" for target in targets]
    assert len(translator.loaded_models) <= 1
