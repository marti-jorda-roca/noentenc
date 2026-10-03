import pytest

from noentenc.languages import (
    ANY_LANGUAGE,
    AnyLanguage,
    Language,
    LanguageSchema,
    UnsupportedLanguageError,
)
from noentenc.translation.models.m2m100 import M2M100_CODES, M2M100Model
from noentenc.translation.models.nllb import NLLB_CODES, NLLBModel
from noentenc.translation.models.opus_mt import OPUS_MT_PAIRS
from noentenc.translation.models.small100 import SMaLL100Model

EN, ES, FR = Language.ENGLISH, Language.SPANISH, Language.FRENCH


def test_language_parses_codes() -> None:
    assert Language("es") is Language.SPANISH
    assert Language.SPANISH == "es"
    with pytest.raises(ValueError):
        Language("xx")


def test_any_language_is_a_singleton() -> None:
    assert AnyLanguage() is ANY_LANGUAGE
    assert repr(ANY_LANGUAGE) == "ANY_LANGUAGE"


def test_any_language_accepts_everything() -> None:
    schema = LanguageSchema(source=ANY_LANGUAGE, target=ANY_LANGUAGE)
    assert all(schema.supports(language, language) for language in Language)
    assert schema.supports(None, EN)


def test_schema_rejects_unsupported_target() -> None:
    schema = LanguageSchema(source=frozenset({EN, ES}), target=frozenset({EN, ES}))
    with pytest.raises(UnsupportedLanguageError, match="into"):
        schema.validate(EN, FR, "Model")


def test_schema_rejects_unsupported_source() -> None:
    schema = LanguageSchema(source=frozenset({EN, ES}), target=frozenset({EN, ES}))
    with pytest.raises(UnsupportedLanguageError, match="from"):
        schema.validate(FR, ES, "Model")


def test_schema_requires_source_when_ambiguous() -> None:
    schema = LanguageSchema(source=frozenset({EN, ES}), target=frozenset({FR}))
    assert schema.requires_source
    with pytest.raises(UnsupportedLanguageError, match="requires a source"):
        schema.validate(None, FR, "Model")


def test_single_source_language_is_implied() -> None:
    schema = LanguageSchema(source=frozenset({EN}), target=frozenset({ES}))
    schema.validate(None, ES, "Model")
    with pytest.raises(UnsupportedLanguageError):
        schema.validate(None, FR, "Model")


def test_model_schemas() -> None:
    assert M2M100Model.schema.requires_source
    assert not M2M100Model.schema.supports(None, EN)
    assert SMaLL100Model.schema.supports(None, Language.JAPANESE)
    assert SMaLL100Model.schema.source is ANY_LANGUAGE
    assert NLLBModel.schema.supports(Language.ASTURIAN, Language.ZULU)


def test_native_code_maps_match_schemas() -> None:
    assert M2M100Model.schema.target == frozenset(M2M100_CODES)
    assert SMaLL100Model.schema.target == frozenset(M2M100_CODES)
    assert NLLBModel.schema.source == frozenset(NLLB_CODES)
    assert len(set(NLLB_CODES.values())) == len(NLLB_CODES)
    assert M2M100_CODES[Language.NORTHERN_SOTHO] == "ns"
    assert NLLB_CODES[Language.CHINESE] == "zho_Hans"


def test_every_language_is_used_by_a_model() -> None:
    assert set(Language) == set(NLLB_CODES) | set(M2M100_CODES)


def test_opus_pairs_are_directional() -> None:
    assert (EN, ES) in OPUS_MT_PAIRS
    assert (ES, EN) in OPUS_MT_PAIRS
    assert all(source != target for source, target in OPUS_MT_PAIRS)
