"""Plans, support queries and licence/size constraints: nothing is loaded or downloaded."""

import urllib.request
from pathlib import Path

import pytest

import noentenc
from noentenc import (
    ANY_LANGUAGE,
    Language,
    ModelConstraintError,
    UnsupportedLanguageError,
)
from noentenc._catalog import OPUS_MT_LICENSES, TRANSLATION_FILE_SIZES
from noentenc.language_detection import LanguageDetector
from noentenc.language_detection import _download as download_module
from noentenc.language_detection import base as detector_module
from noentenc.translation import OpusMTModel, SMaLL100Model, Translator
from noentenc.translation import _routing as routing_module
from noentenc.translation.models import _hub
from noentenc.translation.models.opus_mt import OPUS_MT_REVISIONS
from tests.unit.language_detection.helpers import FASTTEXT_FIXTURES
from tests.unit.translation.test_auto import ScriptedDetector
from tests.unit.translation.test_translator import (  # noqa: F401 - fixture
    FakeNLLB,
    FakeOpus,
    FakeSeq2Seq,
    FakeSmall100,
    UpperModel,
    fake_models,
)

EN, ES, JA, CA = Language.ENGLISH, Language.SPANISH, Language.JAPANESE, Language.CATALAN
OPEN = ["MIT", "Apache-2.0", "CC-BY-4.0"]


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tried to download")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(_hub, "require", refuse)
    monkeypatch.setenv("NOENTENC_CACHE", str(tmp_path / "empty"))


def test_translation_plan_reports_the_model_and_its_costs() -> None:
    (model,) = Translator().plan("es", "en").models
    revision = OPUS_MT_REVISIONS[EN, ES]
    sizes = TRANSLATION_FILE_SIZES[f"Xenova/opus-mt-en-es@{revision}"]
    assert model.name == "OpusMTModel(Xenova/opus-mt-en-es)"
    assert model.precision == str(OpusMTModel.default_precision)
    assert model.files == tuple(OpusMTModel.filenames())
    assert model.download_bytes == sum(sizes[name] for name in model.files)
    assert model.license == OPUS_MT_LICENSES["Xenova/opus-mt-en-es"]
    assert model.memory_basis == "measured on Opus-MT en→es"
    assert not model.cached


@pytest.mark.parametrize(
    ("profile", "source", "target", "name", "licence"),
    [
        ("speed", JA, CA, "SMaLL100Model(casawolice/small100-onnx)", "MIT"),
        ("speed", None, EN, "SMaLL100Model(casawolice/small100-onnx)", "MIT"),
        (
            "balance",
            JA,
            CA,
            "NLLBModel(Xenova/nllb-200-distilled-600M)",
            "CC-BY-NC-4.0",
        ),
        ("quality", EN, ES, "OpusMTModel(Xenova/opus-mt-en-es)", "Apache-2.0"),
    ],
)
def test_plans_name_what_each_profile_picks(
    profile: str, source: Language | None, target: Language, name: str, licence: str
) -> None:
    (model,) = Translator(profile).plan(target, source).models
    assert (model.name, model.license) == (name, licence)


def test_quality_uses_fp32_nllb_with_measured_memory() -> None:
    (model,) = Translator("quality").plan(CA, JA).models
    assert model.precision == "fp32"
    assert model.memory_basis == "measured"
    assert model.download_bytes > 3_000_000_000


def test_plans_agree_with_what_the_translator_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = [
        ("speed", ES, EN),
        ("speed", CA, JA),
        ("balance", CA, JA),
        ("speed", EN, None),
    ]
    planned = [
        Translator(profile).plan(target, source).models[0].name.split("(")[1][:-1]
        for profile, target, source in pairs
    ]
    FakeSeq2Seq.loads = []
    monkeypatch.setattr(routing_module, "OpusMTModel", FakeOpus)
    monkeypatch.setattr(routing_module, "SMaLL100Model", FakeSmall100)
    monkeypatch.setattr(routing_module, "NLLBModel", FakeNLLB)
    for profile, target, source in pairs:
        Translator(profile).translate("hi", target, source)
    assert [load["model"] for load in FakeSeq2Seq.loads] == planned


def test_plan_reports_cache_state(tmp_path: Path) -> None:
    revision = OPUS_MT_REVISIONS[EN, ES]
    snapshot = (
        tmp_path / "hub" / "models--Xenova--opus-mt-en-es" / "snapshots" / revision
    )
    translator = Translator(cache_dir=tmp_path)
    assert not translator.plan(ES, EN).models[0].cached
    for name in OpusMTModel.filenames():
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(b"")
    plan = translator.plan(ES, EN)
    assert plan.models[0].cached
    assert plan.missing_bytes == 0
    assert plan.download_bytes > 0


def test_auto_plan_holds_the_detector_and_every_route() -> None:
    plan = Translator().plan(EN, "auto")
    detector, *models = plan.models
    assert detector.task == "detection"
    assert detector.name == "FastTextModel(lid176)"
    assert detector.memory_basis.startswith("measured")
    assert sum(m.name.startswith("OpusMTModel") for m in models) == 25
    assert sum(m.name.startswith("SMaLL100Model") for m in models) == 1
    assert plan.download_bytes == sum(m.download_bytes for m in plan.models)
    assert plan.memory_bytes == sum(m.memory_bytes for m in plan.models)


class UnloadableFastText(detector_module.FastTextModel):
    """Has the presets' files but fails if a profile tries to load it."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pytest.fail("loaded a detection model")


class TinyFastText(detector_module.FastTextModel):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__(FASTTEXT_FIXTURES / "tiny-softmax.bin")


def test_detector_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "FastTextModel", TinyFastText)
    (model,) = LanguageDetector("balance").plan().models
    assert model.name == "FastTextModel(openlid-v3)"
    assert model.license == "GPL-3.0"
    assert model.download_bytes > 1_000_000_000
    assert not model.cached
    with pytest.raises(ValueError, match="profile models"):
        LanguageDetector(TinyFastText()).plan()


@pytest.fixture
def no_fasttext(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "FastTextModel", UnloadableFastText)


@pytest.mark.usefixtures("no_fasttext")
def test_detection_constraints_fail_before_loading() -> None:
    with pytest.raises(ModelConstraintError, match="GPL-3.0 is not allowed"):
        LanguageDetector("balance", allowed_licenses=["Apache-2.0"])
    with pytest.raises(ModelConstraintError, match="over max_download_bytes"):
        LanguageDetector("quality", max_download_bytes=100 << 20)


@pytest.mark.usefixtures("fake_models")
def test_licence_constraints_skip_to_the_next_model() -> None:
    translator = Translator("balance", allowed_licenses=OPEN)
    translator.translate("hi", CA, JA)
    assert FakeNLLB.created == []
    assert FakeSmall100.created == 1
    (model,) = translator.plan(CA, JA).models
    assert model.license == "MIT"


@pytest.mark.usefixtures("fake_models")
def test_constraint_failures_happen_before_any_download() -> None:
    translator = Translator(max_download_bytes=100 << 20)
    with pytest.raises(ModelConstraintError, match="over max_download_bytes") as error:
        translator.translate("hi", ES, EN)
    assert "(Xenova/opus-mt-en-es): 290 MB" in str(error.value)
    assert "(casawolice/small100-onnx): 586 MB" in str(error.value)
    assert FakeSeq2Seq.loads == []
    assert not translator.supports(ES, EN)
    with pytest.raises(ModelConstraintError):
        translator.plan(ES, EN)
    with pytest.raises(ModelConstraintError):
        noentenc.prepare(
            detection=False, translation=[("en", "es")], max_download_bytes=100 << 20
        )


@pytest.mark.usefixtures("fake_models")
def test_auto_routing_respects_constraints() -> None:
    detector = LanguageDetector(ScriptedDetector(), min_score=0.5)
    translator = Translator(
        "balance", allowed_licenses=["Apache-2.0", "CC-BY-4.0"], detector=detector
    )
    results = translator.translate_batch(
        ["Buenos días", "Bon dia a tothom"], EN, "auto", detailed=True
    )
    assert results[0].model == "FakeOpus(upper)"
    # Catalan has no Opus-MT model, NLLB is non-commercial and SMaLL-100 is MIT.
    assert results[1].status == "unsupported_source"
    assert FakeNLLB.created == []
    assert FakeSmall100.created == 0


def test_supports_and_supported_languages() -> None:
    speed = Translator()
    assert speed.supports(ES, EN)
    assert speed.supports(EN)  # without a source: SMaLL-100
    assert speed.supports(EN, "auto")
    assert not speed.supports(Language.ACEHNESE, EN)  # only NLLB writes Acehnese
    assert Translator("balance").supports(Language.ACEHNESE, EN)
    schema = speed.supported_languages()
    assert schema.source is ANY_LANGUAGE
    assert schema.target == SMaLL100Model.schema.target
    restricted = Translator(allowed_licenses=["Apache-2.0", "CC-BY-4.0"])
    pairs = restricted.supported_languages()
    assert isinstance(pairs.source, frozenset)
    assert isinstance(pairs.target, frozenset)
    assert ES in pairs.target and Language.ACEHNESE not in pairs.target
    assert not restricted.supports(EN)


def test_explicit_models_answer_from_their_schema() -> None:
    translator = Translator(UpperModel())
    assert translator.supports(ES, EN)
    assert translator.supported_languages() == UpperModel.schema
    with pytest.raises(ValueError, match="profile models"):
        translator.plan(ES)
    with pytest.raises(ValueError, match="a profile loads"):
        Translator(UpperModel(), allowed_licenses=OPEN)


def test_module_plan_matches_prepare_arguments() -> None:
    plan = noentenc.plan("speed", translation=[("en", "es"), (None, "en")])
    assert [m.name for m in plan.models] == [
        "FastTextModel(lid176)",
        "OpusMTModel(Xenova/opus-mt-en-es)",
        "SMaLL100Model(casawolice/small100-onnx)",
    ]
    assert plan.licenses == {"CC-BY-SA-3.0", "Apache-2.0", "MIT"}
    assert plan.missing_bytes == plan.download_bytes
    with pytest.raises(ModelConstraintError, match="CC-BY-SA-3.0 is not allowed"):
        noentenc.plan(allowed_licenses=OPEN)
    with pytest.raises(UnsupportedLanguageError):
        noentenc.plan(detection=False, translation=[("en", "ace")])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"allowed_licenses": "MIT"}, TypeError),
        ({"max_download_bytes": -1}, ValueError),
        ({"max_download_bytes": 1.5}, ValueError),
    ],
)
def test_invalid_constraints(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        Translator(**kwargs)  # ty: ignore[invalid-argument-type]


def test_prepare_checks_detection_constraints_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(download_module, "fetch", pytest.fail)
    with pytest.raises(ModelConstraintError):
        noentenc.prepare("balance", allowed_licenses=["Apache-2.0"])


def test_download_limit_falls_back_to_the_smaller_opus_export() -> None:
    (default,) = Translator().plan(ES, EN).models
    (small,) = Translator(max_download_bytes=200 << 20).plan(ES, EN).models
    assert default.precision == "q4"
    assert small.name == "OpusMTModel(Xenova/opus-mt-en-es)"
    assert small.precision == "int8"
    assert small.download_bytes < 200 << 20 < default.download_bytes
    assert small.memory_bytes < default.memory_bytes
