"""Cache locations, offline loading and `noentenc.prepare`."""

import shutil
import urllib.request
from pathlib import Path

import pytest

import noentenc
from noentenc import Language
from noentenc._cache import cache_root, translation_cache
from noentenc.language_detection import FastTextModel, LanguageDetector
from noentenc.language_detection import _download as download_module
from noentenc.language_detection.models.fasttext.model import PRESETS
from noentenc.translation import OpusMTModel, SMaLL100Model, Translator
from noentenc.translation.models import _hub
from tests.unit.language_detection.helpers import FASTTEXT_FIXTURES
from tests.unit.translation.test_translator import (  # noqa: F401 - fixture
    FakeSeq2Seq,
    UpperModel,
    fake_models,
)

EN, ES = Language.ENGLISH, Language.SPANISH
TINY = FASTTEXT_FIXTURES / "tiny-softmax.bin"


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tried to download")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("NOENTENC_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    return tmp_path


def test_cache_root_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert cache_root() == tmp_path / "home" / ".cache" / "noentenc"
    assert translation_cache() is None  # the Hugging Face default cache
    monkeypatch.setenv("NOENTENC_CACHE", str(tmp_path / "env"))
    assert cache_root() == tmp_path / "env"
    assert translation_cache() == tmp_path / "env" / "hub"
    assert cache_root(tmp_path / "arg") == tmp_path / "arg"
    assert translation_cache(str(tmp_path / "arg")) == tmp_path / "arg" / "hub"


def _prepared_lid176(root: Path) -> None:
    target = root / PRESETS["lid176"].remote.cache_path
    target.parent.mkdir(parents=True)
    shutil.copy(TINY, target)


@pytest.mark.usefixtures("no_network")
def test_detector_reads_a_prepared_cache_offline(tmp_path: Path) -> None:
    _prepared_lid176(tmp_path / "models")
    detector = LanguageDetector(only_local_files=True, cache_dir=tmp_path / "models")
    assert detector.detect("der hund lief im park") == "deu"


@pytest.mark.usefixtures("no_network")
def test_detector_offline_miss_fails_at_once(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"noentenc\.prepare"):
        LanguageDetector("balance", only_local_files=True, cache_dir=tmp_path)


def test_offline_options_are_for_profiles_only() -> None:
    with pytest.raises(ValueError, match="a profile loads"):
        LanguageDetector(FastTextModel(TINY), only_local_files=True)
    with pytest.raises(ValueError, match="a profile loads"):
        Translator(UpperModel(), cache_dir="/models")


@pytest.mark.usefixtures("fake_models")
def test_translator_loads_models_with_its_offline_options() -> None:
    translator = Translator(only_local_files=True, cache_dir="/models")
    translator.translate("hi", ES, EN)
    translator.translate("hi", ES)
    assert [
        (load["class"], load["only_local_files"], load["cache_dir"])
        for load in FakeSeq2Seq.loads
    ] == [
        ("FakeOpus", True, "/models"),
        ("FakeSmall100", True, "/models"),
    ]


class RecordingHub:
    def __init__(self, missing: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.missing = missing

    def hf_hub_download(self, repo: str, filename: str, **kwargs: object) -> str:
        self.calls.append({"repo": repo, "filename": filename, **kwargs})
        if self.missing and kwargs["local_files_only"]:
            raise FileNotFoundError("LocalEntryNotFoundError")
        return f"/hub/{repo}/{filename}"


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> RecordingHub:
    recording = RecordingHub()
    monkeypatch.setattr(_hub, "require", lambda _module, _extra: recording)
    return recording


def test_download_fetches_files_without_loading(
    hub: RecordingHub, tmp_path: Path
) -> None:
    paths = SMaLL100Model.download(cache_dir=tmp_path, force=True)
    assert set(paths) == set(SMaLL100Model.filenames())
    call = hub.calls[0]
    assert call["repo"] == SMaLL100Model.default_model
    assert call["revision"] == SMaLL100Model.default_revision
    assert call["cache_dir"] == tmp_path / "hub"
    assert call["force_download"] is True
    assert call["local_files_only"] is False
    assert "onnx/encoder_model_quantized.onnx" in OpusMTModel.filenames("int8")


def test_translation_offline_miss_names_prepare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording = RecordingHub(missing=True)
    monkeypatch.setattr(_hub, "require", lambda _module, _extra: recording)
    with pytest.raises(FileNotFoundError, match=r"noentenc\.prepare"):
        _hub.resolve_files(
            "Xenova/opus-mt-en-es",
            ["config.json"],
            only_local_files=True,
            cache_dir=tmp_path,
        )


@pytest.mark.usefixtures("fake_models")
def test_prepare_downloads_what_the_profile_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetched: list[tuple[str, object, object, object]] = []

    def fake_fetch(remote: download_module.RemoteFile, **kwargs: object) -> Path:
        fetched.append(
            (remote.cache_path, kwargs["cache_dir"], kwargs["force"], kwargs["verify"])
        )
        return tmp_path / remote.cache_path

    monkeypatch.setattr(download_module, "fetch", fake_fetch)
    paths = noentenc.prepare(
        "speed",
        translation=[("en", "es"), (EN, ES), (None, "en")],
        cache_dir=tmp_path,
    )
    assert fetched == [("fasttext/lid.176.ftz", tmp_path, False, True)]
    downloads = [load for load in FakeSeq2Seq.loads if "download" in load]
    assert [(d["class"], d["download"], d["cache_dir"]) for d in downloads] == [
        ("FakeOpus", "Xenova/opus-mt-en-es", tmp_path),
        ("FakeSmall100", SMaLL100Model.default_model, tmp_path),
    ]
    assert not [load for load in FakeSeq2Seq.loads if "download" not in load]
    assert paths[0] == tmp_path / "fasttext/lid.176.ftz"
    assert len(paths) == 3


@pytest.mark.usefixtures("fake_models")
def test_prepare_without_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download_module, "fetch", pytest.fail)
    noentenc.prepare("quality", detection=False, translation=[("ja", "ca")])
    (download,) = FakeSeq2Seq.loads
    assert download["class"] == "FakeNLLB"


def test_prepare_rejects_unknown_profiles_and_pairs() -> None:
    with pytest.raises(ValueError, match="'fast' is not a valid Profile"):
        noentenc.prepare("fast")
    with pytest.raises(ValueError, match="'xx' is not a language"):
        noentenc.prepare(detection=False, translation=[("xx", "en")])
