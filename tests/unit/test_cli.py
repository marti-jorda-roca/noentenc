"""The `noentenc` command: input, output formats and exit codes."""

import json
import shutil
import sys
import urllib.request
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from noentenc.cli import main
from noentenc.cli._app import cli
from noentenc.language_detection import _download as download_module
from noentenc.language_detection.models.fasttext.model import PRESETS
from tests.unit.language_detection.helpers import FASTTEXT_FIXTURES
from tests.unit.translation.test_translator import (  # noqa: F401 - fixture
    FakeSeq2Seq,
    fake_models,
)

TINY = FASTTEXT_FIXTURES / "tiny-softmax.bin"


@pytest.fixture(autouse=True)
def no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tried to download")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.delenv("NOENTENC_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A cache holding the tiny fastText fixture as the `speed` detection model."""
    root = tmp_path / "models"
    target = root / PRESETS["lid176"].remote.cache_path
    target.parent.mkdir(parents=True)
    shutil.copy(TINY, target)
    return root


def run(*args: str | Path, stdin: str | None = None) -> Result:
    return CliRunner().invoke(cli, [str(arg) for arg in args], input=stdin)


def offline(cache: Path) -> tuple[str, str, Path]:
    return "--offline", "--cache-dir", cache


def test_detect_prints_one_label_per_text_in_order(cache: Path) -> None:
    result = run(
        "detect", *offline(cache), "der hund lief im park", "el perro corrió", "", "42"
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["deu", "spa", "und", "zxx"]


def test_detect_reads_stdin_and_files_line_by_line(cache: Path, tmp_path: Path) -> None:
    lines = "der hund lief im park\r\n\nel perro corrió\n"
    result = run("detect", *offline(cache), stdin=lines)
    assert result.stdout.splitlines() == ["deu", "und", "spa"]
    path = tmp_path / "texts.txt"
    # Bytes, so Windows doesn't turn "\n" into "\r\n" and "\r\n" into "\r\r\n".
    path.write_bytes(lines.encode())
    assert run("detect", *offline(cache), "--file", path).stdout == result.stdout
    from_dash = run("detect", *offline(cache), "--file", "-", stdin=lines)
    assert from_dash.stdout == result.stdout


def test_detect_json_has_status_and_score(cache: Path) -> None:
    result = run(
        "detect", *offline(cache), "--json", "--min-letters", "4", "dog", "el perro"
    )
    short, spanish = map(json.loads, result.stdout.splitlines())
    assert short == {
        "text": "dog",
        "language": "und",
        "status": "insufficient_text",
        "top_language": None,
        "score": None,
    }
    assert spanish["language"] == spanish["top_language"] == "spa"
    assert spanish["status"] == "detected"
    assert 0 < spanish["score"] <= 1


def test_detect_candidates(cache: Path) -> None:
    result = run("detect", *offline(cache), "--candidates", "eng, cat", "el perro")
    assert result.stdout.splitlines() == ["cat"]


def test_detect_rejects_texts_and_a_file_together(cache: Path) -> None:
    result = run("detect", *offline(cache), "--file", "-", "hello", stdin="x\n")
    assert result.exit_code == 2
    assert "not both" in result.stderr


def test_offline_cache_miss_exits_3(tmp_path: Path) -> None:
    result = run("detect", "--offline", "--cache-dir", tmp_path, "hello")
    assert result.exit_code == 3
    assert "noentenc download" in result.stderr
    assert result.stdout == ""


def test_unknown_profile_exits_2(cache: Path) -> None:
    result = run("detect", *offline(cache), "--profile", "fast", "hello")
    assert result.exit_code == 2


@pytest.mark.usefixtures("fake_models")
def test_translate_prints_one_translation_per_text(cache: Path) -> None:
    result = run("translate", *offline(cache), "--from", "es", "--to", "en", "hola", "")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["HOLA:en", ""]


@pytest.mark.usefixtures("fake_models")
def test_translate_keeps_terms_and_patterns(cache: Path) -> None:
    result = run(
        "translate",
        *offline(cache),
        "--from",
        "es",
        "--to",
        "en",
        "--keep",
        "Nike",
        "--keep-regex",
        "SKU-[0-9]+",
        "compra Nike SKU-7",
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == ["COMPRA Nike SKU-7:en"]


@pytest.mark.parametrize("option", [("--keep", " "), ("--keep-regex", "(")])
def test_translate_rejects_bad_keep_options(
    cache: Path, option: tuple[str, str]
) -> None:
    result = run("translate", *offline(cache), "--to", "en", *option, "hola")
    assert result.exit_code == 2
    assert option[0] in result.stderr


@pytest.mark.usefixtures("fake_models")
def test_translate_detects_the_source_by_default(cache: Path) -> None:
    result = run(
        "translate", *offline(cache), "--to", "en", "--json", stdin="el perro corrió\n"
    )
    (record,) = map(json.loads, result.stdout.splitlines())
    assert record["text"] == "el perro corrió"
    assert record["translation"] == "EL PERRO CORRIÓ:en"
    assert record["status"] == "translated"
    assert record["source_language"] == "es"
    assert record["detected_language"] == "spa"
    assert record["model"].startswith("FakeOpus")


@pytest.mark.usefixtures("fake_models")
def test_translate_unknown_source_raise_exits_4(cache: Path) -> None:
    result = run(
        "translate", *offline(cache), "--to", "en", "--unknown-source", "raise", "lol"
    )
    assert result.exit_code == 4
    assert "unknown_source" in result.stderr


@pytest.mark.usefixtures("fake_models")
def test_failed_texts_keep_their_line_and_exit_4(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predict = FakeSeq2Seq.predict_batch

    def fail_on_boom(self: FakeSeq2Seq, texts: list[str], *args: object) -> list[str]:
        if "boom" in texts:
            raise RuntimeError("model broke")
        return predict(self, texts, *args)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(FakeSeq2Seq, "predict_batch", fail_on_boom)
    args = ("translate", *offline(cache), "--from", "es", "--to", "en")
    result = run(*args, stdin="hola\nboom\nadiós\n")
    assert result.exit_code == 4
    assert result.stdout.splitlines() == ["HOLA:en", "boom", "ADIÓS:en"]
    assert "Text 2 failed: RuntimeError: model broke" in result.stderr
    assert "1 of 3 texts failed" in result.stderr
    records = [
        json.loads(line) for line in run(*args, "--json", "boom").stdout.splitlines()
    ]
    assert records[0]["translation"] is None
    assert records[0]["status"] == "failed"
    assert records[0]["error"] == "RuntimeError: model broke"


def test_translate_needs_a_known_target(cache: Path) -> None:
    assert run("translate", *offline(cache), "hola").exit_code == 2
    result = run("translate", *offline(cache), "--to", "xx", "hola")
    assert result.exit_code == 2
    assert "'xx' is not a language" in result.stderr


@pytest.mark.usefixtures("fake_models")
def test_download_prepares_the_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetched: list[str] = []

    def fake_fetch(remote: download_module.RemoteFile, **_kwargs: object) -> Path:
        fetched.append(remote.cache_path)
        return tmp_path / remote.cache_path

    monkeypatch.setattr(download_module, "fetch", fake_fetch)
    result = run(
        "download",
        "--pair",
        "es:en",
        "--pair",
        ":en",
        "--cache-dir",
        tmp_path,
        "--json",
    )
    assert result.exit_code == 0, result.output
    assert fetched == ["fasttext/lid.176.ftz"]
    assert [load["download"] for load in FakeSeq2Seq.loads] == [
        "Xenova/opus-mt-es-en",
        "casawolice/small100-onnx",
    ]
    report = json.loads(result.stdout)
    assert [model["name"] for model in report["models"]] == [
        "FastTextModel(lid176)",
        "FakeOpus(Xenova/opus-mt-es-en)",
        "FakeSmall100(casawolice/small100-onnx)",
    ]
    assert report["files"][0] == str(tmp_path / "fasttext/lid.176.ftz")
    assert "Downloading" in result.stderr


@pytest.mark.usefixtures("fake_models")
def test_download_dry_run_downloads_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(download_module, "fetch", pytest.fail)
    result = run("download", "--pair", "auto:en", "--cache-dir", tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert FakeSeq2Seq.loads == []
    lines = result.stdout.splitlines()
    assert lines[0].split() == [
        "TASK",
        "MODEL",
        "PRECISION",
        "LICENSE",
        "SIZE",
        "CACHED",
    ]
    assert "FastTextModel(lid176)" in lines[1]
    assert "not cached yet" in lines[-1]


@pytest.mark.parametrize(
    "args",
    [("--pair", "es-en"), ("--pair", "es:"), ("--no-detection",), ("--pair", "xx:en")],
)
def test_download_rejects_bad_pairs(args: tuple[str, ...], tmp_path: Path) -> None:
    result = run("download", *args, "--cache-dir", tmp_path, "--dry-run")
    assert result.exit_code == 2


def test_list_shows_every_model_and_its_cache_state(cache: Path) -> None:
    result = run("list", "--json", "--cache-dir", cache)
    records = [json.loads(line) for line in result.stdout.splitlines()]
    models = {model["name"]: model for model in records}
    lid176 = models["FastTextModel(lid176)"]
    assert lid176["profiles"] == ["speed"]
    assert lid176["cached"] is True
    assert lid176["extra"] is None
    assert models["FastTextModel(glotlid)"]["cached"] is False
    assert models["LinguaModel"]["download_bytes"] is None
    small100 = models["SMaLL100Model(casawolice/small100-onnx)"]
    assert small100["profiles"] == ["speed", "balance", "quality"]
    assert small100["extra"] == "translation"
    assert models["M2M100Model(Xenova/m2m100_418M)"]["profiles"] == []
    nllb = [m for m in records if m["name"].startswith("NLLBModel")]
    assert {(m["precision"], tuple(m["profiles"])) for m in nllb} == {
        ("int8", ("balance",)),
        ("fp32", ("quality",)),
    }


def test_list_filters_by_task(cache: Path) -> None:
    result = run("list", "--task", "detection", "--cache-dir", cache)
    header, *rows = result.stdout.splitlines()
    assert header.split()[:2] == ["TASK", "MODEL"]
    assert rows
    assert all(row.startswith("detection") for row in rows)


def test_info_reports_the_cache_without_downloading(cache: Path) -> None:
    repo = cache / "hub" / "models--Xenova--opus-mt-es-en"
    blob = cache / "hub" / "blobs" / "ab" / "abcdef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 1000)
    snapshot = repo / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"y" * 10)
    _link_or_copy(blob, snapshot / "encoder.onnx")
    (cache / "fasttext" / "tmp123.part").write_bytes(b"partial")

    report = json.loads(run("info", "--json", "--cache-dir", cache).stdout)
    assert report["cache_root"] == str(cache)
    assert report["cache_root_from"] == "--cache-dir"
    assert report["translation_cache"] == str(cache / "hub")
    assert report["extras"]["cli"] is True
    assert [(m["task"], m["name"], m["size_bytes"]) for m in report["models"]] == [
        ("detection", "FastTextModel(lid176)", TINY.stat().st_size),
        ("translation", "Xenova/opus-mt-es-en", 1010),
    ]
    assert report["total_bytes"] == TINY.stat().st_size + 1010

    plain = run("info", "--cache-dir", cache).stdout
    assert "FastTextModel(lid176)" in plain
    assert "in 2 models." in plain


def test_info_on_an_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOENTENC_CACHE", str(tmp_path / "env"))
    result = run("info")
    assert result.exit_code == 0
    assert f"{tmp_path / 'env'} (NOENTENC_CACHE)" in result.stdout
    assert "No models cached." in result.stdout


def test_main_names_the_extra_when_click_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "click", None)  # as if it weren't installed
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    assert "uv add 'noentenc[cli]'" in capsys.readouterr().err


def _link_or_copy(target: Path, link: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError:  # Windows without developer mode
        shutil.copy(target, link)
