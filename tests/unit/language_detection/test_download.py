import hashlib
from pathlib import Path

import pytest

from noentenc.language_detection._download import RemoteFile, cache_dir, fetch


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("NOENTENC_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _source(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "weights.bin"
    src.write_bytes(b"weights")
    return src, hashlib.sha256(b"weights").hexdigest()


def test_fetch_downloads_once_and_caches(tmp_path: Path) -> None:
    src, sha = _source(tmp_path)
    remote = RemoteFile(url=src.as_uri(), cache_path="x/weights.bin", sha256=sha)
    path = fetch(remote)
    assert path == cache_dir() / "x/weights.bin" and path.read_bytes() == b"weights"
    src.unlink()
    assert fetch(remote) == path  # served from cache


def test_only_local_files_does_not_download(tmp_path: Path) -> None:
    src, _ = _source(tmp_path)
    with pytest.raises(FileNotFoundError, match="only_local_files"):
        fetch(
            RemoteFile(url=src.as_uri(), cache_path="weights.bin"),
            only_local_files=True,
        )


def test_checksum_mismatch_leaves_nothing_behind(tmp_path: Path) -> None:
    src, _ = _source(tmp_path)
    with pytest.raises(OSError, match="sha256 mismatch"):
        fetch(RemoteFile(url=src.as_uri(), cache_path="weights.bin", sha256="0" * 64))
    assert list(cache_dir().iterdir()) == []


def test_huggingface_urls_are_pinned() -> None:
    remote = RemoteFile.huggingface("org/repo", "onnx/model.onnx", revision="abc123")
    assert (
        remote.url == "https://huggingface.co/org/repo/resolve/abc123/onnx/model.onnx"
    )
    assert remote.cache_path == "huggingface/org/repo/abc123/onnx/model.onnx"
