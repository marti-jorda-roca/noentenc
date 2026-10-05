import hashlib
import io
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from pathlib import Path

import pytest

from noentenc._cache import cache_root
from noentenc.language_detection import _download as download_module
from noentenc.language_detection._download import RemoteFile, fetch


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
    assert path == cache_root() / "x/weights.bin" and path.read_bytes() == b"weights"
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
    assert list(cache_root().iterdir()) == []


def test_huggingface_urls_are_pinned() -> None:
    remote = RemoteFile.huggingface("org/repo", "onnx/model.onnx", revision="abc123")
    assert (
        remote.url == "https://huggingface.co/org/repo/resolve/abc123/onnx/model.onnx"
    )
    assert remote.cache_path == "huggingface/org/repo/abc123/onnx/model.onnx"


def test_only_local_files_error_names_prepare(tmp_path: Path) -> None:
    src, _ = _source(tmp_path)
    remote = RemoteFile(url=src.as_uri(), cache_path="weights.bin")
    with pytest.raises(FileNotFoundError, match=r"noentenc\.prepare"):
        fetch(remote, only_local_files=True)


def test_cache_dir_argument_overrides_the_environment(tmp_path: Path) -> None:
    src, sha = _source(tmp_path)
    remote = RemoteFile(url=src.as_uri(), cache_path="x/weights.bin", sha256=sha)
    path = fetch(remote, cache_dir=tmp_path / "other")
    assert path == tmp_path / "other" / "x" / "weights.bin"
    assert not (cache_root() / "x").exists()


def test_force_and_verify_replace_a_corrupt_file(tmp_path: Path) -> None:
    src, sha = _source(tmp_path)
    remote = RemoteFile(url=src.as_uri(), cache_path="weights.bin", sha256=sha)
    path = fetch(remote)
    path.write_bytes(b"corrupt")
    assert fetch(remote).read_bytes() == b"corrupt"  # trusted without verify
    assert fetch(remote, verify=True).read_bytes() == b"weights"
    path.write_bytes(b"corrupt")
    assert fetch(remote, force=True).read_bytes() == b"weights"


class FakeResponse(io.BytesIO):
    def __init__(
        self, body: bytes, status: int = 200, fail_after: int | None = None
    ) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}
        self._fail_after = fail_after

    def read(self, size: int | None = -1) -> bytes:
        """Like a socket that stalls after `fail_after` bytes."""
        if self._fail_after is None:
            return super().read(size)
        left = self._fail_after - self.tell()
        if left <= 0:
            raise TimeoutError("the read operation timed out")
        return super().read(left if size is None or size < 0 else min(size, left))


class FakeServer:
    """Serves `body`, scripted per request: a response factory or an exception."""

    def __init__(self, body: bytes, script: list[object]) -> None:
        self.body = body
        self.script = script
        self.requests: list[tuple[dict[str, str], float]] = []

    def urlopen(self, request: urllib.request.Request, timeout: float) -> object:
        headers = dict(request.header_items())
        self.requests.append((headers, timeout))
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        assert callable(step)
        return step(self, headers)


def _whole(server: FakeServer, _headers: dict[str, str]) -> FakeResponse:
    return FakeResponse(server.body)


def _cut_at(n: int) -> Callable[[FakeServer, dict[str, str]], FakeResponse]:
    return lambda server, _headers: FakeResponse(server.body, fail_after=n)


def _rest(server: FakeServer, headers: dict[str, str]) -> FakeResponse:
    start = int(headers["Range"].removeprefix("bytes=").removesuffix("-"))
    return FakeResponse(server.body[start:], status=206)


BODY = bytes(range(256)) * 64
SHA = hashlib.sha256(BODY).hexdigest()
REMOTE = RemoteFile(url="https://example.com/w.bin", cache_path="w.bin", sha256=SHA)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(download_module.time, "sleep", slept.append)
    return slept


def _serve(monkeypatch: pytest.MonkeyPatch, script: list[object]) -> FakeServer:
    server = FakeServer(BODY, script)
    monkeypatch.setattr(download_module.urllib.request, "urlopen", server.urlopen)
    return server


def test_transient_failures_are_retried_with_backoff(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    server = _serve(
        monkeypatch,
        [
            urllib.error.URLError(TimeoutError("timed out")),
            urllib.error.HTTPError(REMOTE.url, 503, "unavailable", Message(), None),
            _whole,
        ],
    )
    assert fetch(REMOTE).read_bytes() == BODY
    assert sleeps == [1.0, 2.0]
    assert len(server.requests) == 3


def test_permanent_failures_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _serve(
        monkeypatch,
        [urllib.error.HTTPError(REMOTE.url, 404, "not found", Message(), None)],
    )
    with pytest.raises(OSError, match="failed after 1 attempt"):
        fetch(REMOTE)
    assert sleeps == []
    assert list(cache_root().iterdir()) == []


def test_retries_are_bounded(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _serve(monkeypatch, [ConnectionResetError("reset")] * 4)
    with pytest.raises(OSError, match="failed after 4 attempt"):
        fetch(REMOTE)
    assert len(sleeps) == 3
    assert list(cache_root().iterdir()) == []


def test_a_stalled_download_resumes_where_it_stopped(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    server = _serve(monkeypatch, [_cut_at(5000), _rest])
    assert fetch(REMOTE).read_bytes() == BODY
    assert "Range" not in server.requests[0][0]
    assert server.requests[1][0]["Range"] == "bytes=5000-"
    assert sleeps == [1.0]


def test_a_server_without_ranges_restarts_the_download(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    server = _serve(monkeypatch, [_cut_at(5000), _whole])
    assert fetch(REMOTE).read_bytes() == BODY  # the checksum covers the restart
    assert server.requests[1][0]["Range"] == "bytes=5000-"
    assert len(sleeps) == 1


def test_timeout_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    server = _serve(monkeypatch, [_whole])
    monkeypatch.setenv("NOENTENC_DOWNLOAD_TIMEOUT", "2.5")
    fetch(REMOTE)
    assert server.requests[0][1] == 2.5
    assert sleeps == []


def test_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _serve(monkeypatch, [_whole])
    monkeypatch.delenv("NOENTENC_DOWNLOAD_TIMEOUT", raising=False)
    fetch(REMOTE)
    assert server.requests[0][1] == download_module.DEFAULT_TIMEOUT


def test_progress_bar_tracks_the_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _serve(monkeypatch, [_whole])
    fetch(REMOTE)
    assert "w.bin" in capsys.readouterr().err
