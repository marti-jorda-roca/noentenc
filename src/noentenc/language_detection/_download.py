"""Minimal cached downloader for model weights (no ``huggingface_hub`` dependency).

Downloads show a tqdm progress bar (``TQDM_DISABLE=1`` hides it), give up on a connection
that stays silent for ``NOENTENC_DOWNLOAD_TIMEOUT`` seconds (30 by default), and retry
transient failures, resuming where the last attempt stopped when the server allows it.
"""

from __future__ import annotations

import hashlib
import os
import socket
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

from tqdm import tqdm

from noentenc._cache import cache_root

if TYPE_CHECKING:
    from http.client import HTTPResponse

TIMEOUT_ENV = "NOENTENC_DOWNLOAD_TIMEOUT"
DEFAULT_TIMEOUT = 30.0
# Attempts per file, and the wait before the second one (doubling after each failure).
ATTEMPTS = 4
FIRST_BACKOFF = 1.0
_CHUNK = 1 << 20
_PARTIAL_CONTENT = 206
_RANGE_NOT_SATISFIABLE = 416
_TOO_MANY_REQUESTS = 429
_SERVER_ERROR = 500


@dataclass(frozen=True)
class RemoteFile:
    """A file we can fetch over HTTPS, cached under ``<cache>/<cache_path>``."""

    url: str
    cache_path: str
    sha256: str | None = None

    @classmethod
    def huggingface(
        cls, repo: str, filename: str, revision: str, sha256: str | None = None
    ) -> RemoteFile:
        return cls(
            url=f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
            cache_path=f"huggingface/{repo}/{revision}/{filename}",
            sha256=sha256,
        )


def is_cached(remote: RemoteFile, cache_dir: str | Path | None = None) -> bool:
    return (cache_root(cache_dir) / remote.cache_path).is_file()


def fetch(
    remote: RemoteFile,
    *,
    only_local_files: bool = False,
    cache_dir: str | Path | None = None,
    force: bool = False,
    verify: bool = False,
) -> Path:
    """Return the local path of ``remote``, downloading it first unless it is already cached.

    ``force`` downloads it again. ``verify`` checks a cached file against ``remote.sha256``
    and downloads it again if it doesn't match.
    """
    target = cache_root(cache_dir) / remote.cache_path
    if (
        target.exists()
        and not force
        and (not verify or _matches(target, remote.sha256))
    ):
        return target
    if only_local_files:
        raise FileNotFoundError(
            f"{target} is not cached and only_local_files=True. Download it first with "
            "noentenc.prepare(...) using the same cache directory, or copy a prepared "
            f"cache there (source: {remote.url})"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, delete=False, suffix=".part"
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            digest = _download(remote, tmp)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    if remote.sha256 and digest != remote.sha256:
        tmp_path.unlink(missing_ok=True)
        raise OSError(
            f"sha256 mismatch for {remote.url}: expected {remote.sha256}, got {digest}"
        )
    os.replace(tmp_path, target)
    return target


class _Partial:
    """A download in progress: the file so far, its running hash and the progress bar."""

    def __init__(self, out: IO[bytes], progress: tqdm) -> None:
        self.out = out
        self.progress = progress
        self.digest = hashlib.sha256()

    @property
    def size(self) -> int:
        return self.out.tell()

    def write(self, chunk: bytes) -> None:
        self.digest.update(chunk)
        self.out.write(chunk)
        self.progress.update(len(chunk))

    def restart(self) -> None:
        self.out.seek(0)
        self.out.truncate()
        self.digest = hashlib.sha256()
        self.progress.reset()


def _download(remote: RemoteFile, out: IO[bytes]) -> str:
    """Write ``remote`` to ``out``, retrying transient failures; return its sha256."""
    backoff = FIRST_BACKOFF
    with tqdm(
        desc=remote.cache_path.rsplit("/", 1)[-1],
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as progress:
        partial = _Partial(out, progress)
        for attempt in range(1, ATTEMPTS + 1):
            try:
                _transfer(remote, partial)
                return partial.digest.hexdigest()
            except OSError as error:  # URLError and socket timeouts are OSErrors
                if attempt == ATTEMPTS or not _is_transient(error):
                    raise OSError(
                        f"downloading {remote.url} failed after {attempt} attempt(s): "
                        f"{error}. Check the network, or raise {TIMEOUT_ENV} "
                        f"(now {_timeout():g} s) for slow links."
                    ) from error
                time.sleep(backoff)
                backoff *= 2
    raise AssertionError("unreachable")  # pragma: no cover


def _transfer(remote: RemoteFile, partial: _Partial) -> None:
    """One attempt, continuing from what earlier attempts wrote if the server allows it."""
    done = partial.size
    try:
        response = _open(remote, done)
    except urllib.error.HTTPError as error:
        if error.code != _RANGE_NOT_SATISFIABLE or not done:
            raise
        # The partial file doesn't fit the remote one any more; start over.
        partial.restart()
        response = _open(remote, 0)
    with response:
        if done and response.status != _PARTIAL_CONTENT:
            partial.restart()
        length = response.headers.get("Content-Length")
        if length is not None:
            partial.progress.total = partial.size + int(length)
            partial.progress.refresh()
        while chunk := response.read(_CHUNK):
            partial.write(chunk)
    partial.out.flush()


def _open(remote: RemoteFile, start: int) -> HTTPResponse:
    headers = {"User-Agent": "noentenc"}
    token = os.environ.get("HF_TOKEN")
    if token and remote.url.startswith("https://huggingface.co/"):
        headers["Authorization"] = f"Bearer {token}"
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(remote.url, headers=headers)
    return urllib.request.urlopen(request, timeout=_timeout())  # nosec B310 - https preset URLs


def _is_transient(error: BaseException) -> bool:
    """Timeouts, dropped connections, rate limits and server errors; not 404s."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code == _TOO_MANY_REQUESTS or error.code >= _SERVER_ERROR
    if isinstance(error, urllib.error.URLError):
        return isinstance(error.reason, OSError)
    return isinstance(error, TimeoutError | socket.timeout | ConnectionError)


def _timeout() -> float:
    value = os.environ.get(TIMEOUT_ENV)
    return float(value) if value else DEFAULT_TIMEOUT


def _matches(path: Path, sha256: str | None) -> bool:
    if sha256 is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest() == sha256
