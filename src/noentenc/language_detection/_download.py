"""Minimal cached downloader for model weights (no ``huggingface_hub`` dependency)."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CACHE_ENV = "NOENTENC_CACHE"
_CHUNK = 1 << 20


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


def cache_dir() -> Path:
    env = os.environ.get(CACHE_ENV)
    return Path(env) if env else Path.home() / ".cache" / "noentenc"


def fetch(remote: RemoteFile, *, only_local_files: bool = False) -> Path:
    """Return the local path of ``remote``, downloading it first unless it is already cached."""
    target = cache_dir() / remote.cache_path
    if target.exists():
        return target
    if only_local_files:
        raise FileNotFoundError(
            f"{target} is not cached and only_local_files=True (source: {remote.url})"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "noentenc"}
    token = os.environ.get("HF_TOKEN")
    if token and remote.url.startswith("https://huggingface.co/"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(remote.url, headers=headers)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(
        dir=target.parent, delete=False, suffix=".part"
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(request) as response:  # nosec B310 - https preset URLs
                while chunk := response.read(_CHUNK):
                    digest.update(chunk)
                    tmp.write(chunk)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    if remote.sha256 and digest.hexdigest() != remote.sha256:
        tmp_path.unlink(missing_ok=True)
        raise OSError(
            f"sha256 mismatch for {remote.url}: expected {remote.sha256}, got {digest.hexdigest()}"
        )
    os.replace(tmp_path, target)
    return target
