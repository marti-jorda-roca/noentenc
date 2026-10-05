"""Where model weights are cached, for detection and translation alike.

The cache root is, in order of precedence:

1. the `cache_dir` passed to a model, `LanguageDetector`, `Translator` or `prepare`;
2. the `NOENTENC_CACHE` environment variable;
3. `~/.cache/noentenc`.

Detection weights live directly under the root, translation weights in a Hugging Face
cache under `<root>/hub`.
"""

from __future__ import annotations

import os
from pathlib import Path

CACHE_ENV = "NOENTENC_CACHE"
_TRANSLATION_SUBDIR = "hub"


def cache_root(cache_dir: str | Path | None = None) -> Path:
    """The directory detection weights are cached in."""
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get(CACHE_ENV)
    return Path(env) if env else Path.home() / ".cache" / "noentenc"


def translation_cache(cache_dir: str | Path | None = None) -> Path:
    """The Hugging Face cache translation weights are kept in."""
    return cache_root(cache_dir) / _TRANSLATION_SUBDIR
