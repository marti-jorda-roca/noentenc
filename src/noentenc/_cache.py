"""Where model weights are cached, for detection and translation alike.

The cache root is, in order of precedence:

1. the `cache_dir` passed to a model, `LanguageDetector`, `Translator` or `prepare`;
2. the `NOENTENC_CACHE` environment variable;
3. `~/.cache/noentenc`.

Detection weights live directly under the root. Translation weights live in a Hugging Face
cache under `<root>/hub`, but only when a root was set by (1) or (2). Otherwise they stay in
the Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME` or `~/.cache/huggingface/hub`), as before
`NOENTENC_CACHE` covered translation, so existing downloads are reused.
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


def translation_cache(cache_dir: str | Path | None = None) -> Path | None:
    """The Hugging Face cache for translation weights; None for Hugging Face's default."""
    if cache_dir is None and not os.environ.get(CACHE_ENV):
        return None
    return cache_root(cache_dir) / _TRANSLATION_SUBDIR
