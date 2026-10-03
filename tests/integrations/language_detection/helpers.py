"""Shared helpers for the language-detection integration tests."""

import os

import pytest

from noentenc.language_detection._download import RemoteFile, cache_dir

RUN_INTEGRATION = os.environ.get("NOENTENC_RUN_INTEGRATION") == "1"


def skip_unless_available(*remotes: RemoteFile) -> None:
    """Integration tests run when NOENTENC_RUN_INTEGRATION=1 or the weights are already cached."""
    if RUN_INTEGRATION:
        return
    missing = [r.url for r in remotes if not (cache_dir() / r.cache_path).exists()]
    if missing:
        pytest.skip(
            f"model weights not cached ({', '.join(missing)}); set NOENTENC_RUN_INTEGRATION=1"
        )
