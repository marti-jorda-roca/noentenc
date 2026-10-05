"""Prepare weights into a fresh directory, then detect and translate with the network blocked.

Downloads lid176 (0.9 MB) and Opus-MT en→es (287 MB), so it only runs when
NOENTENC_RUN_INTEGRATION=1.
"""

import os
import socket
from pathlib import Path

import pytest

import noentenc
from noentenc import Language
from noentenc.language_detection import LanguageDetector
from noentenc.translation import Translator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("NOENTENC_RUN_INTEGRATION") != "1",
        reason="downloads weights; set NOENTENC_RUN_INTEGRATION=1",
    ),
]


def test_prepare_then_run_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = noentenc.prepare(
        "speed",
        translation=[(Language.ENGLISH, Language.SPANISH)],
        cache_dir=tmp_path,
    )
    assert all(path.is_file() and tmp_path in path.parents for path in paths)

    def no_network(*_args: object, **_kwargs: object) -> None:
        raise OSError("network access during an offline run")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    detector = LanguageDetector(only_local_files=True, cache_dir=tmp_path)
    translator = Translator(only_local_files=True, cache_dir=tmp_path)
    assert detector.detect("Bon dia! Com estàs?") == "cat"
    text = translator.translate(
        "The weather is nice today.", Language.SPANISH, Language.ENGLISH
    )
    assert "tiempo" in text.lower()
    # A pair that wasn't prepared fails at once instead of downloading.
    with pytest.raises(FileNotFoundError, match=r"noentenc\.prepare"):
        translator.translate("Hello", Language.GERMAN, Language.ENGLISH)
