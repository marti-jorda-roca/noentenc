"""Smoke tests for the backends that wrap third-party detectors (skipped if the extra is missing)."""

import pytest

from noentenc.language_detection.labels import valid_iso639_3_codes
from noentenc.language_detection.models.base import BaseModel

TEXTS = [
    "Bon dia a tothom, com esteu? Avui fa molt bon temps a Barcelona.",
    "",
    "The weather is lovely today.",
]


def _build(name: str) -> BaseModel:
    if name == "lingua":
        pytest.importorskip("lingua")
        from noentenc.language_detection.models.lingua_model import (
            LinguaModel,
        )

        return LinguaModel()
    if name == "cld3":
        pytest.importorskip("gcld3")
        from noentenc.language_detection.models.cld3_model import (
            Cld3Model,
        )

        return Cld3Model()
    pytest.importorskip("heliport")
    from noentenc.language_detection.models.heliport_model import (
        HeliportModel,
    )

    return HeliportModel()


@pytest.mark.parametrize("name", ["lingua", "cld3", "heliport"])
def test_wrapper_detects_and_normalises(name: str) -> None:
    model = _build(name)
    assert model.predict_batch(TEXTS) == ["cat", "und", "eng"]
    assert model.predict(TEXTS[0]) == "cat"
    scores = model.predict_score(TEXTS[0], top_k=2)
    assert 1 <= len(scores) <= 2 and next(iter(scores)) == "cat"
    assert set(model.labels) <= valid_iso639_3_codes()


def test_lingua_language_restriction() -> None:
    pytest.importorskip("lingua")
    from noentenc.language_detection.models.lingua_model import (
        LinguaModel,
    )

    model = LinguaModel(languages=["spa", "cat"])
    assert sorted(model.labels) == ["cat", "spa"]
    assert model.predict("The weather is lovely today.") in {"cat", "spa"}


def test_missing_extra_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from noentenc.language_detection.models.lingua_model import (
        LinguaModel,
    )

    monkeypatch.setitem(sys.modules, "lingua", None)
    with pytest.raises(ImportError, match=r"noentenc\[lingua\]"):
        LinguaModel()
