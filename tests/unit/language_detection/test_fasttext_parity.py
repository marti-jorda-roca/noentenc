"""Our numpy engine against predictions recorded from the reference ``fasttext`` package.

Fixtures come from ``scripts/make_fasttext_fixtures.py``.
"""

import pytest

from noentenc.language_detection.models.fasttext import FastTextModel
from tests.unit.language_detection.helpers import (
    FASTTEXT_EXPECTED,
    FASTTEXT_FIXTURES,
    assert_fasttext_parity,
)

TINY_MODELS = sorted(name for name in FASTTEXT_EXPECTED if name.startswith("tiny-"))


@pytest.mark.parametrize("name", TINY_MODELS)
def test_tiny_models_match_fasttext(name: str, sentences: list[str]) -> None:
    model = FastTextModel(FASTTEXT_FIXTURES / name, normalize_labels=False)
    assert_fasttext_parity(model, sentences, FASTTEXT_EXPECTED[name])


def test_normalized_labels_on_tiny_model(sentences: list[str]) -> None:
    model = FastTextModel(FASTTEXT_FIXTURES / "tiny-softmax.bin")
    assert set(model.labels) == {"eng", "spa", "cat", "deu", "rus", "jpn"}
    assert model.predict("el gat seia a la catifa") == "cat"
    assert model.predict_batch(["", "   ", sentences[0]])[:2] == ["und", "und"]
