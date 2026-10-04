"""Our numpy engine against predictions recorded from the reference ``fasttext`` package.

Fixtures come from ``scripts/make_fasttext_fixtures.py``.
"""

import numpy as np
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


@pytest.mark.parametrize("cache_size", [None, 0, 1, 7])
@pytest.mark.parametrize("name", ["tiny-hs.bin", "tiny-wordngrams.bin"])
def test_word_cache_sizes_give_the_same_hidden_vectors(
    name: str, cache_size: int | None, sentences: list[str]
) -> None:
    model = FastTextModel(FASTTEXT_FIXTURES / name, cache_size=cache_size)
    tok = model._tokenizer
    expected = np.stack(
        [model._input[tok.text_rows(s) or [0]].mean(0) for s in sentences]
    )
    # Twice, so the second pass reads the words the first one cached (or evicted).
    for _ in range(2):
        np.testing.assert_allclose(model.hidden(sentences), expected, atol=1e-6)
        for sentence, row in zip(sentences, expected, strict=True):
            np.testing.assert_allclose(model.hidden([sentence])[0], row, atol=1e-6)
