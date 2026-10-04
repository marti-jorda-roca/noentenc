"""Real langid weights against predictions recorded from the reference ``langid`` package.

Fixtures come from ``scripts/make_langid_fixtures.py``.
"""

import json

import numpy as np
import pytest

from noentenc.language_detection.models.langid_model import LANGID_SDIST, LangidModel
from tests.integrations.language_detection.helpers import skip_unless_available
from tests.unit.language_detection.helpers import FIXTURES, load_sentences

pytestmark = pytest.mark.integration

EXPECTED = json.loads(
    (FIXTURES / "langid" / "expected.json").read_text(encoding="utf-8")
)
RESTRICTED = ["cat", "spa", "por", "eng"]
ATOL = 1e-9


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_matches_langid(name: str) -> None:
    skip_unless_available(LANGID_SDIST)
    languages = RESTRICTED if name == "langid-restricted" else None
    model = LangidModel(normalize_labels=False, languages=languages)
    sentences = load_sentences()
    got = model.predict_batch_score(sentences, top_k=None)
    top1 = model.predict_batch(sentences)
    for sentence, scores, best, ref in zip(
        sentences, got, top1, EXPECTED[name], strict=True
    ):
        ours = [scores[label] for label in ref["labels"]]
        np.testing.assert_allclose(ours, ref["probs"], atol=ATOL, err_msg=sentence)
        assert best == ref["labels"][0], sentence


def test_detects_and_normalises() -> None:
    skip_unless_available(LANGID_SDIST)
    model = LangidModel()
    texts = [
        "Bon dia a tothom, com esteu? Avui fa molt bon temps a Barcelona.",
        "",
        "The weather is lovely today.",
    ]
    assert model.predict_batch(texts) == ["cat", "und", "eng"]
    assert len(model.labels) == 97
