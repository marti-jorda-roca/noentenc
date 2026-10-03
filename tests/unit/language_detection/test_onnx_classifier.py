from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

from noentenc.language_detection.labels import LabelMapper
from noentenc.language_detection.models.onnx_classifier import OnnxClassifierModel


class _FakeSession:
    def __init__(self) -> None:
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, _outputs: None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feeds)
        # Score = number of real tokens, so label 1 wins for longer texts.
        n = feeds["attention_mask"].sum(axis=1).astype(np.float32)
        return [np.stack([np.full_like(n, 3.0), n], axis=1)]


class _FakeTokenizer:
    def encode_batch(self, texts: list[str]) -> list[SimpleNamespace]:
        return [SimpleNamespace(ids=list(range(1, len(t.split()) + 1))) for t in texts]


def _fake_model(input_names: set[str]) -> tuple[OnnxClassifierModel, _FakeSession]:
    model = OnnxClassifierModel.__new__(OnnxClassifierModel)
    model.normalize_labels, model.collapse_macrolanguages = True, False
    model._mapper = LabelMapper(["en", "es"])
    model._pad_id = 9
    model._tokenizer = cast("Tokenizer", _FakeTokenizer())
    session = _FakeSession()
    model._session = cast("InferenceSession", session)
    model._input_names = input_names
    return model, session


def test_pads_each_batch_only_to_its_longest_text() -> None:
    model, session = _fake_model({"input_ids", "attention_mask"})
    texts = ["a", "a b c d e", "a b", "a b c d e f g"]
    assert model.predict_batch(texts, batch_size=2) == ["eng", "spa", "eng", "spa"]
    # Sorted by length: batches are (1, 2) words and (5, 7) words.
    widths = [feeds["input_ids"].shape[1] for feeds in session.feeds]
    assert widths == [2, 7]
    first = session.feeds[0]
    np.testing.assert_array_equal(first["input_ids"], [[1, 9], [1, 2]])
    np.testing.assert_array_equal(first["attention_mask"], [[1, 0], [1, 1]])
    assert "token_type_ids" not in first


def test_feeds_token_type_ids_only_when_declared() -> None:
    model, session = _fake_model({"input_ids", "attention_mask", "token_type_ids"})
    model.predict("a b")
    np.testing.assert_array_equal(session.feeds[0]["token_type_ids"], [[0, 0]])


def test_probabilities_are_softmax() -> None:
    model, _ = _fake_model({"input_ids", "attention_mask"})
    scores = model.predict_score("a b c", top_k=None)
    assert scores == {"eng": pytest.approx(0.5), "spa": pytest.approx(0.5)}
