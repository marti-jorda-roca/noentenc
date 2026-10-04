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
    model._max_chars = 1000
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


class _TruncatingTokenizer:
    """One token per word, truncated to `max_length` with the rest in `overflowing`."""

    def __init__(self, max_length: int) -> None:
        self.max_length = max_length
        self.seen: list[str] = []

    def encode_batch(self, texts: list[str]) -> list[SimpleNamespace]:
        self.seen += texts
        out = []
        for text in texts:
            ids = [len(word) for word in text.split()]
            out.append(
                SimpleNamespace(
                    ids=ids[: self.max_length],
                    overflowing=[ids[self.max_length :]]
                    if len(ids) > self.max_length
                    else [],
                )
            )
        return out


def test_long_texts_are_cut_without_changing_their_tokens() -> None:
    model, _ = _fake_model({"input_ids", "attention_mask"})
    tokenizer = _TruncatingTokenizer(max_length=4)
    model._tokenizer = cast("Tokenizer", tokenizer)
    model._max_chars = 20
    short_words = "ab " * 30  # 4 tokens fit well within the cut
    long_words = "abcdefghij " * 5  # 2 words per 20 characters: the cut keeps too few
    small = "ab cd"
    ids = model.token_ids([short_words, long_words, small])
    assert ids == [[2, 2, 2, 2], [10, 10, 10, 10], [2, 2]]
    # The first text was tokenized from its cut, the second again in full after its cut.
    assert tokenizer.seen[0] == short_words[:17].rstrip()
    assert tokenizer.seen[3:] == [long_words]
