"""Our langid engine on a tiny hand-built model, packaged like the real langid sdist."""

import array
import base64
import bz2
import io
import math
import pickle
import tarfile
from pathlib import Path

import pytest

from noentenc.language_detection._download import CACHE_ENV
from noentenc.language_detection.models.langid_model import (
    LANGID_SDIST,
    MODEL_MEMBER,
    LangidModel,
    load_weights,
)

# Two unigram features, "a" (0) and "b" (1), and an Aho-Corasick DFA like langid's: state 1
# after an "a", state 2 after a "b", state 3 (no features) after a "c", the root (0) after any
# other byte. "a" is 9x likelier in English, "b" 9x likelier in Catalan.
LABELS = ["en", "ca"]
FEATURE_LOG_PROB = [math.log(0.9), math.log(0.1), math.log(0.1), math.log(0.9)]
FEATURE_STATES = {ord("a"): 1, ord("b"): 2, ord("c"): 3}


def _model_pickle(obj: object = None) -> bytes:
    if obj is None:
        next_state = array.array(
            "H", [FEATURE_STATES.get(byte, 0) for _ in range(4) for byte in range(256)]
        )
        state_features = {1: (0,), 2: (1,)}
        obj = (
            array.array("f", FEATURE_LOG_PROB),
            array.array("f", [math.log(0.5)] * 2),
            LABELS,
            next_state,
            state_features,
        )
    return pickle.dumps(obj, protocol=0)


def _write_sdist(path: Path, model_pickle: bytes) -> Path:
    source = b'model=b"""\n' + base64.b64encode(bz2.compress(model_pickle)) + b'\n"""\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(MODEL_MEMBER)
        info.size = len(source)
        tar.addfile(info, io.BytesIO(source))
    return path


@pytest.fixture
def tiny_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    return _write_sdist(tmp_path / LANGID_SDIST.cache_path, _model_pickle())


@pytest.mark.usefixtures("tiny_cache")
def test_tiny_model_scores() -> None:
    model = LangidModel()
    assert model.labels == ["eng", "cat"]
    assert model.predict_batch(["aab", "", "abb"]) == ["eng", "und", "cat"]
    # Two "a" and one "b": 0.9² · 0.1 against 0.1² · 0.9, normalised.
    p_eng = 0.081 / (0.081 + 0.009)
    scores = model.predict_score("aab")
    assert list(scores) == ["eng", "cat"]
    assert scores["eng"] == pytest.approx(p_eng, abs=1e-6)
    assert scores["cat"] == pytest.approx(1 - p_eng, abs=1e-6)
    assert model.predict_score("aab", top_k=1) == {
        "eng": pytest.approx(p_eng, abs=1e-6)
    }


@pytest.mark.usefixtures("tiny_cache")
def test_native_labels_and_language_restriction() -> None:
    assert LangidModel(normalize_labels=False).predict("abb") == "ca"
    for languages in (["cat"], ["ca"]):
        model = LangidModel(languages=languages)
        assert model.labels == ["cat"]
        assert model.predict_score("aaaa") == {"cat": pytest.approx(1.0)}
    with pytest.raises(ValueError, match="no language deu"):
        LangidModel(languages=["cat", "deu"])


def test_only_array_globals_are_unpickled(tmp_path: Path) -> None:
    path = _write_sdist(tmp_path / "evil.tar.gz", _model_pickle(print))
    with pytest.raises(pickle.UnpicklingError, match=r"unexpected global .*\.print"):
        load_weights(path)


def test_unknown_model_name(tiny_cache: Path) -> None:  # noqa: ARG001 - sets the cache
    with pytest.raises(ValueError, match="only one is 'langid'"):
        LangidModel("langid-v2")


def _walk(model: LangidModel, text: str) -> list[int]:
    """Reference: langid's byte-by-byte DFA walk, with featureless states reported as 0."""
    state, states = 0, []
    for byte in text.encode():
        state = int(model._next_state[(state << 8) + byte])
        states.append(state if model._state_log_prob[state].any() else 0)
    return states


@pytest.mark.usefixtures("tiny_cache")
def test_batched_states_match_a_sequential_walk() -> None:
    model = LangidModel()
    texts = ["aab", "", "b", "xxaxb\0a", "ñacb", "bac" * 50]
    states, owner = model._states(texts)
    for i, text in enumerate(texts):
        assert states[owner == i].tolist() == _walk(model, text), text
    assert set(owner.tolist()) <= set(range(len(texts) + 1))


@pytest.mark.usefixtures("tiny_cache")
def test_batch_scores_match_single_text_scores() -> None:
    model = LangidModel()
    texts = ["aab", "abb", "b", "xxcaxb", "bac" * 50]
    batched = model.predict_batch_score(texts, top_k=None)
    for text, scores in zip(texts, batched, strict=True):
        assert scores == pytest.approx(model.predict_score(text, top_k=None))
