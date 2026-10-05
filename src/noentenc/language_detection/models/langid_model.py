"""langid.py's naive Bayes classifier over byte n-grams (97 languages), reimplemented with numpy.

The weights come from the ``langid`` 1.1.6 sdist on PyPI (BSD-2-Clause, Marco Lui), which embeds
them in ``langid.py`` as a base64, bz2-compressed pickle. We download that tarball (sha256-pinned)
and unpickle it allowing only ``array.array``, so the ``langid`` package is never installed or run.
"""

import array
import base64
import bz2
import io
import pickle  # nosec B403 - only array.array can be unpickled, see _ArrayUnpickler
import re
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from noentenc.language_detection._download import RemoteFile, fetch
from noentenc.language_detection.labels import to_iso639_3
from noentenc.language_detection.models.base import BaseModel

LANGID_SDIST = RemoteFile(
    url="https://files.pythonhosted.org/packages/ea/4c/0fb7d900d3b0b9c8703be316fbddffecdab23c64e1b46c7a83561d78bd43/langid-1.1.6.tar.gz",
    cache_path="langid/langid-1.1.6.tar.gz",
    sha256="044bcae1912dab85c33d8e98f2811b8f4ff1213e5e9a9e9510137b84da2cb293",
)
MODEL_MEMBER = "langid-1.1.6/langid/langid.py"
_BYTE_BITS = 8
_BYTE_VALUES = 1 << _BYTE_BITS
# `_states` looks up two bytes at once from the root, so windows have at least two bytes.
_MIN_WINDOW = 2
# Below 1 byte per this many DFA states, `_log_scores` finds the visited states by sorting.
_SORT_RATIO = 8
_MODEL_STRING = re.compile(rb'^model=b"""(.*?)"""', re.DOTALL | re.MULTILINE)


@dataclass(frozen=True)
class LangidWeights:
    """langid's model; the comments give the names it uses."""

    labels: list[str]  # nb_classes, ISO 639-1
    class_log_prior: np.ndarray  # nb_pc, shape (n_classes,)
    feature_log_prob: np.ndarray  # nb_ptc, shape (n_features, n_classes)
    # tk_nextmove: a DFA over UTF-8 bytes, the next state is next_state[(state << 8) + byte].
    next_state: array.array
    # tk_output: the byte n-gram features that end on entering each state.
    state_features: dict[int, tuple[int, ...]]

    @property
    def n_states(self) -> int:
        return len(self.next_state) >> 8


class _ArrayUnpickler(pickle.Unpickler):
    """langid's model pickle holds lists, tuples, dicts and ``array.array``s, nothing else."""

    def find_class(self, module: str, name: str) -> type:
        if (module, name) == ("array", "array"):
            return array.array
        raise pickle.UnpicklingError(
            f"unexpected global {module}.{name} in langid model"
        )


def load_weights(path: Path) -> LangidWeights:
    """Read the model embedded in ``langid.py`` inside the langid sdist at ``path``."""
    with tarfile.open(path, "r:gz") as tar:
        member = tar.extractfile(MODEL_MEMBER)
        if member is None:
            raise ValueError(f"{path} has no {MODEL_MEMBER}")
        source = member.read()
    match = _MODEL_STRING.search(source)
    if match is None:
        raise ValueError(f"{path}: {MODEL_MEMBER} has no embedded model")
    raw = bz2.decompress(base64.b64decode(match[1]))
    nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = _ArrayUnpickler(
        io.BytesIO(raw)
    ).load()
    return LangidWeights(
        labels=list(nb_classes),
        class_log_prior=np.asarray(nb_pc),
        feature_log_prob=np.asarray(nb_ptc).reshape(-1, len(nb_pc)),
        next_state=tk_nextmove,
        state_features=tk_output,
    )


class LangidModel(BaseModel):
    """langid.py's classifier, run with numpy only (no ``langid`` package).

    Scores are langid's ``norm_probs=True`` probabilities: a softmax over the candidate
    languages. ``languages`` restricts the candidates, given as ISO 639-3 codes or as langid's
    own ISO 639-1 codes.
    """

    def __init__(
        self,
        model: str = "langid",
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
        languages: Iterable[str] | None = None,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        weights = load_weights(
            fetch(
                self.remote_files(model)[0],
                only_local_files=only_local_files,
                cache_dir=cache_dir,
            )
        )
        keep = _candidates(weights.labels, languages)
        self._next_state = np.asarray(weights.next_state, dtype=np.uint16)
        self._n_states = weights.n_states
        # Any window at least as long as the longest n-gram gives the same states.
        self._window = max(_max_depth(self._next_state), _MIN_WINDOW)
        idle = np.flatnonzero(self._next_state[:_BYTE_VALUES] == 0)
        if not len(idle):
            raise ValueError("langid's DFA has no byte that keeps it at its root")
        # Fed to the root, this byte leaves it there: padding with it resets the DFA.
        self._pad = bytes([int(idle[0])]) * (self._window - 1)
        self._state_log_prob = _state_log_prob(weights)[:, keep]
        # The walk's last step sends states where no feature ends to the root, which has no
        # features either: fewer distinct states to count, same scores.
        scored = self._state_log_prob.any(axis=1)
        if scored[0]:
            raise ValueError("langid's DFA root state has features")
        self._last_step = np.where(scored[self._next_state], self._next_state, 0)
        # The state reached from the root on any two bytes, indexed by `(b1 << 8) | b2`.
        second = self._last_step if self._window == _MIN_WINDOW else self._next_state
        self._two_steps = second[
            _shift_byte(self._next_state[:_BYTE_VALUES, None]) + np.arange(_BYTE_VALUES)
        ].ravel()
        self._class_log_prior = weights.class_log_prior[keep]
        self._mapper = self._label_mapper([weights.labels[i] for i in keep])

    @staticmethod
    def remote_files(model: str = "langid") -> list[RemoteFile]:
        """The file the model downloads: the langid sdist."""
        if model != "langid":
            raise ValueError(
                f"unknown langid model {model!r}; the only one is 'langid'"
            )
        return [LANGID_SDIST]

    def _states(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(states, owner)``: the DFA state entered after each UTF-8 byte of every text.

        States with no features come back as the root, 0.

        langid's DFA is an Aho-Corasick automaton over byte n-grams of at most ``window``
        bytes, so the state after a byte depends only on the last ``window`` bytes: running
        the DFA from the root over each window gives the same states as a walk over the
        whole text, and all windows are computed at once. Texts are joined with padding
        that keeps the DFA at the root, so windows never mix two texts. ``owner`` is the
        text of each byte, or ``len(texts)`` for padding bytes.
        """
        encoded = [text.encode() for text in texts]
        pad = self._pad
        blob = np.frombuffer(pad + pad.join(encoded), dtype=np.uint8).astype(np.intp)
        window = self._window
        n = len(blob) - window + 1
        state = self._two_steps[_shift_byte(blob[:n]) + blob[1 : n + 1]]
        for k in range(2, window):
            table = self._last_step if k == window - 1 else self._next_state
            state = table[_shift_byte(state) + blob[k : n + k]]
        # Window i ends on blob byte i + len(pad): text 0, padding, text 1, ..., text B-1.
        spans = np.full(2 * len(texts) - 1, len(pad))
        spans[::2] = [len(e) for e in encoded]
        owners = np.full(len(spans), len(texts))
        owners[::2] = np.arange(len(texts))
        return state, np.repeat(owners, spans)

    def _log_scores(self, texts: list[str]) -> np.ndarray:
        """Unnormalised log P(class | text), shape ``(len(texts), n_candidates)``.

        Counts state visits per text, over only the states this batch visits.
        """
        states, owner = self._states(texts)
        if len(states) * _SORT_RATIO < self._n_states:
            # Few bytes: sorting them is cheaper than scanning every state.
            visited, column = np.unique(states, return_inverse=True)
        else:
            visited = np.flatnonzero(np.bincount(states, minlength=self._n_states))
            columns = np.zeros(self._n_states, dtype=np.intp)
            columns[visited] = np.arange(len(visited))
            column = columns[states]
        rows = len(texts) + 1  # the last row collects the padding bytes
        visits = np.bincount(
            owner * len(visited) + column, minlength=rows * len(visited)
        ).reshape(rows, len(visited))[:-1]
        return (
            visits.astype(np.float64) @ self._state_log_prob[visited]
            + self._class_log_prior
        )

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        scores = self._log_scores(texts)
        if not self._mapper.identity:
            scores = self._mapper.reduce(_softmax(scores))
        return self._mapper.top_label(scores.argmax(axis=1))

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        probs = self._mapper.reduce(_softmax(self._log_scores(texts)))
        return self._mapper.to_dicts(probs, top_k)


def _candidates(labels: list[str], languages: Iterable[str] | None) -> list[int]:
    if languages is None:
        return list(range(len(labels)))
    wanted = set(languages)
    keep = [
        i
        for i, code in enumerate(labels)
        if code in wanted or to_iso639_3(code) in wanted
    ]
    known = {name for i in keep for name in (labels[i], to_iso639_3(labels[i]))}
    if unknown := wanted - known:
        raise ValueError(f"langid has no language {', '.join(sorted(unknown))}")
    return keep


def _shift_byte(states: np.ndarray) -> np.ndarray:
    """``states << 8`` as row offsets into a ``(state, byte)`` table, widened before shifting."""
    return np.left_shift(states, _BYTE_BITS, dtype=np.intp)


def _max_depth(next_state: np.ndarray) -> int:
    """Longest byte string a state stands for: its breadth-first distance from the root."""
    table = next_state.reshape(-1, _BYTE_VALUES)
    seen = np.zeros(len(table), dtype=bool)
    seen[0] = True
    frontier = np.array([0])
    depth = 0
    while True:
        reached = np.unique(table[frontier])
        frontier = reached[~seen[reached]]
        if not len(frontier):
            return depth
        seen[frontier] = True
        depth += 1


def _state_log_prob(weights: LangidWeights) -> np.ndarray:
    """Summed log P(feature | class) of the features that end in each state.

    langid counts features and multiplies by ``nb_ptc``. Each visit to a state adds the same
    features, so counting state visits and multiplying by this matrix gives the same scores.
    """
    out = np.zeros((weights.n_states, len(weights.labels)))
    for state, features in weights.state_features.items():
        if features:
            out[state] = weights.feature_log_prob[list(features)].sum(
                axis=0, dtype=np.float64
            )
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)
