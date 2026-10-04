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
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        if model != "langid":
            raise ValueError(
                f"unknown langid model {model!r}; the only one is 'langid'"
            )
        weights = load_weights(fetch(LANGID_SDIST, only_local_files=only_local_files))
        keep = _candidates(weights.labels, languages)
        self._next_state = weights.next_state
        self._n_states = weights.n_states
        self._state_log_prob = _state_log_prob(weights)[:, keep]
        self._class_log_prior = weights.class_log_prior[keep]
        self._mapper = self._label_mapper([weights.labels[i] for i in keep])

    def _states(self, text: str) -> list[int]:
        """The DFA state entered after each UTF-8 byte of ``text``."""
        next_state = self._next_state
        state = 0
        states = []
        for byte in text.encode():
            state = next_state[(state << 8) + byte]
            states.append(state)
        return states

    def _log_scores(self, texts: list[str]) -> np.ndarray:
        """Unnormalised log P(class | text), shape ``(len(texts), n_candidates)``."""
        visits = np.stack(
            [np.bincount(self._states(t), minlength=self._n_states) for t in texts]
        )
        return visits @ self._state_log_prob + self._class_log_prior

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
