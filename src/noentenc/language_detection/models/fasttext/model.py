"""Pure-numpy inference for fastText language-identification models."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path

import numpy as np

from noentenc.language_detection._download import RemoteFile, fetch
from noentenc.language_detection.models.base import BaseModel
from noentenc.language_detection.models.fasttext import format as ftz
from noentenc.language_detection.models.fasttext.format import Loss
from noentenc.language_detection.models.fasttext.tokenizer import (
    Tokenizer,
    split_words,
)

# fastText's `std_log(x) = log(x + 1e-5)`, used along hierarchical-softmax paths.
HS_LOG_EPSILON = 1e-5
# exp(-x) overflows float32 past ~88; the sigmoid there is already 0 to float precision.
SIGMOID_CLIP = 80.0
# Marks the end of each text among the words looked up in the word cache.
END_OF_TEXT = ""
# fastText's one-vs-all loss uses a lookup-table sigmoid (`Loss::sigmoid` in loss.cc).
SIGMOID_TABLE_SIZE = 512
MAX_SIGMOID = 8.0
_SIGMOID_TABLE = (
    1.0
    / (
        1.0
        + np.exp(
            -(
                np.arange(SIGMOID_TABLE_SIZE + 1, dtype=np.float32)
                * np.float32(2 * MAX_SIGMOID / SIGMOID_TABLE_SIZE)
                - np.float32(MAX_SIGMOID)
            )
        )
    )
).astype(np.float32)


@dataclass(frozen=True)
class FastTextPreset:
    remote: RemoteFile
    languages: int
    license: str
    description: str


_FB = "https://dl.fbaipublicfiles.com/fasttext/supervised-models"

PRESETS: dict[str, FastTextPreset] = {
    "lid176": FastTextPreset(
        remote=RemoteFile(
            url=f"{_FB}/lid.176.ftz",
            cache_path="fasttext/lid.176.ftz",
            sha256="8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83",
        ),
        languages=176,
        license="CC-BY-SA-3.0",
        description="fastText lid.176, quantized (0.9 MB). Smallest and fastest; the default.",
    ),
    "lid176-bin": FastTextPreset(
        remote=RemoteFile(url=f"{_FB}/lid.176.bin", cache_path="fasttext/lid.176.bin"),
        languages=176,
        license="CC-BY-SA-3.0",
        description="fastText lid.176, full precision (126 MB). Slightly more accurate than lid176.",
    ),
    "nllb-lid218e": FastTextPreset(
        remote=RemoteFile.huggingface(
            "facebook/fasttext-language-identification",
            "model.bin",
            revision="3af127d4124fc58b75666f3594bb5143b9757e78",
            sha256="8ded5749a2ad79ae9ab7c9190c7c8b97ff20d54ad8b9527ffa50107238fc7f6a",
        ),
        languages=218,
        license="CC-BY-NC-4.0",
        description="NLLB's LID model (1.2 GB). Non-commercial licence.",
    ),
    "openlid-v2": FastTextPreset(
        remote=RemoteFile.huggingface(
            "laurievb/OpenLID-v2",
            "model.bin",
            revision="10b538ebcc452255917c8bdfc38faa49af134cd2",
            sha256="f01b47c6182909291a3394f77f682031aaeade346e443c4e496e94bd2a9fa703",
        ),
        languages=200,
        license="GPL-3.0",
        description="OpenLID-v2 (1.2 GB): the FLORES-200 languages, high precision.",
    ),
    "openlid-v3": FastTextPreset(
        remote=RemoteFile.huggingface(
            "HPLT/OpenLID-v3",
            "openlid-v3.bin",
            revision="6b9560483e17e42f48d86cebf22b4b58dffeaa70",
            sha256="01ec5bbf85975c52673c52b3d8585bc4cc0e4eb4636ad5591467bab56b118821",
        ),
        languages=195,
        license="GPL-3.0",
        description="OpenLID-v3 (1.2 GB): better at telling closely related languages apart.",
    ),
    "glotlid": FastTextPreset(
        remote=RemoteFile.huggingface(
            "cis-lmu/glotlid",
            "model_v3.bin",
            revision="85cd6716494360367b75f642b5bc78667605d0b4",
            sha256="a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e",
        ),
        languages=2102,
        license="Apache-2.0",
        description="GlotLID v3 (1.7 GB): 2102 labels, the widest coverage.",
    ),
}
DEFAULT_PRESET = "lid176"


class FastTextModel(BaseModel):
    """A fastText supervised classifier, run with numpy only (no ``fasttext`` package).

    ``model`` is a preset name (see ``PRESETS``) or the path to a ``.bin``/``.ftz`` file.
    Dense weights are memory-mapped, so loading is near-instant even for multi-GB models.

    Probabilities: softmax models return the true softmax probabilities. Hierarchical-softmax
    models (lid.176) return fastText's path scores, ``exp(sum(log(p + 1e-5)))``, which is what
    ``fasttext.predict`` reports (clipped at 1); they don't sum exactly to 1.

    ``cache_size`` is how many distinct words keep their summed input rows in memory
    (``dim + 1`` floats each: 68 bytes for lid176, 1 KB for the 256-dim models), so repeated
    words skip tokenization. ``None`` keeps every word, ``0`` none.
    """

    def __init__(
        self,
        model: str | Path = DEFAULT_PRESET,
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
        cache_size: int | None = 1 << 17,
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        weights = ftz.load(self._resolve(model, only_local_files))
        self.args = weights.args
        self.native_labels = weights.labels
        self._tokenizer = Tokenizer(weights, cache_size=cache_size)
        self._input = weights.input_matrix
        self._output = weights.output_matrix
        self._words = _WordCache(self._word_vectors, self.args.dim + 1, cache_size)
        self._mapper = self._label_mapper(weights.labels)
        if self.args.loss == Loss.HS:
            right, left = _hs_path_matrices(weights.label_counts)
            # Internal tree node `nlabels + i` uses output row i; the last row is unused.
            # Negated, so `hidden @ _hs_negated.T` is the `-x` that the sigmoid needs.
            self._hs_negated = -self._output[: right.shape[0]]
            self._hs_paths = np.concatenate((right, left), axis=0)

    @staticmethod
    def _resolve(model: str | Path, only_local_files: bool) -> Path:
        if isinstance(model, str) and model in PRESETS:
            return fetch(PRESETS[model].remote, only_local_files=only_local_files)
        path = Path(model)
        if not path.is_file():
            raise FileNotFoundError(
                f"{model!r} is neither a fastText preset ({', '.join(PRESETS)}) nor a file"
            )
        return path

    def hidden(self, texts: list[str]) -> np.ndarray:
        """Mean input-matrix row of each text, shape ``(len(texts), dim)``.

        Each distinct word is tokenized once, into the sum of its rows (see ``_WordCache``); a
        text's hidden vector is then the sum over its words, its ``</s>`` row and its word n-grams,
        divided by their total row count.
        """
        words: list[str] = []
        starts: list[int] = []
        for text in texts:
            starts.append(len(words))
            words += split_words(text)
            words.append(END_OF_TEXT)
        # Column `dim` of each word entry holds its row count.
        summed = np.add.reduceat(self._words.lookup(words), starts, axis=0)
        total, count = summed[:, :-1], summed[:, -1]
        if self._tokenizer.word_ngrams > 1:
            ends = [*starts[1:], len(words)]
            ngrams = [
                self._tokenizer.word_ngram_rows(words[a : b - 1])
                for a, b in zip(starts, ends, strict=True)
            ]
            lengths = np.array([len(rows) for rows in ngrams])
            flat = np.array([row for rows in ngrams for row in rows], dtype=np.intp)
            total += _gather_sums(self._input, flat, lengths)
            count += lengths
        # No rows at all (only when the vocabulary lacks `</s>`, and fastText would then
        # predict nothing): use row 0 so the scores stay finite.
        if not count.all():
            empty = count == 0
            total[empty] = self._input[0]
            count[empty] = 1
        return total / count[:, None]

    def _word_vectors(self, words: list[str]) -> np.ndarray:
        """Summed input rows of each word (``""`` is a text's end), then its row count."""
        owner, rows = self._tokenizer.word_rows_batch(words)
        counts = np.bincount(owner, minlength=len(words))
        out = np.empty((len(words), self.args.dim + 1), dtype=np.float32)
        out[:, :-1] = _gather_sums(self._input, rows, counts)
        out[:, -1] = counts
        return out

    def _native_scores(self, hidden: np.ndarray) -> np.ndarray:
        """Per-label probabilities (or hs path scores) on the model's native label axis."""
        if self.args.loss == Loss.HS:
            # The +1e-5 per step can push a near-certain leaf a hair above 1.
            return np.minimum(np.exp(self._hs_log_scores(hidden)), 1.0)
        logits = hidden @ self._output.T
        if self.args.loss == Loss.SOFTMAX:
            return _softmax(logits)
        # One-vs-all and negative-sampling models score each label independently.
        return _table_sigmoid(logits)

    def _hs_log_scores(self, hidden: np.ndarray) -> np.ndarray:
        """Log path score of every leaf, in one matmul against the stacked path matrices.

        Like fastText, the left branch is ``log(1 - f + eps)`` with ``f`` the float32 sigmoid,
        not the more precise ``sigmoid(-x)``, so scores match ``fasttext.predict``.
        """
        negated = hidden @ self._hs_negated.T
        internal = negated.shape[1]
        branch = np.empty((len(hidden), 2 * internal), dtype=negated.dtype)
        right, left = branch[:, :internal], branch[:, internal:]
        # sigmoid(x) = 1 / (1 + exp(-x)); exp overflows float32 past ~88, where the
        # sigmoid is already 0 to float precision.
        np.exp(np.minimum(negated, SIGMOID_CLIP, out=negated), out=right)
        right += 1.0
        np.reciprocal(right, out=right)
        np.subtract(1.0, right, out=left)
        branch += HS_LOG_EPSILON
        return np.log(branch, out=branch) @ self._hs_paths

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        hidden = self.hidden(texts)
        if not self._mapper.identity:
            scores = self._mapper.reduce(self._native_scores(hidden))
        elif self.args.loss == Loss.HS:
            # exp is monotonic: rank on log path scores.
            scores = self._hs_log_scores(hidden)
        else:
            # softmax and sigmoid are monotonic: rank on raw logits.
            scores = hidden @ self._output.T
        return self._mapper.top_label(scores.argmax(axis=1))

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        unified = self._mapper.reduce(self._native_scores(self.hidden(texts)))
        return self._mapper.to_dicts(unified, top_k)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def _gather_sums(
    matrix: np.ndarray, rows: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Sum ``matrix[rows]`` over consecutive runs, ``lengths[i]`` rows for run ``i`` (may be 0).

    Runs of equal length are summed together as one ``(runs, length, dim)`` block, row by row in
    order. That is several times faster than ``np.add.reduceat`` over the gathered rows, which
    handles wide rows split into many short runs poorly.
    """
    out = np.zeros((len(lengths), matrix.shape[1]), dtype=matrix.dtype)
    if not len(lengths):
        return out
    starts = np.cumsum(lengths) - lengths
    order = np.argsort(lengths, kind="stable")
    by_length = lengths[order]
    bounds = [*np.flatnonzero(np.diff(by_length)).tolist(), len(order) - 1]
    first = 0
    for last in bounds:
        length = int(by_length[last])
        if length:
            runs = order[first : last + 1]
            block = rows[starts[runs, None] + np.arange(length)]
            out[runs] = matrix[block].sum(axis=1)
        first = last + 1
    return out


class _WordCache:
    """Per-word entries computed once and reused: here, summed input rows plus a row count.

    ``compute`` fills in the words a lookup hasn't seen, all in one call. Once ``capacity``
    words are stored the cache empties and starts over, which is cheaper than LRU bookkeeping
    and bounds memory to ``capacity`` entries. ``None`` never evicts; ``0`` never stores.
    """

    def __init__(
        self,
        compute: Callable[[list[str]], np.ndarray],
        width: int,
        capacity: int | None,
    ) -> None:
        self._compute = compute
        self._capacity = capacity
        self._slots: dict[str, int] = {}
        self._table = np.empty((0, width), dtype=np.float32)
        # Lookups mutate the table, so concurrent callers take turns.
        self._lock = threading.Lock()

    def lookup(self, words: list[str]) -> np.ndarray:
        """The entries of ``words``, one row per word."""
        n = len(words)
        with self._lock:
            slots = self._slots
            at = np.fromiter(map(slots.get, words, repeat(-1)), np.intp, n)
            miss = np.flatnonzero(at < 0)
            if len(miss):
                missed = list(map(words.__getitem__, miss.tolist()))
                missing = list(dict.fromkeys(missed))
                capacity = self._capacity
                if capacity is not None and len(slots) + len(missing) > capacity:
                    slots.clear()
                    missed = words
                    missing = list(dict.fromkeys(words))
                    miss = np.arange(n)
                    if len(missing) > capacity:
                        index = dict(zip(missing, range(len(missing)), strict=True))
                        at = np.fromiter(map(index.__getitem__, words), np.intp, n)
                        return self._compute(missing)[at]
                self._store(missing)
                at[miss] = np.fromiter(
                    map(slots.__getitem__, missed), np.intp, len(miss)
                )
            return self._table[at]

    def _store(self, words: list[str]) -> None:
        entries = self._compute(words)
        start = len(self._slots)
        end = start + len(words)
        if end > len(self._table):
            size = max(end, 2 * len(self._table))
            if self._capacity is not None:
                size = min(size, self._capacity)
            grown = np.empty((size, self._table.shape[1]), dtype=self._table.dtype)
            grown[:start] = self._table[:start]
            self._table = grown
        self._table[start:end] = entries
        self._slots.update(zip(words, range(start, end), strict=True))


def _table_sigmoid(x: np.ndarray) -> np.ndarray:
    index = ((x + MAX_SIGMOID) * (SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2)).astype(
        np.int64
    )
    out = _SIGMOID_TABLE[np.clip(index, 0, SIGMOID_TABLE_SIZE)]
    out[x < -MAX_SIGMOID] = 0.0
    out[x > MAX_SIGMOID] = 1.0
    return out


def _hs_path_matrices(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode fastText's hierarchical-softmax Huffman tree as two ``(internal, leaves)`` 0/1 matrices.

    The tree is built exactly like ``HierarchicalSoftmaxLoss::buildTree``. A leaf's path score is
    the sum over its ancestors of ``log(f + eps)`` when the path goes right and ``log(1 - f + eps)``
    when it goes left, where ``f`` is the sigmoid of the ancestor's output row. So
    ``log(f + eps) @ right + log(1 - f + eps) @ left`` scores every leaf at once.
    """
    n = len(counts)
    size = 2 * n - 1
    tree_count = np.full(size, 1e15)
    tree_count[:n] = counts
    parent = np.full(size, -1, dtype=np.intp)
    is_right = np.zeros(size, dtype=bool)
    leaf, node = n - 1, n
    for i in range(n, size):
        pair = []
        for _ in range(2):
            if leaf >= 0 and tree_count[leaf] < tree_count[node]:
                pair.append(leaf)
                leaf -= 1
            else:
                pair.append(node)
                node += 1
        left, right = pair
        tree_count[i] = tree_count[left] + tree_count[right]
        parent[left] = parent[right] = i
        is_right[right] = True
    right_paths = np.zeros((n - 1, n), dtype=np.float32)
    left_paths = np.zeros((n - 1, n), dtype=np.float32)
    for leaf_id in range(n):
        child = leaf_id
        while parent[child] != -1:
            target = right_paths if is_right[child] else left_paths
            target[parent[child] - n, leaf_id] = 1.0
            child = parent[child]
    return right_paths, left_paths
