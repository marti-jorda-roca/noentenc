"""Pure-numpy inference for fastText language-identification models."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from noentenc.language_detection._download import RemoteFile, fetch
from noentenc.language_detection.labels import LabelMapper
from noentenc.language_detection.models.base import BaseModel
from noentenc.language_detection.models.fasttext import format as ftz
from noentenc.language_detection.models.fasttext.format import Loss
from noentenc.language_detection.models.fasttext.tokenizer import Tokenizer

# fastText's `std_log(x) = log(x + 1e-5)`, used along hierarchical-softmax paths.
HS_LOG_EPSILON = 1e-5
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
        self._mapper = LabelMapper(
            weights.labels,
            normalize=normalize_labels,
            collapse_macrolanguages=collapse_macrolanguages,
        )
        if self.args.loss == Loss.HS:
            right, left = _hs_path_matrices(weights.label_counts)
            self._hs_internal = right.shape[0]
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

    @property
    def labels(self) -> list[str]:
        return list(self._mapper.labels)

    def hidden(self, texts: list[str]) -> np.ndarray:
        """Mean input-matrix row of each text, shape ``(len(texts), dim)``.

        All rows of the batch are gathered with one fancy-index call and summed per text with
        ``np.add.reduceat``.
        """
        rows: list[int] = []
        offsets: list[int] = []
        counts: list[int] = []
        text_rows = self._tokenizer.text_rows
        for text in texts:
            offsets.append(len(rows))
            # Only empty when the vocabulary lacks `</s>` (fastText would then predict
            # nothing); a placeholder row keeps reduceat aligned.
            ids = text_rows(text) or [0]
            counts.append(len(ids))
            rows.extend(ids)
        sums = np.add.reduceat(self._input[rows], offsets, axis=0)
        return sums / np.array(counts, dtype=np.float32)[:, None]

    def _native_scores(self, hidden: np.ndarray) -> np.ndarray:
        """Per-label probabilities (or hs path scores) on the model's native label axis."""
        logits = hidden @ self._output.T
        if self.args.loss == Loss.HS:
            # The +1e-5 per step can push a near-certain leaf a hair above 1.
            return np.minimum(np.exp(self._hs_log_scores(logits)), 1.0)
        if self.args.loss == Loss.SOFTMAX:
            return _softmax(logits)
        # One-vs-all and negative-sampling models score each label independently.
        return _table_sigmoid(logits)

    def _hs_log_scores(self, logits: np.ndarray) -> np.ndarray:
        """Log path score of every leaf, in one matmul against the stacked path matrices."""
        # Internal tree node `nlabels + i` uses output row i; the last row is unused.
        sig = _sigmoid(logits[:, : self._hs_internal])
        branch = np.log(np.concatenate((sig, 1.0 - sig), axis=1) + HS_LOG_EPSILON)
        return branch @ self._hs_paths

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        hidden = self.hidden(texts)
        if self._mapper.identity:
            # exp, softmax and sigmoid are monotonic: rank on raw logits / log path scores.
            logits = hidden @ self._output.T
            scores = (
                self._hs_log_scores(logits) if self.args.loss == Loss.HS else logits
            )
        else:
            scores = self._mapper.reduce(self._native_scores(hidden))
        return self._mapper.top_label(scores.argmax(axis=1))

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        unified = self._mapper.reduce(self._native_scores(self.hidden(texts)))
        return self._mapper.to_dicts(unified, top_k)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
