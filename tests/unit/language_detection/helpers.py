"""Shared helpers for the language-detection tests."""

import json
import struct
from pathlib import Path

import numpy as np

from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.language_detection.models.fasttext.format import Loss

FIXTURES = Path(__file__).parent / "fixtures"
FASTTEXT_FIXTURES = FIXTURES / "fasttext"
# Predictions recorded by ``scripts/make_fasttext_fixtures.py``.
FASTTEXT_EXPECTED = json.loads(
    (FASTTEXT_FIXTURES / "expected.json").read_text(encoding="utf-8")
)
# fastText reports p + 1e-5 for softmax/one-vs-all; hierarchical softmax scores are compared as-is.
FASTTEXT_OFFSET = 1e-5
ATOL = 2e-6


def load_sentences() -> list[str]:
    return (FIXTURES / "sentences.txt").read_text(encoding="utf-8").splitlines()


def assert_fasttext_parity(
    model: FastTextModel, sentences: list[str], expected: list[dict]
) -> None:
    """Compare our predictions with those recorded from the reference ``fasttext`` package."""
    got = model.predict_batch_score(sentences, top_k=None)
    top1 = model.predict_batch(sentences)
    offset = 0.0 if model.args.loss == Loss.HS else FASTTEXT_OFFSET
    for sentence, scores, best, ref in zip(sentences, got, top1, expected, strict=True):
        ref_probs = np.array(ref["probs"])
        ours = np.array([scores.get(label, 0.0) for label in ref["labels"]]) + offset
        np.testing.assert_allclose(
            np.minimum(ours, 1.0),
            np.minimum(ref_probs, 1.0),
            atol=ATOL,
            err_msg=sentence,
        )
        # Ties aside, the best label must agree with fastText.
        if len(ref_probs) < 2 or ref_probs[0] - ref_probs[1] > ATOL:
            assert best == ref["labels"][0], sentence


def write_fasttext_model(
    path: Path,
    *,
    words: list[str],
    labels: list[str],
    input_block: bytes,
    output_block: bytes,
    quant_input: bool = False,
    qout: bool = False,
    loss: int = 3,
    dim: int = 4,
    bucket: int = 16,
    minn: int = 2,
    maxn: int = 3,
    word_ngrams: int = 1,
    label_counts: list[int] | None = None,
    pruneidx: dict[int, int] | None = None,
) -> Path:
    """Write a fastText model file from raw parts (see ``format.py`` for the layout)."""
    counts = label_counts or list(range(len(labels) * 10, 0, -10))
    out = bytearray(struct.pack("<ii", 793712314, 12))
    out += struct.pack(
        "<12i", dim, 5, 1, 1, 5, word_ngrams, loss, 3, bucket, minn, maxn, 100
    )
    out += struct.pack("<d", 1e-4)
    pruneidx_size = -1 if pruneidx is None else len(pruneidx)
    out += struct.pack(
        "<3iqq", len(words) + len(labels), len(words), len(labels), 1000, pruneidx_size
    )
    for i, word in enumerate(words):
        out += word.encode() + b"\0" + struct.pack("<qb", 100 - i, 0)
    for label, count in zip(labels, counts, strict=True):
        out += label.encode() + b"\0" + struct.pack("<qb", count, 1)
    for key, value in (pruneidx or {}).items():
        out += struct.pack("<ii", key, value)
    out += struct.pack("<?", quant_input) + input_block
    out += struct.pack("<?", qout) + output_block
    path.write_bytes(bytes(out))
    return path


def dense_block(matrix: np.ndarray) -> bytes:
    return struct.pack("<qq", *matrix.shape) + matrix.astype("<f4").tobytes()


def pq_block(
    dim: int, nsubq: int, dsub: int, lastdsub: int, centroids: np.ndarray
) -> bytes:
    return (
        struct.pack("<4i", dim, nsubq, dsub, lastdsub)
        + centroids.astype("<f4").tobytes()
    )


def quant_block(
    codes: np.ndarray,
    centroids: np.ndarray,
    *,
    dim: int,
    dsub: int,
    norm_codes: np.ndarray | None = None,
    norm_centroids: np.ndarray | None = None,
) -> bytes:
    m, nsubq = codes.shape
    lastdsub = dim - dsub * (nsubq - 1)
    out = (
        struct.pack("<?qqi", norm_codes is not None, m, dim, codes.size)
        + codes.astype(np.uint8).tobytes()
    )
    out += pq_block(dim, nsubq, dsub, lastdsub, centroids)
    if norm_codes is not None and norm_centroids is not None:
        out += norm_codes.astype(np.uint8).tobytes() + pq_block(
            1, 1, 1, 1, norm_centroids
        )
    return out
