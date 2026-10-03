"""Reader for fastText's binary model format (``.bin`` and quantized ``.ftz``).

Layout, from facebookresearch/fastText ``src/fasttext.cc::loadModel`` (all little-endian):

    int32 magic, int32 version
    Args:        12 x int32 (dim ws epoch minCount neg wordNgrams loss model bucket minn maxn lrUpdateRate), float64 t
    Dictionary:  int32 size, nwords, nlabels; int64 ntokens, pruneidx_size
                 size x (utf-8 word, b"\\0", int64 count, int8 type); pruneidx_size x (int32, int32)
    bool quant_input, input matrix  (DenseMatrix or QuantMatrix)
    bool qout,        output matrix (QuantMatrix only if quant_input and qout)

Dense matrices are memory-mapped, so even multi-GB models open instantly and only the rows that
are used get paged in. Quantized matrices are small and are dequantized into RAM at load time.
"""

import mmap
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

FASTTEXT_MAGIC = 793712314
FASTTEXT_VERSION = 12
PQ_KSUB = 256

_ARG_NAMES = (
    "dim",
    "ws",
    "epoch",
    "min_count",
    "neg",
    "word_ngrams",
    "loss",
    "model",
    "bucket",
    "minn",
    "maxn",
    "lr_update_rate",
)


class Loss(IntEnum):
    HS = 1
    NS = 2
    SOFTMAX = 3
    OVA = 4


class ModelKind(IntEnum):
    CBOW = 1
    SKIPGRAM = 2
    SUPERVISED = 3


@dataclass(frozen=True)
class Args:
    dim: int
    ws: int
    epoch: int
    min_count: int
    neg: int
    word_ngrams: int
    loss: Loss
    model: ModelKind
    bucket: int
    minn: int
    maxn: int
    lr_update_rate: int
    t: float


@dataclass
class FastTextWeights:
    args: Args
    words: list[str]
    labels: list[str]
    label_counts: np.ndarray
    pruneidx: dict[int, int] | None
    input_matrix: np.ndarray
    output_matrix: np.ndarray
    word_ids: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.word_ids = {word: i for i, word in enumerate(self.words)}

    @property
    def nwords(self) -> int:
        return len(self.words)


class _Reader:
    def __init__(self, buffer: mmap.mmap | bytes) -> None:
        self.buffer = buffer
        self.offset = 0

    def unpack(self, fmt: str) -> tuple:
        values = struct.unpack_from("<" + fmt, self.buffer, self.offset)
        self.offset += struct.calcsize("<" + fmt)
        return values

    def array(self, dtype: type[np.generic], count: int) -> np.ndarray:
        arr = np.frombuffer(
            self.buffer,
            dtype=np.dtype(dtype).newbyteorder("<"),
            count=count,
            offset=self.offset,
        )
        self.offset += arr.nbytes
        return arr


def load(path: str | Path) -> FastTextWeights:
    """Parse a fastText supervised model file."""
    with Path(path).open("rb") as fh:
        buffer = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    reader = _Reader(buffer)

    magic, version = reader.unpack("ii")
    if magic != FASTTEXT_MAGIC:
        raise ValueError(f"{path} is not a fastText model (bad magic number {magic})")
    if version > FASTTEXT_VERSION:
        raise ValueError(
            f"{path} uses fastText format version {version}; only <= {FASTTEXT_VERSION} is supported"
        )
    raw_args = dict(zip(_ARG_NAMES, reader.unpack("12i"), strict=True))
    (t,) = reader.unpack("d")
    if version == FASTTEXT_VERSION - 1 and raw_args["model"] == ModelKind.SUPERVISED:
        raw_args["maxn"] = 0  # fastText < 0.9 supervised models had no subwords
    raw_args["loss"] = Loss(raw_args["loss"])
    raw_args["model"] = ModelKind(raw_args["model"])
    args = Args(**raw_args, t=t)
    if args.model != ModelKind.SUPERVISED:
        raise ValueError(
            f"{path} is a {args.model.name.lower()} model; only supervised classifiers are supported"
        )

    words, labels, label_counts, pruneidx = _read_dictionary(reader)
    (quant_input,) = reader.unpack("?")
    input_matrix = _read_matrix(reader, quantized=quant_input)
    (qout,) = reader.unpack("?")
    output_matrix = _read_matrix(reader, quantized=quant_input and qout)
    return FastTextWeights(
        args=args,
        words=words,
        labels=labels,
        label_counts=label_counts,
        pruneidx=pruneidx,
        input_matrix=input_matrix,
        output_matrix=output_matrix,
    )


def _read_dictionary(
    reader: _Reader,
) -> tuple[list[str], list[str], np.ndarray, dict[int, int] | None]:
    size, nwords, nlabels = reader.unpack("3i")
    _ntokens, pruneidx_size = reader.unpack("qq")
    buffer, offset = reader.buffer, reader.offset
    find = buffer.find
    entries: list[bytes] = []
    counts: list[int] = []
    unpack_count = struct.Struct("<q").unpack_from
    for _ in range(size):
        end = find(b"\0", offset)
        entries.append(buffer[offset:end])
        (count,) = unpack_count(buffer, end + 1)
        counts.append(count)
        offset = end + 10  # NUL + int64 count + int8 type
    reader.offset = offset
    words = [w.decode("utf-8", errors="replace") for w in entries[:nwords]]
    labels = [
        w.decode("utf-8", errors="replace") for w in entries[nwords : nwords + nlabels]
    ]
    label_counts = np.array(counts[nwords : nwords + nlabels], dtype=np.int64)

    pruneidx = None
    if pruneidx_size >= 0:
        pairs = reader.array(np.int32, 2 * pruneidx_size).reshape(-1, 2)
        pruneidx = dict(zip(pairs[:, 0].tolist(), pairs[:, 1].tolist(), strict=True))
    return words, labels, label_counts, pruneidx


def _read_matrix(reader: _Reader, *, quantized: bool) -> np.ndarray:
    if not quantized:
        m, n = reader.unpack("qq")
        return reader.array(np.float32, m * n).reshape(m, n)

    (qnorm,) = reader.unpack("?")
    m, n = reader.unpack("qq")
    (codesize,) = reader.unpack("i")
    codes = reader.array(np.uint8, codesize)
    matrix = _decode_product_quantizer(reader, codes, m)
    if qnorm:
        norm_codes = reader.array(np.uint8, m)
        norms = _decode_product_quantizer(reader, norm_codes, m)
        matrix *= norms
    if matrix.shape != (m, n):
        raise ValueError(
            f"quantized matrix decoded to {matrix.shape}, expected {(m, n)}"
        )
    return matrix


def _decode_product_quantizer(reader: _Reader, codes: np.ndarray, m: int) -> np.ndarray:
    """Rebuild the dense ``(m, dim)`` float32 matrix encoded by a fastText ProductQuantizer."""
    dim, nsubq, dsub, lastdsub = reader.unpack("4i")
    centroids = reader.array(np.float32, dim * PQ_KSUB)
    codes = codes.reshape(m, nsubq).astype(np.intp)
    parts = []
    for j in range(nsubq):
        width = lastdsub if j == nsubq - 1 else dsub
        # Sub-quantizer j's centroids start at j * ksub * dsub; each has `width` floats.
        table = centroids[
            j * PQ_KSUB * dsub : j * PQ_KSUB * dsub + PQ_KSUB * width
        ].reshape(PQ_KSUB, width)
        parts.append(table[codes[:, j]])
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)
