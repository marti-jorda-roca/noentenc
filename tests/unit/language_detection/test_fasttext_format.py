import numpy as np
import pytest

from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.language_detection.models.fasttext import format as ftz
from noentenc.language_detection.models.fasttext.format import Loss
from tests.unit.language_detection.helpers import (
    dense_block,
    quant_block,
    write_fasttext_model,
)

WORDS = ["</s>", "hello", "world"]
LABELS = ["__label__en", "__label__es", "__label__ca"]
DIM, BUCKET = 4, 16


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_dense_model_roundtrip_and_prediction(tmp_path, rng) -> None:  # noqa: ANN001
    w_in = rng.normal(size=(len(WORDS) + BUCKET, DIM)).astype(np.float32)
    w_out = rng.normal(size=(len(LABELS), DIM)).astype(np.float32)
    path = write_fasttext_model(
        tmp_path / "m.bin",
        words=WORDS,
        labels=LABELS,
        input_block=dense_block(w_in),
        output_block=dense_block(w_out),
        dim=DIM,
        bucket=BUCKET,
    )
    weights = ftz.load(path)
    assert weights.words == WORDS and weights.labels == LABELS
    assert weights.args.loss == Loss.SOFTMAX
    np.testing.assert_array_equal(weights.input_matrix, w_in)
    np.testing.assert_array_equal(weights.output_matrix, w_out)

    model = FastTextModel(path, normalize_labels=False)
    rows = model._tokenizer.text_rows("hello there")
    hidden = w_in[rows].mean(axis=0)
    logits = w_out @ hidden
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    got = model.predict_score("hello there")
    assert list(got) == [
        LABELS[i].removeprefix("__label__") for i in np.argsort(-probs)
    ]
    np.testing.assert_allclose(list(got.values()), np.sort(probs)[::-1], rtol=1e-5)
    assert model.predict("hello there") == LABELS[int(probs.argmax())].removeprefix(
        "__label__"
    )


def test_quantized_input_and_output_with_norms(tmp_path, rng) -> None:  # noqa: ANN001
    dsub, nsubq = 3, 2  # dim 4 = 3 + lastdsub 1
    m_in = len(WORDS) + BUCKET
    codes_in = rng.integers(0, 256, size=(m_in, nsubq))
    cents_in = rng.normal(size=DIM * 256).astype(np.float32)
    norm_codes = rng.integers(0, 256, size=m_in)
    norm_cents = rng.uniform(0.5, 2.0, size=256).astype(np.float32)
    codes_out = rng.integers(0, 256, size=(len(LABELS), nsubq))
    cents_out = rng.normal(size=DIM * 256).astype(np.float32)
    path = write_fasttext_model(
        tmp_path / "m.ftz",
        words=WORDS,
        labels=LABELS,
        dim=DIM,
        bucket=BUCKET,
        quant_input=True,
        qout=True,
        input_block=quant_block(
            codes_in,
            cents_in,
            dim=DIM,
            dsub=dsub,
            norm_codes=norm_codes,
            norm_centroids=norm_cents,
        ),
        output_block=quant_block(codes_out, cents_out, dim=DIM, dsub=dsub),
    )

    def decode(codes: np.ndarray, cents: np.ndarray) -> np.ndarray:
        # productquantizer.cc get_centroids: sub-quantizer j < nsubq-1 at (j*256 + code)*dsub,
        # the last one at j*256*dsub + code*lastdsub.
        rows = []
        for code in codes:
            first = cents[code[0] * dsub : code[0] * dsub + dsub]
            last = cents[256 * dsub + code[1] : 256 * dsub + code[1] + 1]
            rows.append(np.concatenate([first, last]))
        return np.array(rows, dtype=np.float32)

    weights = ftz.load(path)
    expected_in = decode(codes_in, cents_in) * norm_cents[norm_codes][:, None]
    np.testing.assert_allclose(weights.input_matrix, expected_in, rtol=1e-6)
    np.testing.assert_allclose(
        weights.output_matrix, decode(codes_out, cents_out), rtol=1e-6
    )


def test_output_matrix_is_dense_when_input_is_not_quantized(tmp_path, rng) -> None:  # noqa: ANN001
    w_in = rng.normal(size=(len(WORDS) + BUCKET, DIM)).astype(np.float32)
    w_out = rng.normal(size=(len(LABELS), DIM)).astype(np.float32)
    # fastText only reads a QuantMatrix for the output when quant_input is also set.
    path = write_fasttext_model(
        tmp_path / "m.bin",
        words=WORDS,
        labels=LABELS,
        input_block=dense_block(w_in),
        output_block=dense_block(w_out),
        qout=True,
        dim=DIM,
        bucket=BUCKET,
    )
    np.testing.assert_array_equal(ftz.load(path).output_matrix, w_out)


def test_rejects_non_fasttext_files(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "junk.bin"
    path.write_bytes(b"\0" * 64)
    with pytest.raises(ValueError, match="not a fastText model"):
        ftz.load(path)


def test_unknown_preset_or_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="neither a fastText preset"):
        FastTextModel("does-not-exist")
