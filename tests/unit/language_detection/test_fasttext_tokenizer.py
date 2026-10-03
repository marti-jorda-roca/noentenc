import numpy as np
import pytest

from noentenc.language_detection.models.fasttext.format import (
    Args,
    FastTextWeights,
    Loss,
    ModelKind,
)
from noentenc.language_detection.models.fasttext.tokenizer import (
    EOS,
    Tokenizer,
    fnv1a_32,
    split_words,
)


def _weights(
    words: list[str],
    *,
    minn: int = 2,
    maxn: int = 3,
    bucket: int = 1000,
    pruneidx: dict[int, int] | None = None,
    word_ngrams: int = 1,
) -> FastTextWeights:
    args = Args(
        dim=2,
        ws=5,
        epoch=1,
        min_count=1,
        neg=5,
        word_ngrams=word_ngrams,
        loss=Loss.SOFTMAX,
        model=ModelKind.SUPERVISED,
        bucket=bucket,
        minn=minn,
        maxn=maxn,
        lr_update_rate=100,
        t=1e-4,
    )
    return FastTextWeights(
        args=args,
        words=words,
        labels=["__label__a"],
        label_counts=np.array([1]),
        pruneidx=pruneidx,
        input_matrix=np.zeros((len(words) + bucket, 2), np.float32),
        output_matrix=np.zeros((1, 2), np.float32),
    )


def _reference_subwords(word: str, minn: int, maxn: int) -> list[bytes]:
    """Straightforward port of Dictionary::computeSubwords, returning the n-gram bytes."""
    data = ("<" + word + ">").encode()
    ngrams = []
    for i in range(len(data)):
        if data[i] & 0xC0 == 0x80:
            continue
        ngram = b""
        j, n = i, 1
        while j < len(data) and n <= maxn:
            ngram += data[j : j + 1]
            j += 1
            while j < len(data) and data[j] & 0xC0 == 0x80:
                ngram += data[j : j + 1]
                j += 1
            if n >= minn and not (n == 1 and (i == 0 or j == len(data))):
                ngrams.append(ngram)
            n += 1
    return ngrams


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", 2166136261),
        (b"a", 0xE40C292C),  # standard FNV-1a for ASCII
        (b"foobar", 0xBF9CF968),
    ],
)
def test_fnv1a_matches_reference_for_ascii(data: bytes, expected: int) -> None:
    assert fnv1a_32(data) == expected


def test_fnv1a_sign_extends_high_bytes() -> None:
    # fastText XORs int8(byte): 0xC3 contributes 0xFFFFFFC3, not 0xC3.
    expected = ((2166136261 ^ 0xFFFFFFC3) * 16777619) & 0xFFFFFFFF
    assert fnv1a_32(b"\xc3") == expected


def test_split_words_only_splits_on_fasttext_whitespace() -> None:
    assert split_words("a  b\tc\nd\x0be\x0cf\rg\0h") == list("abcdefgh")
    assert split_words("non breaking") == ["non breaking"]
    assert split_words("   ") == []


@pytest.mark.parametrize("word", ["hello", "à", "çava", "日本語", "😀👍", "a"])
@pytest.mark.parametrize(("minn", "maxn"), [(2, 3), (1, 5), (3, 3)])
def test_subword_rows_match_reference(word: str, minn: int, maxn: int) -> None:
    tok = Tokenizer(_weights([EOS]), cache_size=0)
    tok.minn, tok.maxn = minn, maxn
    expected = [1 + fnv1a_32(ng) % 1000 for ng in _reference_subwords(word, minn, maxn)]
    assert tok.subword_rows(word) == expected


def test_text_rows_in_vocab_oov_eos_and_label_tokens() -> None:
    tok = Tokenizer(_weights([EOS, "hi"]), cache_size=0)
    hi = tok.subword_rows("hi")
    oov = tok.subword_rows("yo")
    assert tok.text_rows("hi yo __label__x") == [1, *hi, *oov, 0]
    assert tok.text_rows("") == [0]


def test_pruned_models_remap_and_drop_hashes() -> None:
    unpruned = Tokenizer(_weights([EOS]), cache_size=0).subword_rows("hello")
    keep = unpruned[0] - 1
    tok = Tokenizer(_weights([EOS], pruneidx={keep: 7}), cache_size=0)
    assert tok.subword_rows("hello") == [1 + 7]


def test_word_ngrams_follow_fasttext_hashing() -> None:
    tok = Tokenizer(_weights([EOS], word_ngrams=2, maxn=0), cache_size=0)

    def signed(word: str) -> int:
        h = fnv1a_32(word.encode())
        return h - (1 << 32) if h >= 1 << 31 else h

    hashes = [signed("a"), signed("b"), signed(EOS)]
    mask = (1 << 64) - 1
    expected = [
        ((hashes[i] & mask) * 116049371 + hashes[i + 1]) & mask for i in range(2)
    ]
    rows = tok.text_rows("a b")
    assert rows[-2:] == [1 + h % 1000 for h in expected]


def test_word_rows_are_cached() -> None:
    tok = Tokenizer(_weights([EOS]))
    tok.text_rows("hello hello")
    info = tok.word_rows.cache_info()  # ty: ignore[unresolved-attribute]
    assert (info.hits, info.misses) == (1, 1)
