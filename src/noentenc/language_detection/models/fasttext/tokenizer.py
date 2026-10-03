"""fastText's supervised-model tokenization, reproduced bit-for-bit (``src/dictionary.cc``).

A text becomes the list of input-matrix rows whose mean is the hidden vector:

* split on the ASCII whitespace fastText recognises (not on Unicode spaces such as U+00A0);
* append the end-of-sentence token ``</s>``;
* an in-vocabulary word contributes its own row plus its character n-grams; an out-of-vocabulary
  word contributes only its n-grams; ``</s>`` contributes only its own row;
* character n-grams of ``<word>`` have ``minn..maxn`` UTF-8 code points and are hashed with
  32-bit FNV-1a into ``bucket`` rows after the vocabulary (remapped through ``pruneidx`` for
  quantized models).

One deliberate difference: fastText stops reading at the first newline, silently ignoring the rest
of the text, so here newlines are treated as ordinary spaces.
"""

import functools
from collections.abc import Callable

from noentenc.language_detection.labels import FASTTEXT_LABEL_PREFIX
from noentenc.language_detection.models.fasttext.format import FastTextWeights

EOS = "</s>"
BOW = "<"
EOW = ">"
FNV_OFFSET_BASIS = 2166136261
FNV_PRIME = 16777619
UINT32_MASK = 0xFFFFFFFF
UINT64_MASK = 0xFFFFFFFFFFFFFFFF
WORD_NGRAM_MULTIPLIER = 116049371
_INT32_SIGN = 0x80000000
_UTF8_CONTINUATION_MASK = 0xC0
_UTF8_CONTINUATION = 0x80
_ASCII_LIMIT = 0x80

# fastText XORs `uint32_t(int8_t(byte))`, i.e. bytes >= 0x80 are sign-extended to 0xFFFFFFxx.
_XOR_TABLE: tuple[int, ...] = tuple(
    b if b < _ASCII_LIMIT else 0xFFFFFF00 | b for b in range(256)
)
# fastText's word separators: the only characters `readWord` splits on.
_SEPARATORS = str.maketrans(dict.fromkeys("\n\r\t\v\f\0", " "))


def fnv1a_32(data: bytes) -> int:
    """fastText's ``Dictionary::hash`` (FNV-1a with sign-extended bytes)."""
    h = FNV_OFFSET_BASIS
    for byte in data:
        h = ((h ^ _XOR_TABLE[byte]) * FNV_PRIME) & UINT32_MASK
    return h


def _signed_fnv(word: str) -> int:
    """fastText stores word hashes as ``int32_t``, so they are sign-extended when combined."""
    h = fnv1a_32(word.encode("utf-8", errors="replace"))
    return h - (1 << 32) if h & _INT32_SIGN else h


def split_words(text: str) -> list[str]:
    """Split ``text`` the way fastText's ``readWord`` does, with newlines treated as spaces."""
    return [word for word in text.translate(_SEPARATORS).split(" ") if word]


class Tokenizer:
    """Maps texts to input-matrix row ids, caching the rows of each distinct word."""

    def __init__(
        self, weights: FastTextWeights, cache_size: int | None = 1 << 17
    ) -> None:
        args = weights.args
        self.minn = args.minn
        self.maxn = args.maxn
        self.bucket = args.bucket
        self.nwords = weights.nwords
        self.word_ids = weights.word_ids
        self.pruneidx = weights.pruneidx
        self.eos_rows: tuple[int, ...] = (
            (self.word_ids[EOS],) if EOS in self.word_ids else ()
        )
        self.word_ngrams = args.word_ngrams
        compute = self._compute_word_rows
        self.word_rows: Callable[[str], tuple[int, ...]] = (
            compute
            if cache_size == 0
            else functools.lru_cache(maxsize=cache_size)(compute)
        )
        self.word_hash: Callable[[str], int] = (
            _signed_fnv
            if cache_size == 0
            else functools.lru_cache(maxsize=cache_size)(_signed_fnv)
        )

    def text_rows(self, text: str) -> list[int]:
        """Input-matrix rows for ``text`` (never empty: ``</s>`` is always appended)."""
        rows: list[int] = []
        word_rows = self.word_rows
        words = split_words(text)
        for word in words:
            rows.extend(word_rows(word))
        rows.extend(self.eos_rows)
        if self.word_ngrams > 1:
            rows.extend(self.word_ngram_rows(words))
        return rows

    def word_ngram_rows(self, words: list[str]) -> list[int]:
        """Rows of the hashed word n-grams (``Dictionary::addWordNgrams``)."""
        word_hash = self.word_hash
        hashes = [
            word_hash(w) for w in words if not w.startswith(FASTTEXT_LABEL_PREFIX)
        ]
        hashes.append(word_hash(EOS))
        ids: list[int] = []
        n, bucket = self.word_ngrams, self.bucket
        for i in range(len(hashes)):
            h = hashes[i] & UINT64_MASK
            for j in range(i + 1, min(i + n, len(hashes))):
                h = (h * WORD_NGRAM_MULTIPLIER + hashes[j]) & UINT64_MASK
                ids.append(h % bucket)
        return self._push_hashes(ids)

    def _compute_word_rows(self, word: str) -> tuple[int, ...]:
        wid = self.word_ids.get(word)
        if wid is None:
            if word.startswith(FASTTEXT_LABEL_PREFIX):
                return ()  # fastText treats these tokens as labels, never as input
            return tuple(self.subword_rows(word))
        if self.maxn <= 0 or word == EOS:
            return (wid,)
        return (wid, *self.subword_rows(word))

    def subword_rows(self, word: str) -> list[int]:
        """Rows of the character n-grams of ``<word>`` (``Dictionary::computeSubwords``).

        The FNV hash is extended byte by byte as each n-gram grows, instead of rehashing every
        n-gram from scratch.
        """
        data = (BOW + word + EOW).encode("utf-8", errors="replace")
        size = len(data)
        starts = [
            k
            for k in range(size)
            if data[k] & _UTF8_CONTINUATION_MASK != _UTF8_CONTINUATION
        ]
        starts.append(size)
        n_chars = len(starts) - 1
        minn, maxn, bucket = self.minn, self.maxn, self.bucket
        xor = _XOR_TABLE
        hashes: list[int] = []
        for first in range(n_chars):
            h = FNV_OFFSET_BASIS
            last = min(first + maxn, n_chars)
            for end in range(first + 1, last + 1):
                for byte in data[starts[end - 1] : starts[end]]:
                    h = ((h ^ xor[byte]) * FNV_PRIME) & UINT32_MASK
                length = end - first
                if length >= minn and not (
                    length == 1 and (first == 0 or end == n_chars)
                ):
                    hashes.append(h % bucket)
        return self._push_hashes(hashes)

    def _push_hashes(self, hashes: list[int]) -> list[int]:
        """``Dictionary::pushHash``: offset by ``nwords``, remapping/dropping ids for pruned models."""
        nwords = self.nwords
        pruneidx = self.pruneidx
        if pruneidx is None:
            return [nwords + h for h in hashes]
        return [nwords + pruneidx[h] for h in hashes if h in pruneidx]
