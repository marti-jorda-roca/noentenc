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

``word_rows_batch`` hashes the n-grams of many words at once with numpy; ``word_rows`` and
``subword_rows`` are the plain per-word versions of the same rules.
"""

import functools
import re
import sys
from collections.abc import Callable
from itertools import chain, repeat

import numpy as np

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
_XOR_ARRAY = np.array(_XOR_TABLE, dtype=np.uint32)
_FNV_PRIME_U32 = np.uint32(FNV_PRIME)
_FNV_OFFSET_U32 = np.uint32(FNV_OFFSET_BASIS)
# The longest UTF-8 encoding of one code point.
_MAX_CHAR_BYTES = 4
_PRUNED_U16 = np.iinfo(np.uint16).max
# Up to this many words, hashing them one by one beats the fixed cost of the numpy pass.
_PER_WORD_MAX = 5
# fastText's word separators: the only characters `readWord` splits on.
_FASTTEXT_SPACES = " \n\r\t\v\f\0"
_SEPARATORS = str.maketrans(dict.fromkeys(_FASTTEXT_SPACES, " "))
# Where `str.split()` and fastText disagree: Unicode spaces fastText keeps inside words, and NUL,
# which only fastText splits on. Texts without them take the faster `str.split()`.
_SPLIT_MISMATCH = re.compile(
    "["
    + re.escape(
        "".join(
            c
            for c in map(chr, range(sys.maxunicode + 1))
            if (c.isspace() or c == "\0") and c not in _FASTTEXT_SPACES[:-1]
        )
    )
    + "]"
)


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
    if _SPLIT_MISMATCH.search(text) is None:
        return text.split()
    return list(filter(None, text.translate(_SEPARATORS).split(" ")))


class Tokenizer:
    """Maps words and texts to input-matrix row ids."""

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
        self.word_hash: Callable[[str], int] = (
            _signed_fnv
            if cache_size == 0
            else functools.lru_cache(maxsize=cache_size)(_signed_fnv)
        )
        self._eos_id = self.eos_rows[0] if self.eos_rows else -1
        # Quantized models keep only some buckets: a dense bucket -> kept-index table (the
        # dtype's max marks dropped buckets) replaces the dict lookups in `word_rows_batch`.
        self._prune_table: np.ndarray | None = None
        if self.pruneidx is not None:
            dtype = np.uint16 if len(self.pruneidx) < _PRUNED_U16 else np.uint32
            table = np.full(self.bucket, np.iinfo(dtype).max, dtype=dtype)
            table[np.fromiter(self.pruneidx.keys(), np.int64, len(self.pruneidx))] = (
                np.fromiter(self.pruneidx.values(), np.int64, len(self.pruneidx))
            )
            self._prune_table = table

    def word_rows(self, word: str) -> tuple[int, ...]:
        """Rows of one word: its own row if in the vocabulary, then its character n-grams."""
        wid = self.word_ids.get(word)
        if wid is None:
            if word.startswith(FASTTEXT_LABEL_PREFIX):
                return ()  # fastText treats these tokens as labels, never as input
            return tuple(self.subword_rows(word))
        if self.maxn <= 0 or word == EOS:
            return (wid,)
        return (wid, *self.subword_rows(word))

    def word_rows_batch(self, words: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """``word_rows`` of many words at once, as ``(owner, rows)`` arrays.

        ``rows`` lists every word's rows in ``word_rows`` order, word after word, and ``owner``
        holds the index in ``words`` that each row belongs to. The empty string, which
        ``split_words`` never returns, stands for the end of a text and gets ``eos_rows``.
        """
        n = len(words)
        if n <= _PER_WORD_MAX:
            per_word = [self.word_rows(w) if w else self.eos_rows for w in words]
            owner = np.repeat(np.arange(n), [len(rows) for rows in per_word])
            return owner, np.fromiter(chain.from_iterable(per_word), np.int64)
        own = np.fromiter(map(self.word_ids.get, words, repeat(-1)), np.int64, n)
        # `</s>` (and the end-of-text marker standing for it) has no n-grams; OOV label
        # tokens have no rows at all.
        no_subwords = np.zeros(n, dtype=bool)
        for special in ("", EOS) if self.eos_rows else ("",):
            at = _index(words, special)
            if at >= 0:
                own[at] = self._eos_id
                no_subwords[at] = True
        no_subwords |= (own < 0) & np.fromiter(
            map(str.startswith, words, repeat(FASTTEXT_LABEL_PREFIX)), bool, n
        )
        in_vocab = np.flatnonzero(own >= 0)
        if self.maxn <= 0:
            return in_vocab, own[in_vocab]
        owner, rows = self._subword_rows_batch(words)
        if no_subwords.any():
            keep = ~no_subwords[owner]
            owner, rows = owner[keep], rows[keep]
        # Each in-vocabulary word's own row goes in front of its n-grams.
        first = np.searchsorted(owner, in_vocab)
        return np.insert(owner, first, in_vocab), np.insert(rows, first, own[in_vocab])

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

    def _subword_rows_batch(self, words: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """``subword_rows`` of many words in one pass of numpy ops, as ``(owner, rows)``.

        All n-grams that start on the same character share one running FNV hash, so the loop
        runs ``maxn`` times (once per n-gram length) whatever the number of words. ``uint32``
        arithmetic wraps exactly like fastText's. Hashes land in a ``(chars, maxn)`` matrix, so
        reading the valid cells row by row gives fastText's order: by start, then by length.
        """
        if not words:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        # NUL never occurs inside a word (it is a separator), so it delimits the words.
        blob = (BOW + (EOW + "\0" + BOW).join(words) + EOW).encode(
            "utf-8", errors="replace"
        )
        data = np.frombuffer(blob, dtype=np.uint8)
        heads = np.flatnonzero(data & _UTF8_CONTINUATION_MASK != _UTF8_CONTINUATION)
        is_char = data[heads] != 0
        char_start = heads[is_char]
        char_end = np.append(heads[1:], len(blob))[is_char]
        char_owner = np.cumsum(~is_char)[is_char]
        n_chars = len(char_start)
        # Characters left in the word from each character on, itself included.
        word_end = np.cumsum(np.bincount(char_owner, minlength=len(words)))
        left = word_end[char_owner] - np.arange(n_chars)
        # Per-byte XOR operands, with zero padding so that n-grams running past the last word
        # read valid memory (their hashes are never emitted).
        operand = np.zeros(len(blob) + _MAX_CHAR_BYTES, dtype=np.uint32)
        operand[: len(blob)] = _XOR_ARRAY[data]
        width = np.append(char_end - char_start, np.ones(self.maxn, dtype=np.intp))
        char_start = np.append(char_start, np.full(self.maxn, len(blob)))

        table = np.empty((n_chars, self.maxn), dtype=np.uint32)
        h = np.full(n_chars, _FNV_OFFSET_U32, dtype=np.uint32)
        step = np.empty_like(h)
        for length in range(1, self.maxn + 1):
            last = slice(length - 1, length - 1 + n_chars)
            pos = char_start[last]
            last_width = width[last]
            for k in range(int(last_width.max())):
                np.bitwise_xor(h, operand[pos + k], out=step)
                np.multiply(step, _FNV_PRIME_U32, out=step)
                if k == 0:
                    h, step = step, h
                else:
                    np.copyto(h, step, where=last_width > k)
            table[:, length - 1] = h

        valid = left[:, None] >= np.arange(1, self.maxn + 1)
        valid[:, : max(self.minn - 1, 0)] = False
        if self.minn <= 1:
            # The single characters `<` and `>` are not n-grams.
            first = np.r_[True, char_owner[1:] != char_owner[:-1]]
            valid[:, 0] &= ~(first | (left == 1))
        owner = np.broadcast_to(char_owner[:, None], table.shape)[valid]
        ids = table[valid] % np.uint32(self.bucket)
        if self._prune_table is None:
            return owner, self.nwords + ids.astype(np.int64)
        kept_ids = self._prune_table[ids]
        kept = kept_ids != np.iinfo(kept_ids.dtype).max
        return owner[kept], self.nwords + kept_ids[kept].astype(np.int64)

    def _push_hashes(self, hashes: list[int]) -> list[int]:
        """``Dictionary::pushHash``: offset by ``nwords``, remapping/dropping ids for pruned models."""
        nwords = self.nwords
        pruneidx = self.pruneidx
        if pruneidx is None:
            return [nwords + h for h in hashes]
        return [nwords + pruneidx[h] for h in hashes if h in pruneidx]


def _index(items: list[str], item: str) -> int:
    try:
        return items.index(item)
    except ValueError:
        return -1
