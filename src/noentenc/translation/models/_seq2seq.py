from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from noentenc._batching import check_batch_size
from noentenc._cache import translation_cache
from noentenc._onnx import pad
from noentenc._optional import TRANSLATION_EXTRA, require
from noentenc.languages import Language
from noentenc.translation._segment import split_sentences, split_to_fit
from noentenc.translation.models._engine import (
    DecoderShape,
    GenerationConfig,
    Seq2SeqOnnxEngine,
)
from noentenc.translation.models._hub import resolve_files
from noentenc.translation.models.base import (
    BaseModel,
    InputTooLongError,
    Translation,
)

if TYPE_CHECKING:
    from tokenizers import Tokenizer

_PREVIEW_LENGTH = 60

# Sentences tokenized at once. Bounds the tokenizer's memory on large calls.
_TOKENIZE_WINDOW = 2048

# With `split_long`, pieces of a long sentence are cut this share of the limit short of
# it: a piece tokenized on its own can take a token or two more than inside the sentence.
_SPLIT_MARGIN = 16
# ...and to at most this many tokens. The models are trained on sentences, and drop
# content from inputs anywhere near their limit.
_SPLIT_WINDOW = 128

# With `split_long`, a sentence whose output hits the length limit is translated again in
# halves, unless it's shorter than this: halving a few tokens doesn't help.
_MIN_RETRY_TOKENS = 8


class Precision(StrEnum):
    FP32 = "fp32"
    INT8 = "int8"
    Q4 = "q4"


# File names of the `Xenova/*` exports (optimum + transformers.js quantizations).
XENOVA_ONNX_FILES: dict[Precision, tuple[str, str]] = {
    Precision.FP32: ("onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"),
    Precision.INT8: (
        "onnx/encoder_model_quantized.onnx",
        "onnx/decoder_model_merged_quantized.onnx",
    ),
    Precision.Q4: ("onnx/encoder_model_q4.onnx", "onnx/decoder_model_merged_q4.onnx"),
}


class Seq2SeqModel(BaseModel):
    """Shared split → tokenize → encode → greedy decode → detokenize → join pipeline.

    These models are trained on single sentences, so each text is split into
    sentences, they are translated in one batch, and the translations are joined back
    with the original whitespace, line breaks included.

    Subclasses describe where their files live and how language tokens frame the
    input; the decoding itself is done by `Seq2SeqOnnxEngine`.

    Hugging Face repos download into `<cache_dir>/hub` (see `noentenc._cache` for the
    default).
    """

    default_model: ClassVar[str]
    # Commit of `default_model` to download; other repos use the `revision` argument.
    default_revision: ClassVar[str]
    default_precision: ClassVar[Precision]
    # (encoder, decoder) ONNX filenames inside the model repo, per precision.
    onnx_files: ClassVar[dict[Precision, tuple[str, str]]] = XENOVA_ONNX_FILES
    extra_files: ClassVar[tuple[str, ...]] = ()
    # SPDX licence of the default weights; None when it differs per model (Opus-MT).
    weights_license: ClassVar[str | None] = None

    def __init__(
        self,
        model: str | Path | None = None,
        only_local_files: bool = False,
        *,
        revision: str | None = None,
        precision: Precision | str | None = None,
        num_threads: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        # Checked before downloading weights that couldn't be run.
        for module in ("tokenizers", "onnxruntime"):
            require(module, TRANSLATION_EXTRA)
        if model is None:
            model, revision = self.default_model, self.default_revision
        super().__init__(model, only_local_files)
        self.precision = self._check_precision(precision)
        encoder, decoder = self.onnx_files[self.precision]
        files = resolve_files(
            self.model,
            self.filenames(self.precision),
            only_local_files,
            revision=revision,
            cache_dir=translation_cache(cache_dir),
        )
        self.config = json.loads(files["config.json"].read_text())
        self.tokenizer = self._load_tokenizer(files)
        self.max_length: int = self.config["max_position_embeddings"]
        self._generation = GenerationConfig(
            decoder_start_id=self.config["decoder_start_token_id"],
            eos_id=self.config["eos_token_id"],
            pad_id=self.config["pad_token_id"],
            max_new_tokens=self.max_length,
            banned_ids=self._banned_ids(),
        )
        self._engine = Seq2SeqOnnxEngine(
            files[encoder],
            files[decoder],
            DecoderShape.from_config(self.config),
            num_threads,
        )

    @classmethod
    def filenames(cls, precision: Precision | str | None = None) -> list[str]:
        """The files a model of this class loads at `precision` (`None`: the default)."""
        encoder, decoder = cls.onnx_files[cls._check_precision(precision)]
        return ["config.json", "tokenizer.json", encoder, decoder, *cls.extra_files]

    @classmethod
    def download(
        cls,
        model: str | None = None,
        *,
        revision: str | None = None,
        precision: Precision | str | None = None,
        cache_dir: str | Path | None = None,
        force: bool = False,
    ) -> dict[str, Path]:
        """Download a model's files without loading it; return their local paths.

        Takes the same `model`, `revision`, `precision` and `cache_dir` as the
        constructor, which then loads the files offline with `only_local_files=True`.
        `force` downloads them again, e.g. to replace a corrupt file.
        """
        if model is None:
            model, revision = cls.default_model, cls.default_revision
        return resolve_files(
            model,
            cls.filenames(precision),
            revision=revision,
            cache_dir=translation_cache(cache_dir),
            force=force,
        )

    @classmethod
    def _check_precision(cls, precision: Precision | str | None) -> Precision:
        checked = Precision(precision or cls.default_precision)
        if checked not in cls.onnx_files:
            raise ValueError(
                f"{cls.__name__} has no {checked} weights; "
                f"available: {[str(p) for p in cls.onnx_files]}"
            )
        return checked

    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
        split_long: bool = False,
    ) -> list[str]:
        translations = self.predict_batch_detailed(
            texts,
            target_language,
            source_language,
            batch_size,
            truncate=truncate,
            split_long=split_long,
        )
        return [translation.text for translation in translations]

    def predict_batch_detailed(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
        split_long: bool = False,
    ) -> list[Translation]:
        """Translate each text sentence by sentence, keeping the whitespace between them.

        A sentence longer than the model reads raises `InputTooLongError`, or with
        `truncate` loses its end. With `split_long` it is cut into pieces that fit
        instead, at clause punctuation or whitespace, and each is translated; a
        sentence whose output hits the length limit is also translated again in halves.
        `batch_size` counts sentences. A sentence that occurs more than once in `texts`
        is translated once.
        """
        check_batch_size(batch_size)
        target = Language(target_language)
        source = None if source_language is None else Language(source_language)
        self.schema.validate(source, target, type(self).__name__)

        documents = [split_sentences(text) for text in texts]
        # Each distinct sentence, in order of first appearance, and its position.
        slots: dict[str, int] = {}
        for document in documents:
            for sentence in document.sentences:
                slots.setdefault(sentence, len(slots))
        translated = self._translate_sentences(
            list(slots), source, target, batch_size, truncate, split_long
        )
        results: list[Translation] = []
        for document in documents:
            at = [slots[sentence] for sentence in document.sentences]
            results.append(
                Translation(
                    document.join([translated.outputs[i] for i in at]),
                    input_truncated=any(translated.truncated[i] for i in at),
                    output_limit_reached=any(translated.exhausted[i] for i in at),
                    input_split=any(translated.split[i] for i in at),
                )
            )
        return results

    def _translate_sentences(
        self,
        sentences: list[str],
        source: Language | None,
        target: Language,
        batch_size: int,
        truncate: bool,
        split_long: bool = False,
        *,
        retry: bool = True,
    ) -> _Sentences:
        """Translate `sentences`, and say which were truncated, cut short or split."""
        if not sentences:
            return _Sentences([], [], [], [])
        units = self._encode(sentences, source, target, truncate, split_long)
        outputs, exhausted_units = self._generate(units.sequences, target, batch_size)
        result = _Sentences(
            [""] * len(sentences),
            units.truncated,
            [False] * len(sentences),
            units.split,
        )
        for unit, owner in enumerate(units.owners):
            result.outputs[owner] += outputs[unit] + units.separators[unit]
            result.exhausted[owner] |= exhausted_units[unit]
        if split_long and retry:
            self._retry_exhausted(
                sentences, result, source, target, batch_size, truncate
            )
        return result

    def _encode(
        self,
        sentences: list[str],
        source: Language | None,
        target: Language,
        truncate: bool,
        split_long: bool,
    ) -> _Units:
        """Token ids of each sentence, or of its pieces when `split_long` cuts it."""
        # Leave room for the language tokens and </s> added by `_frame`.
        limit = self.max_length - len(self._frame([], source, target))
        units = _Units(
            truncated=[False] * len(sentences), split=[False] * len(sentences)
        )
        # A window at a time, keeping only the ids, so the tokenizer's `Encoding`s
        # (offsets, masks, token strings) never exist for the whole call at once.
        for start in range(0, len(sentences), _TOKENIZE_WINDOW):
            encodings = self.tokenizer.encode_batch(
                sentences[start : start + _TOKENIZE_WINDOW], add_special_tokens=False
            )
            for i, encoding in enumerate(encodings, start):
                ids = encoding.ids
                if len(ids) <= limit:
                    units.add(i, self._frame(ids, source, target))
                elif split_long:
                    units.split[i] = True
                    window = min(_SPLIT_WINDOW, limit - max(1, limit // _SPLIT_MARGIN))
                    pieces = split_to_fit(sentences[i], encoding.offsets, window)
                    self._add_pieces(units, i, pieces, limit, source, target)
                elif truncate:
                    units.truncated[i] = True
                    units.add(i, self._frame(ids[:limit], source, target))
                else:
                    raise InputTooLongError(
                        f"{type(self).__name__} reads at most {limit} tokens per "
                        f"sentence, and this one has {len(ids)}: "
                        f"{_preview(sentences[i])!r}. Split it, pass split_long=True "
                        "to translate it in pieces, or truncate=True to translate only "
                        "its start."
                    )
        return units

    def _add_pieces(
        self,
        units: _Units,
        owner: int,
        pieces: list[tuple[str, str]],
        limit: int,
        source: Language | None,
        target: Language,
    ) -> None:
        encodings = self.tokenizer.encode_batch(
            [piece for piece, _ in pieces], add_special_tokens=False
        )
        for (_, separator), encoding in zip(pieces, encodings, strict=True):
            # Only a single token run longer than the window can still be too long.
            ids = encoding.ids
            units.truncated[owner] |= len(ids) > limit
            units.add(owner, self._frame(ids[:limit], source, target), separator)

    def _generate(
        self, sequences: list[list[int]], target: Language, batch_size: int
    ) -> tuple[list[str], list[bool]]:
        """Decoded translations, and which ran out of output tokens."""
        generation = replace(
            self._generation, forced_first_id=self._forced_first_id(target)
        )
        # Similar lengths in the same batch keep padding (and wasted compute) low.
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        outputs: list[list[int]] = [[] for _ in sequences]
        exhausted = [False] * len(sequences)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            input_ids, attention_mask = pad(
                [sequences[i] for i in chunk], generation.pad_id
            )
            max_new_tokens = min(self.max_length, 2 * input_ids.shape[1] + 10)
            chunk_generation = replace(generation, max_new_tokens=max_new_tokens)
            generated = self._engine.generate(
                input_ids, attention_mask, chunk_generation
            )
            for i, ids in zip(chunk, generated, strict=True):
                outputs[i] = ids
                # Rows are cut at </s>, so a row that fills the budget never ended.
                exhausted[i] = len(ids) >= max_new_tokens
        decoded = self.tokenizer.decode_batch(outputs, skip_special_tokens=True)
        return [text.strip() for text in decoded], exhausted

    def _retry_exhausted(
        self,
        sentences: list[str],
        result: _Sentences,
        source: Language | None,
        target: Language,
        batch_size: int,
        truncate: bool,
    ) -> None:
        """Translate the sentences whose output hit the length limit again, in halves.

        Their output usually repeats itself until the limit. The halves replace it when
        neither of them hits the limit; otherwise the first output stays, flagged.
        """
        retry = [i for i, exhausted in enumerate(result.exhausted) if exhausted]
        if not retry:
            return
        encodings = self.tokenizer.encode_batch(
            [sentences[i] for i in retry], add_special_tokens=False
        )
        halves: dict[int, list[tuple[str, str]]] = {}
        for i, encoding in zip(retry, encodings, strict=True):
            size = len(encoding.ids)
            if size >= _MIN_RETRY_TOKENS:
                pieces = split_to_fit(sentences[i], encoding.offsets, (size + 1) // 2)
                if len(pieces) > 1:
                    halves[i] = pieces
        if not halves:
            return
        translated = self._translate_sentences(
            [piece for pieces in halves.values() for piece, _ in pieces],
            source,
            target,
            batch_size,
            truncate,
            split_long=True,
            retry=False,
        )
        position = 0
        for i, pieces in halves.items():
            span = range(position, position + len(pieces))
            position += len(pieces)
            if any(translated.exhausted[j] for j in span):
                continue
            result.outputs[i] = "".join(
                translated.outputs[j] + separator
                for j, (_, separator) in zip(span, pieces, strict=True)
            )
            result.exhausted[i] = False
            result.split[i] = True
            result.truncated[i] |= any(translated.truncated[j] for j in span)

    def _load_tokenizer(self, files: dict[str, Path]) -> Tokenizer:
        return tokenizer_class().from_file(str(files["tokenizer.json"]))

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"{type(self).__name__} tokenizer has no token {token!r}")
        return token_id

    def _frame(
        self, ids: list[int], source: Language | None, target: Language
    ) -> list[int]:
        """Wrap the text token ids with the model's special/language tokens."""
        return [*ids, self.config["eos_token_id"]]

    def _forced_first_id(self, target: Language) -> int | None:
        return None

    def _banned_ids(self) -> tuple[int, ...]:
        return ()


@dataclass
class _Units:
    """What the model translates: whole sentences, or the pieces of split ones."""

    sequences: list[list[int]] = field(default_factory=list)
    # The sentence each unit belongs to, and the whitespace after it in that sentence.
    owners: list[int] = field(default_factory=list)
    separators: list[str] = field(default_factory=list)
    # Per sentence: whether input was dropped, and whether it was cut into pieces.
    truncated: list[bool] = field(default_factory=list)
    split: list[bool] = field(default_factory=list)

    def add(self, owner: int, sequence: list[int], separator: str = "") -> None:
        self.sequences.append(sequence)
        self.owners.append(owner)
        self.separators.append(separator)


@dataclass
class _Sentences:
    """Per sentence: its translation, and whether input was dropped, output cut short
    or the sentence translated in pieces."""

    outputs: list[str]
    truncated: list[bool]
    exhausted: list[bool]
    split: list[bool]


def tokenizer_class() -> type[Tokenizer]:
    """`tokenizers.Tokenizer`, imported on first use (it comes with the translation extra)."""
    return require("tokenizers", TRANSLATION_EXTRA).Tokenizer


def _preview(text: str) -> str:
    return text if len(text) <= _PREVIEW_LENGTH else text[:_PREVIEW_LENGTH] + "…"
