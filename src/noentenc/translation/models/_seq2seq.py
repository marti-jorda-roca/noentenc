import json
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from tokenizers import Tokenizer

from noentenc._batching import check_batch_size
from noentenc._onnx import pad
from noentenc.languages import Language
from noentenc.translation._segment import split_sentences
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

_PREVIEW_LENGTH = 60


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
    """

    default_model: ClassVar[str]
    # Commit of `default_model` to download; other repos use the `revision` argument.
    default_revision: ClassVar[str]
    default_precision: ClassVar[Precision]
    # (encoder, decoder) ONNX filenames inside the model repo, per precision.
    onnx_files: ClassVar[dict[Precision, tuple[str, str]]] = XENOVA_ONNX_FILES
    extra_files: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        model: str | Path | None = None,
        only_local_files: bool = False,
        *,
        revision: str | None = None,
        precision: Precision | str | None = None,
        num_threads: int | None = None,
    ) -> None:
        if model is None:
            model, revision = self.default_model, self.default_revision
        super().__init__(model, only_local_files)
        self.precision = Precision(precision or self.default_precision)
        if self.precision not in self.onnx_files:
            raise ValueError(
                f"{type(self).__name__} has no {self.precision} weights; "
                f"available: {[str(p) for p in self.onnx_files]}"
            )
        encoder, decoder = self.onnx_files[self.precision]
        files = resolve_files(
            self.model,
            ["config.json", "tokenizer.json", encoder, decoder, *self.extra_files],
            only_local_files,
            revision=revision,
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

    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
    ) -> list[str]:
        translations = self.predict_batch_detailed(
            texts, target_language, source_language, batch_size, truncate=truncate
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
    ) -> list[Translation]:
        """Translate each text sentence by sentence, keeping the whitespace between them.

        A sentence longer than the model reads raises `InputTooLongError`, or with
        `truncate` loses its end. `batch_size` counts sentences.
        """
        check_batch_size(batch_size)
        target = Language(target_language)
        source = None if source_language is None else Language(source_language)
        self.schema.validate(source, target, type(self).__name__)

        documents = [split_sentences(text) for text in texts]
        sentences = [
            sentence for document in documents for sentence in document.sentences
        ]
        outputs, truncated, exhausted = self._translate_sentences(
            sentences, source, target, batch_size, truncate
        )
        results: list[Translation] = []
        start = 0
        for document in documents:
            span = slice(start, start + len(document.sentences))
            start = span.stop
            results.append(
                Translation(
                    document.join(outputs[span]),
                    input_truncated=any(truncated[span]),
                    output_limit_reached=any(exhausted[span]),
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
    ) -> tuple[list[str], list[bool], list[bool]]:
        """Translations, and which sentences were truncated or ran out of output tokens."""
        if not sentences:
            return [], [], []
        # Leave room for the language tokens and </s> added by `_frame`.
        limit = self.max_length - len(self._frame([], source, target))
        encodings = self.tokenizer.encode_batch(sentences, add_special_tokens=False)
        truncated = [len(encoding.ids) > limit for encoding in encodings]
        if any(truncated) and not truncate:
            i = truncated.index(True)
            raise InputTooLongError(
                f"{type(self).__name__} reads at most {limit} tokens per sentence, and "
                f"this one has {len(encodings[i].ids)}: {_preview(sentences[i])!r}. "
                "Split it, or pass truncate=True to translate only its start."
            )
        sequences = [self._frame(e.ids[:limit], source, target) for e in encodings]
        generation = replace(
            self._generation, forced_first_id=self._forced_first_id(target)
        )

        # Similar lengths in the same batch keep padding (and wasted compute) low.
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        outputs: list[list[int]] = [[] for _ in sentences]
        exhausted = [False] * len(sentences)
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
        return [text.strip() for text in decoded], truncated, exhausted

    def _load_tokenizer(self, files: dict[str, Path]) -> Tokenizer:
        return Tokenizer.from_file(str(files["tokenizer.json"]))

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


def _preview(text: str) -> str:
    return text if len(text) <= _PREVIEW_LENGTH else text[:_PREVIEW_LENGTH] + "…"
