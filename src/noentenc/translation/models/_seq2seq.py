import json
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from tokenizers import Tokenizer

from noentenc._onnx import pad
from noentenc.languages import Language
from noentenc.translation.models._engine import (
    DecoderShape,
    GenerationConfig,
    Seq2SeqOnnxEngine,
)
from noentenc.translation.models._hub import resolve_files
from noentenc.translation.models.base import BaseModel


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
    """Shared tokenize → encode → greedy decode → detokenize pipeline.

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
    ) -> list[str]:
        target = Language(target_language)
        source = None if source_language is None else Language(source_language)
        self.schema.validate(source, target, type(self).__name__)
        if not texts:
            return []

        # Leave room for the language token and </s> added by `_frame`.
        limit = self.max_length - 2
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        sequences = [self._frame(e.ids[:limit], source, target) for e in encodings]
        generation = replace(
            self._generation, forced_first_id=self._forced_first_id(target)
        )

        # Similar lengths in the same batch keep padding (and wasted compute) low.
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        outputs: list[list[int]] = [[] for _ in texts]
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
        return self.tokenizer.decode_batch(outputs, skip_special_tokens=True)

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
