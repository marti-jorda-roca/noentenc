import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from tokenizers import Tokenizer

from noentenc.languages import Language, LanguageSchema
from noentenc.translation.models._seq2seq import Precision, Seq2SeqModel

L = Language

# Direct single-pair Opus-MT models with an ONNX export under `Xenova/opus-mt-{src}-{tgt}`.
OPUS_MT_PAIRS: frozenset[tuple[Language, Language]] = frozenset(
    {
        (L.AFRIKAANS, L.ENGLISH),
        (L.ARABIC, L.ENGLISH),
        (L.CZECH, L.ENGLISH),
        (L.DANISH, L.GERMAN),
        (L.DANISH, L.ENGLISH),
        (L.GERMAN, L.ENGLISH),
        (L.GERMAN, L.SPANISH),
        (L.GERMAN, L.FRENCH),
        (L.ENGLISH, L.AFRIKAANS),
        (L.ENGLISH, L.ARABIC),
        (L.ENGLISH, L.CZECH),
        (L.ENGLISH, L.DANISH),
        (L.ENGLISH, L.GERMAN),
        (L.ENGLISH, L.SPANISH),
        (L.ENGLISH, L.FINNISH),
        (L.ENGLISH, L.FRENCH),
        (L.ENGLISH, L.HINDI),
        (L.ENGLISH, L.HUNGARIAN),
        (L.ENGLISH, L.INDONESIAN),
        (L.ENGLISH, L.ITALIAN),
        (L.ENGLISH, L.DUTCH),
        (L.ENGLISH, L.ROMANIAN),
        (L.ENGLISH, L.RUSSIAN),
        (L.ENGLISH, L.SWEDISH),
        (L.ENGLISH, L.UKRAINIAN),
        (L.ENGLISH, L.VIETNAMESE),
        (L.ENGLISH, L.XHOSA),
        (L.ENGLISH, L.CHINESE),
        (L.SPANISH, L.GERMAN),
        (L.SPANISH, L.ENGLISH),
        (L.SPANISH, L.FRENCH),
        (L.SPANISH, L.ITALIAN),
        (L.SPANISH, L.RUSSIAN),
        (L.ESTONIAN, L.ENGLISH),
        (L.FINNISH, L.GERMAN),
        (L.FINNISH, L.ENGLISH),
        (L.FRENCH, L.GERMAN),
        (L.FRENCH, L.ENGLISH),
        (L.FRENCH, L.SPANISH),
        (L.FRENCH, L.ROMANIAN),
        (L.FRENCH, L.RUSSIAN),
        (L.HINDI, L.ENGLISH),
        (L.HUNGARIAN, L.ENGLISH),
        (L.INDONESIAN, L.ENGLISH),
        (L.ITALIAN, L.ENGLISH),
        (L.ITALIAN, L.SPANISH),
        (L.ITALIAN, L.FRENCH),
        (L.JAPANESE, L.ENGLISH),
        (L.KOREAN, L.ENGLISH),
        (L.DUTCH, L.ENGLISH),
        (L.DUTCH, L.FRENCH),
        (L.NORWEGIAN, L.GERMAN),
        (L.POLISH, L.ENGLISH),
        (L.ROMANIAN, L.FRENCH),
        (L.RUSSIAN, L.ENGLISH),
        (L.RUSSIAN, L.SPANISH),
        (L.RUSSIAN, L.FRENCH),
        (L.RUSSIAN, L.UKRAINIAN),
        (L.SWEDISH, L.ENGLISH),
        (L.THAI, L.ENGLISH),
        (L.TURKISH, L.ENGLISH),
        (L.UKRAINIAN, L.ENGLISH),
        (L.UKRAINIAN, L.RUSSIAN),
        (L.VIETNAMESE, L.ENGLISH),
        (L.XHOSA, L.ENGLISH),
        (L.CHINESE, L.ENGLISH),
    }
)

# Multi-target models that need a sentence-initial `>>lang<<` token.
_TARGET_PREFIX: dict[tuple[Language, Language], str] = {
    (L.ENGLISH, L.ARABIC): ">>ara<<",
    (L.ENGLISH, L.VIETNAMESE): ">>vie<<",
    (L.ENGLISH, L.CHINESE): ">>cmn_Hans<<",
}


class OpusMTModel(Seq2SeqModel):
    """Helsinki-NLP Opus-MT (MarianMT): one small model per language direction."""

    default_model: ClassVar[str] = "Xenova/opus-mt-en-es"
    default_precision: ClassVar[Precision] = Precision.Q4
    onnx_files: ClassVar[dict[Precision, tuple[str, str]]] = {
        Precision.FP32: ("onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"),
        Precision.INT8: (
            "onnx/encoder_model_quantized.onnx",
            "onnx/decoder_model_merged_quantized.onnx",
        ),
        Precision.Q4: (
            "onnx/encoder_model_q4.onnx",
            "onnx/decoder_model_merged_q4.onnx",
        ),
    }
    # Holds the SentencePiece normalization rules that Xenova's tokenizer.json lacks.
    extra_files: ClassVar[tuple[str, ...]] = ("source.spm",)

    def __init__(
        self,
        model: str | Path | None = None,
        only_local_files: bool = False,
        *,
        precision: Precision | str | None = None,
        num_threads: int | None = None,
    ) -> None:
        super().__init__(
            model, only_local_files, precision=precision, num_threads=num_threads
        )
        source, target = _parse_pair(self._repo_pair())
        self.schema = LanguageSchema(
            source=frozenset({source}), target=frozenset({target})
        )
        prefix = _TARGET_PREFIX.get((source, target))
        self._prefix_ids = [] if prefix is None else [self._token_id(prefix)]

    @classmethod
    def from_pair(
        cls,
        source_language: Language,
        target_language: Language,
        only_local_files: bool = False,
        *,
        precision: Precision | str | None = None,
        num_threads: int | None = None,
    ) -> OpusMTModel:
        source, target = Language(source_language), Language(target_language)
        if (source, target) not in OPUS_MT_PAIRS:
            available = sorted(str(t) for s, t in OPUS_MT_PAIRS if s == source)
            raise ValueError(
                f"No Opus-MT model for {source}->{target}; targets from {source}: {available}"
            )
        return cls(
            f"Xenova/opus-mt-{source}-{target}",
            only_local_files,
            precision=precision,
            num_threads=num_threads,
        )

    def _repo_pair(self) -> tuple[str, str]:
        """(source, target) codes from the upstream repo name, e.g. `Helsinki-NLP/opus-mt-en-es`."""
        name = str(self.config.get("_name_or_path") or self.model)
        source, target = name.rsplit("opus-mt-", 1)[-1].split("-", 1)
        return source, target

    def _load_tokenizer(self, files: dict[str, Path]) -> Tokenizer:
        spec = json.loads(files["tokenizer.json"].read_text())
        normalizer = spec.get("normalizer") or {}
        if (
            normalizer.get("type") == "Precompiled"
            and normalizer.get("precompiled_charsmap") is None
        ):
            charsmap = _spm_precompiled_charsmap(files["source.spm"].read_bytes())
            normalizer["precompiled_charsmap"] = base64.b64encode(charsmap).decode()
        return Tokenizer.from_str(json.dumps(spec))

    def _frame(
        self, ids: list[int], source: Language | None, target: Language
    ) -> list[int]:
        return [*self._prefix_ids, *ids, self.config["eos_token_id"]]

    def _banned_ids(self) -> tuple[int, ...]:
        # Marian reuses <pad> as the decoder start token; it must never be generated.
        return (self.config["pad_token_id"],)


def _parse_pair(codes: tuple[str, str]) -> tuple[Language, Language]:
    try:
        return Language(codes[0]), Language(codes[1])
    except ValueError:
        raise ValueError(
            f"Opus-MT model {codes[0]}-{codes[1]} is not a single-pair model with standard codes"
        )


def _spm_precompiled_charsmap(model_proto: bytes) -> bytes:
    """Extract `normalizer_spec.precompiled_charsmap` from a SentencePiece model file."""
    normalizer_spec = _proto_field(model_proto, 3)
    return _proto_field(normalizer_spec, 2)


def _proto_field(message: bytes, field_number: int) -> bytes:
    for number, value in _proto_fields(message):
        if number == field_number and isinstance(value, bytes):
            return value
    raise ValueError(f"protobuf field {field_number} not found")


def _proto_fields(message: bytes) -> Iterator[tuple[int, int | bytes]]:
    pos = 0
    while pos < len(message):
        key, pos = _varint(message, pos)
        number, wire_type = key >> 3, key & 0x7
        if wire_type == 0:
            value, pos = _varint(message, pos)
            yield number, value
        elif wire_type == 2:  # noqa: PLR2004 - protobuf length-delimited wire type
            length, pos = _varint(message, pos)
            yield number, message[pos : pos + length]
            pos += length
        else:
            # Fixed-width fields: 64-bit (1) or 32-bit (5).
            pos += 8 if wire_type == 1 else 4


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:  # noqa: PLR2004 - continuation bit
            return result, pos
