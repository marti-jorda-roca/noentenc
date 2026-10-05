from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from noentenc.languages import Language, LanguageSchema, UnsupportedLanguageError
from noentenc.translation.models._seq2seq import (
    Precision,
    Seq2SeqModel,
    tokenizer_class,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tokenizers import Tokenizer

L = Language

# Direct single-pair Opus-MT models under `Xenova/opus-mt-{src}-{tgt}`, each pinned to a commit.
OPUS_MT_REVISIONS: dict[tuple[Language, Language], str] = {
    (L.AFRIKAANS, L.ENGLISH): "0eba4e40af86c6ad316a58fc9a7ed1134b05fd90",
    (L.ARABIC, L.ENGLISH): "09c7556866400dcc53562ba8d7035119e7d9a2c1",
    (L.CZECH, L.ENGLISH): "c9f7336d1cf49ba5b7a412cba3b86999fa781716",
    (L.DANISH, L.GERMAN): "3784087953108e4ee9957c8f9eff11c0bf0cabda",
    (L.DANISH, L.ENGLISH): "6496d5e47a016552357be8f06294096609256ac4",
    (L.GERMAN, L.ENGLISH): "399dfd68706739fffd503f876093e455ae268a06",
    (L.GERMAN, L.SPANISH): "d339419ee3bda301bdcd00249d4436b74664ee7e",
    (L.GERMAN, L.FRENCH): "e783da03d247c4d16778e938530e9b5d7bd1523d",
    (L.ENGLISH, L.AFRIKAANS): "3d48312617d092d77cb5aef9f9e9636ad639dd8d",
    (L.ENGLISH, L.ARABIC): "034a684356c19021a187cda4d7f823298b913921",
    (L.ENGLISH, L.CZECH): "ed163ca109a93297e2f055eb3397bd33230141ad",
    (L.ENGLISH, L.DANISH): "33862dc071a5cc43e821e60742553c501c587e77",
    (L.ENGLISH, L.GERMAN): "1ca130c44c4c5441ef16d48aae521a424ab644f7",
    (L.ENGLISH, L.SPANISH): "4b002a4c7edd54a7ced58877258b87f7efd3f892",
    (L.ENGLISH, L.FINNISH): "4e5799273dbd7cce94c1e99cc9d3acb9f191b8eb",
    (L.ENGLISH, L.FRENCH): "28726206f80896b90035bd99cccd5cc1e151f916",
    (L.ENGLISH, L.HINDI): "7aad72006a1588a9c3485ed6203aed99eebbc872",
    (L.ENGLISH, L.HUNGARIAN): "1b7df958c1bd8f094eff1ce76d536468923ed7e3",
    (L.ENGLISH, L.INDONESIAN): "b3b41f654c2fb3286d007dee1da7b0f47fd27d82",
    (L.ENGLISH, L.ITALIAN): "075406e3c8c2c30634d4a1bd8f00c21d9e162011",
    (L.ENGLISH, L.DUTCH): "c3a0541231a7329b4492809fc2bd7ab7d2ceb0bc",
    (L.ENGLISH, L.ROMANIAN): "3d816891778276930df2a45cbfb70689b3fe8981",
    (L.ENGLISH, L.RUSSIAN): "e050376e960175e8ddc5cff85025dc8436cddd68",
    (L.ENGLISH, L.SWEDISH): "33d522d806c041cdebecd549c558d22a2b9b4ea4",
    (L.ENGLISH, L.UKRAINIAN): "e6ca587292984b343fd829b6430ed55858496525",
    (L.ENGLISH, L.VIETNAMESE): "30bcd461c0d7338ad106b36314d3eedc2d4e998c",
    (L.ENGLISH, L.XHOSA): "1ab9c4f4400ea4805f220ff5d551d3d0169b7a96",
    (L.ENGLISH, L.CHINESE): "046f55aec303cdee3e0318604406d4df20f1e8ea",
    (L.SPANISH, L.GERMAN): "7f493e4b81b684478396f37de4c1ebc28ecd5855",
    (L.SPANISH, L.ENGLISH): "eadfd7c658a9d8929ac3b8e996b68a68e2c7d480",
    (L.SPANISH, L.FRENCH): "3bef5f612825d503f6fb85e0527a1fe30f2578c2",
    (L.SPANISH, L.ITALIAN): "596c67cda7cb919f0d67221f3a31bd7df79f19d0",
    (L.SPANISH, L.RUSSIAN): "268ccd60bde3cabde17068d482ca6930512f1a85",
    (L.ESTONIAN, L.ENGLISH): "58104952b730220eb7a9cdbbe19533a064f40dc0",
    (L.FINNISH, L.GERMAN): "6d93339597e19574f543259315d2400939067c1e",
    (L.FINNISH, L.ENGLISH): "a623e4812687be788e3ab720c8a950ab64704793",
    (L.FRENCH, L.GERMAN): "7eea1996b4480d073cf560fcba6d4480aaf9a2bb",
    (L.FRENCH, L.ENGLISH): "6b166a182780e118c997879d0ad5be4b53671644",
    (L.FRENCH, L.SPANISH): "962338be51033b548c13dbb6f4da3ddb8c8aba0e",
    (L.FRENCH, L.ROMANIAN): "d2b0db550987291ebb35cdeb3a5394d176ecdcc6",
    (L.FRENCH, L.RUSSIAN): "c2ec732b045ae9f0e1ae1ea9267840145a8c03fe",
    (L.HINDI, L.ENGLISH): "99853100ed9d932987a6816e8cf7b1b6742a6116",
    (L.HUNGARIAN, L.ENGLISH): "c9e8a7617fb4cb4fca3785c5bb22cfe457cde00b",
    (L.INDONESIAN, L.ENGLISH): "c38ef36c843b71177da903442835c3c870975fd6",
    (L.ITALIAN, L.ENGLISH): "fd0b89b9c052adc1f2f64152f555aa17353728be",
    (L.ITALIAN, L.SPANISH): "0eba27fe6da2208231ea5e7047b0f62241ffc2bb",
    (L.ITALIAN, L.FRENCH): "7843e3b7137d19face19cb32e27256888d26fe84",
    (L.JAPANESE, L.ENGLISH): "1a906cfaaf7c8f4193f67f5885c082aa6dbd9d16",
    (L.KOREAN, L.ENGLISH): "d9fa1ac6008242100fecb4fff9b0a5917a1be81f",
    (L.DUTCH, L.ENGLISH): "82c42d95d1508037cbe7172d85fd4e940a2f3584",
    (L.DUTCH, L.FRENCH): "6c8ebb68522ffd80cf3474b2dc44f4b5f741e9cd",
    (L.NORWEGIAN, L.GERMAN): "5ed60d1d245fe19d3ee80a172de6b38724642eb3",
    (L.POLISH, L.ENGLISH): "aeaf0e003b045248c28a559a5ce6027b54ebdaba",
    (L.ROMANIAN, L.FRENCH): "48d206ed25c6f5b869a49a79207906c401334727",
    (L.RUSSIAN, L.ENGLISH): "afe8c6c738ec81b6d033fd8f44f9678a639a7c67",
    (L.RUSSIAN, L.SPANISH): "9cb5fb9fae4d5c0c731dcc3ba74c4e0a87f15eb0",
    (L.RUSSIAN, L.FRENCH): "c7275f468246036d24bfe14ce8465e94e1037af9",
    (L.RUSSIAN, L.UKRAINIAN): "6d8a744ed6eafd9003bcc2b052c4575de0485e1e",
    (L.SWEDISH, L.ENGLISH): "63a40256869b6a2b29a84e47aa7f03524fa5f3a2",
    (L.THAI, L.ENGLISH): "7f0406b972f742985cfbe3be8f61dead9f1bc0e3",
    (L.TURKISH, L.ENGLISH): "78247f41f354af036d24e6f0f8a3d02c6806f96e",
    (L.UKRAINIAN, L.ENGLISH): "b2ad6afa424aaeff8913ff350860b7a7b7c0a607",
    (L.UKRAINIAN, L.RUSSIAN): "cfb76ccd207734d7847679664fa4538cda5ed597",
    (L.VIETNAMESE, L.ENGLISH): "541dfd72ec07e8f1563795e0224408ad8d729e80",
    (L.XHOSA, L.ENGLISH): "f2950c1503ae17ba532ea858914ea13838bc2ab1",
    (L.CHINESE, L.ENGLISH): "39d480d52a9ea3065a1f117adfe4dbc55de10e6f",
}
OPUS_MT_PAIRS: frozenset[tuple[Language, Language]] = frozenset(OPUS_MT_REVISIONS)

# Multi-target models that need a sentence-initial `>>lang<<` token.
_TARGET_PREFIX: dict[tuple[Language, Language], str] = {
    (L.ENGLISH, L.ARABIC): ">>ara<<",
    (L.ENGLISH, L.VIETNAMESE): ">>vie<<",
    (L.ENGLISH, L.CHINESE): ">>cmn_Hans<<",
}


class OpusMTModel(Seq2SeqModel):
    """Helsinki-NLP Opus-MT (MarianMT): one small model per language direction."""

    default_model: ClassVar[str] = "Xenova/opus-mt-en-es"
    default_revision: ClassVar[str] = OPUS_MT_REVISIONS[L.ENGLISH, L.SPANISH]
    default_precision: ClassVar[Precision] = Precision.Q4
    # Holds the SentencePiece normalization rules that Xenova's tokenizer.json lacks.
    extra_files: ClassVar[tuple[str, ...]] = ("source.spm",)

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
        super().__init__(
            model,
            only_local_files,
            revision=revision,
            precision=precision,
            num_threads=num_threads,
            cache_dir=cache_dir,
        )
        source, target = _language_pair(
            str(self.config.get("_name_or_path") or self.model)
        )
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
        cache_dir: str | Path | None = None,
    ) -> OpusMTModel:
        repo, revision = pair_repo(source_language, target_language)
        return cls(
            repo,
            only_local_files,
            revision=revision,
            precision=precision,
            num_threads=num_threads,
            cache_dir=cache_dir,
        )

    def _load_tokenizer(self, files: dict[str, Path]) -> Tokenizer:
        spec = json.loads(files["tokenizer.json"].read_text())
        normalizer = spec.get("normalizer") or {}
        if (
            normalizer.get("type") == "Precompiled"
            and normalizer.get("precompiled_charsmap") is None
        ):
            charsmap = _spm_precompiled_charsmap(files["source.spm"].read_bytes())
            normalizer["precompiled_charsmap"] = base64.b64encode(charsmap).decode()
        return tokenizer_class().from_str(json.dumps(spec))

    def _frame(
        self, ids: list[int], source: Language | None, target: Language
    ) -> list[int]:
        return [*self._prefix_ids, *ids, self.config["eos_token_id"]]

    def _banned_ids(self) -> tuple[int, ...]:
        # Marian reuses <pad> as the decoder start token; it must never be generated.
        return (self.config["pad_token_id"],)


def pair_repo(source: Language | str, target: Language | str) -> tuple[str, str]:
    """The Hugging Face repo and pinned revision of the Opus-MT model for a pair."""
    source, target = Language(source), Language(target)
    if (source, target) not in OPUS_MT_PAIRS:
        available = sorted(str(t) for s, t in OPUS_MT_PAIRS if s == source)
        raise UnsupportedLanguageError(
            f"No Opus-MT model for {source}->{target}; targets from {source}: {available}"
        )
    return f"Xenova/opus-mt-{source}-{target}", OPUS_MT_REVISIONS[source, target]


def _language_pair(name: str) -> tuple[Language, Language]:
    """(source, target) from an Opus-MT repo name, e.g. `Helsinki-NLP/opus-mt-en-es`."""
    codes = name.rsplit("opus-mt-", 1)[-1]
    try:
        source, target = codes.split("-", 1)
        return Language(source), Language(target)
    except ValueError:
        raise ValueError(
            f"Opus-MT model {codes} is not a single-pair model with standard codes"
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
