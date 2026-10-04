import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from noentenc.languages import Language, UnsupportedLanguageError
from noentenc.translation.models import _seq2seq
from noentenc.translation.models._engine import GenerationConfig
from noentenc.translation.models._hub import resolve_files
from noentenc.translation.models._seq2seq import Precision
from noentenc.translation.models.m2m100 import M2M100Model, _drop_invalid_merges
from noentenc.translation.models.nllb import NLLBModel
from noentenc.translation.models.opus_mt import (
    OPUS_MT_REVISIONS,
    OpusMTModel,
    _proto_field,
    _spm_precompiled_charsmap,
)
from noentenc.translation.models.small100 import SMaLL100Model

EN, ES, FR, ZH = Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.CHINESE
WORDS = ["hello", "world", "hola", "mundo"]


class FakeEngine:
    """Records what the model feeds the engine and echoes the (unpadded) input back."""

    calls: list[tuple[np.ndarray, np.ndarray, GenerationConfig]] = []

    def __init__(
        self, encoder: Path, decoder: Path, shape: object, num_threads: int | None
    ) -> None:
        pass

    def generate(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        config: GenerationConfig,
    ) -> list[list[int]]:
        FakeEngine.calls.append((input_ids, attention_mask, config))
        return [
            row[mask == 1].tolist()
            for row, mask in zip(input_ids, attention_mask, strict=True)
        ]


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeEngine.calls = []
    monkeypatch.setattr(_seq2seq, "Seq2SeqOnnxEngine", FakeEngine)


def write_model_dir(
    path: Path,
    specials: list[str],
    config: dict,
    onnx_files: tuple[str, str],
    extra: tuple[str, ...] = (),
) -> Path:
    vocab = {token: i for i, token in enumerate([*specials, *WORDS])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(specials)
    tokenizer.save(str(path / "tokenizer.json"))
    base_config = {
        "max_position_embeddings": 16,
        "decoder_layers": 1,
        "decoder_attention_heads": 1,
        "d_model": 4,
    }
    (path / "config.json").write_text(json.dumps({**base_config, **config}))
    for name in (*onnx_files, *extra):
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_bytes(b"")
    return path


def token_ids(model: _seq2seq.Seq2SeqModel, tokens: list[str]) -> list[int | None]:
    return [model.tokenizer.token_to_id(token) for token in tokens]


@pytest.fixture
def opus_dir(tmp_path: Path) -> Path:
    return write_model_dir(
        tmp_path,
        ["</s>", "<unk>", "<pad>", ">>cmn_Hans<<"],
        {
            "_name_or_path": "Helsinki-NLP/opus-mt-en-zh",
            "decoder_start_token_id": 2,
            "eos_token_id": 0,
            "pad_token_id": 2,
        },
        OpusMTModel.onnx_files[Precision.Q4],
        ("source.spm",),
    )


@pytest.fixture
def m2m_dir(tmp_path: Path) -> Path:
    specials = ["<s>", "<pad>", "</s>", "<unk>", "__en__", "__es__", "__fr__"]
    config = {"decoder_start_token_id": 2, "eos_token_id": 2, "pad_token_id": 1}
    write_model_dir(tmp_path, specials, config, M2M100Model.onnx_files[Precision.INT8])
    write_model_dir(
        tmp_path, specials, config, SMaLL100Model.onnx_files[Precision.INT8]
    )
    return tmp_path


@pytest.fixture
def nllb_dir(tmp_path: Path) -> Path:
    return write_model_dir(
        tmp_path,
        ["<s>", "<pad>", "</s>", "<unk>", "eng_Latn", "spa_Latn"],
        {"decoder_start_token_id": 2, "eos_token_id": 2, "pad_token_id": 1},
        NLLBModel.onnx_files[Precision.INT8],
    )


def test_opus_frames_with_target_prefix_and_bans_pad(opus_dir: Path) -> None:
    model = OpusMTModel(opus_dir)
    assert model.schema.source == frozenset({EN})
    assert model.schema.target == frozenset({ZH})
    # The fake engine echoes the input; special tokens are skipped when decoding.
    assert model.predict("hello world", ZH) == "hello world"
    input_ids, _, config = FakeEngine.calls[0]
    assert input_ids[0].tolist() == token_ids(
        model, [">>cmn_Hans<<", "hello", "world", "</s>"]
    )
    assert config.banned_ids == (2,)
    assert config.forced_first_id is None


def test_opus_rejects_languages_outside_its_pair(opus_dir: Path) -> None:
    model = OpusMTModel(opus_dir)
    with pytest.raises(UnsupportedLanguageError):
        model.predict("hello", FR)
    with pytest.raises(UnsupportedLanguageError):
        model.predict("hello", ZH, ES)


def test_opus_from_pair_rejects_unknown_pairs() -> None:
    with pytest.raises(UnsupportedLanguageError, match="No Opus-MT model"):
        OpusMTModel.from_pair(Language.CATALAN, Language.JAPANESE)


def test_m2m100_frames_source_and_forces_target(m2m_dir: Path) -> None:
    model = M2M100Model(m2m_dir)
    model.predict("hello", ES, EN)
    input_ids, _, config = FakeEngine.calls[0]
    assert input_ids[0].tolist() == token_ids(model, ["__en__", "hello", "</s>"])
    assert config.forced_first_id == model.tokenizer.token_to_id("__es__")
    with pytest.raises(UnsupportedLanguageError, match="requires a source"):
        model.predict("hello", ES)


def test_small100_puts_target_on_source_side(m2m_dir: Path) -> None:
    model = SMaLL100Model(m2m_dir)
    model.predict("hello", FR)
    input_ids, _, config = FakeEngine.calls[0]
    assert input_ids[0].tolist() == token_ids(model, ["__fr__", "hello", "</s>"])
    assert config.forced_first_id is None


def test_small100_only_has_int8_weights(m2m_dir: Path) -> None:
    with pytest.raises(ValueError, match="no fp32 weights"):
        SMaLL100Model(m2m_dir, precision="fp32")


def test_nllb_uses_flores_codes_and_warns(nllb_dir: Path) -> None:
    with pytest.warns(UserWarning, match="CC-BY-NC"):
        model = NLLBModel(nllb_dir)
    model.predict("hello", ES, EN)
    input_ids, _, config = FakeEngine.calls[0]
    assert input_ids[0].tolist() == token_ids(model, ["eng_Latn", "hello", "</s>"])
    assert config.forced_first_id == model.tokenizer.token_to_id("spa_Latn")


def test_batch_is_sorted_padded_chunked_and_restored(m2m_dir: Path) -> None:
    model = SMaLL100Model(m2m_dir)
    texts = ["hello world hola mundo", "hello", "hello world", "mundo"]
    assert model.predict_batch(texts, EN, batch_size=2) == texts
    (short, short_mask, _), (long, long_mask, config) = FakeEngine.calls
    assert short.shape == (2, 3)
    assert long.shape == (2, 6)
    assert long_mask[0].tolist() == [1, 1, 1, 1, 0, 0]
    assert config.max_new_tokens == min(model.max_length, 2 * 6 + 10)


def test_inputs_are_truncated_to_max_length(m2m_dir: Path) -> None:
    model = SMaLL100Model(m2m_dir)
    model.predict(" ".join(["hello"] * 40), EN)
    input_ids, _, _ = FakeEngine.calls[0]
    assert input_ids.shape[1] == model.max_length


def test_empty_batch(m2m_dir: Path) -> None:
    assert SMaLL100Model(m2m_dir).predict_batch([], EN) == []


def test_spm_charsmap_is_read_from_model_proto() -> None:
    charsmap = b"\x01\x02\x03"
    normalizer_spec = (
        b"\x0a\x08nmt_nfkc" + b"\x12\x03" + charsmap
    )  # name (1), charsmap (2)
    model_proto = (
        b"\x10\x05" + b"\x1a" + bytes([len(normalizer_spec)]) + normalizer_spec
    )
    assert _spm_precompiled_charsmap(model_proto) == charsmap
    with pytest.raises(ValueError, match="field 7"):
        _proto_field(model_proto, 7)


def test_resolve_files_reads_local_dir(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    assert resolve_files(tmp_path, ["config.json"]) == {
        "config.json": tmp_path / "config.json"
    }
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        resolve_files(tmp_path, ["config.json", "tokenizer.json"])


@pytest.mark.parametrize(
    ("make", "repo", "revision"),
    [
        (
            lambda: OpusMTModel.from_pair(ES, EN),
            "Xenova/opus-mt-es-en",
            OPUS_MT_REVISIONS[ES, EN],
        ),
        (SMaLL100Model, SMaLL100Model.default_model, SMaLL100Model.default_revision),
        (M2M100Model, M2M100Model.default_model, M2M100Model.default_revision),
        (
            lambda: M2M100Model("me/m2m100-finetune", revision="abc123"),
            "me/m2m100-finetune",
            "abc123",
        ),
    ],
)
def test_downloads_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
    make: Callable[[], object],
    repo: str,
    revision: str,
) -> None:
    calls: list[tuple[str | Path, str | None]] = []

    def fake_resolve(
        model: str | Path, *_args: object, revision: str | None = None
    ) -> dict[str, Path]:
        calls.append((model, revision))
        raise StopIteration

    monkeypatch.setattr(_seq2seq, "resolve_files", fake_resolve)
    with pytest.raises(StopIteration):
        make()
    assert calls == [(repo, revision)]


def test_opus_pairs_are_pinned_to_commits() -> None:
    assert all(re.fullmatch("[0-9a-f]{40}", r) for r in OPUS_MT_REVISIONS.values())


def test_m2m100_drops_merges_outside_the_vocab() -> None:
    spec = {
        "model": {
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": ["a b", "a c", ["b", "a"]],
        }
    }
    assert _drop_invalid_merges(spec)["model"]["merges"] == ["a b"]
