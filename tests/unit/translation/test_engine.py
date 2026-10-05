from pathlib import Path

import numpy as np
import pytest

from noentenc.translation.models import _engine
from noentenc.translation.models._engine import (
    DecoderShape,
    GenerationConfig,
    Seq2SeqOnnxEngine,
)

SHAPE = DecoderShape(num_layers=2, num_heads=2, head_dim=4)
VOCAB = 10
START, EOS, PAD = 0, 1, 9


class FakeEncoder:
    """Tags each row's hidden states with its row index so the decoder can tell rows apart."""

    def run(
        self, output_names: None, inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        batch, length = inputs["input_ids"].shape
        hidden = np.zeros((batch, length, 8), dtype=np.float32)
        hidden[:, :, 0] = np.arange(batch)[:, None]
        return [hidden]


class FakeDecoder:
    """Emits `scripts[row][step]` as the argmax token for each row."""

    def __init__(self, scripts: list[list[int]]) -> None:
        self.scripts = scripts
        self.batch_sizes: list[int] = []

    def run(
        self, output_names: list[str], inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        rows = inputs["encoder_hidden_states"][:, 0, 0].astype(int)
        step = inputs["past_key_values.0.decoder.key"].shape[2]
        assert inputs["use_cache_branch"][0] == (step > 0)
        self.batch_sizes.append(len(rows))
        logits = np.zeros((len(rows), 1, VOCAB), dtype=np.float32)
        for i, row in enumerate(rows):
            script = self.scripts[row]
            logits[i, 0, script[step] if step < len(script) else EOS] = 1.0
        outputs = [logits]
        for name in output_names[1:]:
            length = (
                step + 1
                if ".decoder." in name
                else inputs["encoder_hidden_states"].shape[1]
            )
            outputs.append(
                np.zeros(
                    (len(rows), SHAPE.num_heads, length, SHAPE.head_dim), np.float32
                )
            )
        return outputs


def make_engine(
    monkeypatch: pytest.MonkeyPatch, scripts: list[list[int]]
) -> tuple[Seq2SeqOnnxEngine, FakeDecoder]:
    decoder = FakeDecoder(scripts)
    sessions = {"encoder": FakeEncoder(), "decoder": decoder}
    monkeypatch.setattr(
        _engine, "create_session", lambda path, num_threads, extra: sessions[path.name]
    )
    return Seq2SeqOnnxEngine(Path("encoder"), Path("decoder"), SHAPE), decoder


def generate(
    engine: Seq2SeqOnnxEngine,
    batch: int,
    max_new_tokens: int = 10,
    forced_first_id: int | None = None,
    banned_ids: tuple[int, ...] = (),
) -> list[list[int]]:
    config = GenerationConfig(
        START, EOS, PAD, max_new_tokens, forced_first_id, banned_ids
    )
    input_ids = np.full((batch, 3), 5, dtype=np.int64)
    return engine.generate(input_ids, np.ones_like(input_ids), config)


def test_stops_at_eos_and_strips_it(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = make_engine(monkeypatch, [[4, 5, EOS], [6, EOS]])
    assert generate(engine, 2) == [[4, 5], [6]]


def test_forced_first_token(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = make_engine(monkeypatch, [[4, 5, EOS]])
    assert generate(engine, 1, forced_first_id=7) == [[7, 5]]


def test_banned_ids_are_never_generated(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = make_engine(monkeypatch, [[PAD, EOS]])
    (output,) = generate(engine, 1, banned_ids=(PAD,))
    assert PAD not in output


def test_max_new_tokens_caps_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = make_engine(monkeypatch, [[4] * 20])
    assert generate(engine, 1, max_new_tokens=3) == [[4, 4, 4]]


def test_compaction_keeps_rows_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = [[2, EOS], [3, 3, 3, 3, EOS], [4, EOS], [5, 5, 5, EOS]]
    engine, decoder = make_engine(monkeypatch, scripts)
    assert generate(engine, 4) == [[2], [3, 3, 3, 3], [4], [5, 5, 5]]
    # Rows 0 and 2 finish at step 1, so later steps only decode the 2 live rows.
    assert decoder.batch_sizes[:3] == [4, 4, 2]
    assert decoder.batch_sizes[-1] == 1


def test_decoder_shape_from_config() -> None:
    config = {"decoder_layers": 6, "decoder_attention_heads": 8, "d_model": 512}
    assert DecoderShape.from_config(config) == DecoderShape(6, 8, 64)
