from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort

# Once at least this fraction of the batch has finished, drop finished rows so the
# remaining decoder steps only compute live sequences.
_COMPACT_FINISHED_RATIO = 0.5


@dataclass(frozen=True)
class DecoderShape:
    num_layers: int
    num_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: dict) -> DecoderShape:
        num_heads = config["decoder_attention_heads"]
        return cls(
            num_layers=config["decoder_layers"],
            num_heads=num_heads,
            head_dim=config["d_model"] // num_heads,
        )


@dataclass(frozen=True)
class GenerationConfig:
    decoder_start_id: int
    eos_id: int
    pad_id: int
    max_new_tokens: int
    forced_first_id: int | None = None
    banned_ids: tuple[int, ...] = ()


def create_session(path: Path, num_threads: int | None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    if num_threads is not None:
        options.intra_op_num_threads = num_threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class _DecodeState:
    """Live rows of a batch being decoded greedily."""

    def __init__(
        self,
        encoder_states: np.ndarray,
        attention_mask: np.ndarray,
        past: dict[str, np.ndarray],
        config: GenerationConfig,
    ) -> None:
        batch_size = encoder_states.shape[0]
        self.encoder_states = encoder_states
        self.attention_mask = attention_mask
        self.past = past
        self.rows = np.arange(batch_size)
        self.next_ids = np.full(batch_size, config.decoder_start_id, dtype=np.int64)
        self.finished = np.zeros(batch_size, dtype=bool)
        self.output = np.full(
            (batch_size, config.max_new_tokens), config.pad_id, dtype=np.int64
        )

    def compact(self) -> None:
        keep = ~self.finished
        self.encoder_states = self.encoder_states[keep]
        self.attention_mask = self.attention_mask[keep]
        self.past = {name: value[keep] for name, value in self.past.items()}
        self.rows = self.rows[keep]
        self.next_ids = self.next_ids[keep]
        self.finished = self.finished[keep]


class Seq2SeqOnnxEngine:
    """Batched greedy decoding over an optimum-style encoder + merged decoder."""

    def __init__(
        self,
        encoder_path: Path,
        decoder_path: Path,
        shape: DecoderShape,
        num_threads: int | None = None,
    ) -> None:
        self._encoder = create_session(encoder_path, num_threads)
        self._decoder = create_session(decoder_path, num_threads)
        self._shape = shape
        layers = range(shape.num_layers)
        self._decoder_cache = [
            f"present.{i}.decoder.{kind}" for i in layers for kind in ("key", "value")
        ]
        self._encoder_cache = [
            f"present.{i}.encoder.{kind}" for i in layers for kind in ("key", "value")
        ]

    def generate(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        config: GenerationConfig,
    ) -> list[list[int]]:
        (encoder_states,) = _run(
            self._encoder,
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        state = _DecodeState(
            encoder_states, attention_mask, self._empty_past(len(input_ids)), config
        )
        for step in range(config.max_new_tokens):
            logits = self._step(state, first=step == 0)
            self._select_next(state, logits, step, config)
            if state.finished.all():
                break
            if state.finished.mean() >= _COMPACT_FINISHED_RATIO:
                state.compact()
        return [_trim(row, config.eos_id) for row in state.output]

    def _empty_past(self, batch_size: int) -> dict[str, np.ndarray]:
        empty = np.zeros(
            (batch_size, self._shape.num_heads, 0, self._shape.head_dim),
            dtype=np.float32,
        )
        return {
            _past_name(name): empty
            for name in self._decoder_cache + self._encoder_cache
        }

    def _step(self, state: _DecodeState, first: bool) -> np.ndarray:
        inputs = {
            "input_ids": state.next_ids[:, None],
            "encoder_attention_mask": state.attention_mask,
            "encoder_hidden_states": state.encoder_states,
            "use_cache_branch": np.array([not first]),
            **state.past,
        }
        # Cross-attention KV only changes on the first step; afterwards it is reused.
        cache_names = self._decoder_cache + (self._encoder_cache if first else [])
        logits, *cache = _run(self._decoder, ["logits", *cache_names], inputs)
        for name, value in zip(cache_names, cache, strict=True):
            state.past[_past_name(name)] = value
        return logits[:, -1, :]

    @staticmethod
    def _select_next(
        state: _DecodeState, logits: np.ndarray, step: int, config: GenerationConfig
    ) -> None:
        if step == 0 and config.forced_first_id is not None:
            next_ids = np.full(len(logits), config.forced_first_id, dtype=np.int64)
        else:
            if config.banned_ids:
                logits[:, list(config.banned_ids)] = -np.inf
            next_ids = logits.argmax(axis=-1)
        next_ids[state.finished] = config.pad_id
        state.output[state.rows, step] = next_ids
        state.finished |= next_ids == config.eos_id
        state.next_ids = next_ids


def _run(
    session: ort.InferenceSession,
    output_names: list[str] | None,
    inputs: dict[str, np.ndarray],
) -> list[np.ndarray]:
    # Every output of these graphs is a dense tensor.
    return cast("list[np.ndarray]", session.run(output_names, inputs))


def _past_name(present_name: str) -> str:
    return present_name.replace("present", "past_key_values", 1)


def _trim(row: np.ndarray, eos_id: int) -> list[int]:
    ids = row.tolist()
    return ids[: ids.index(eos_id)] if eos_id in ids else ids
