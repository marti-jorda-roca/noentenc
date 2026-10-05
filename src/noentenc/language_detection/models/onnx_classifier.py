"""Transformer sequence classifiers for language identification, run with onnxruntime.

onnxruntime and tokenizers come with the ``onnx`` extra and are imported when a model is built.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from noentenc._onnx import create_session, pad
from noentenc._optional import ONNX_EXTRA, require
from noentenc.language_detection._download import RemoteFile, fetch
from noentenc.language_detection.models.base import BaseModel

# Texts are cut before tokenizing at this many characters per token of `max_length`. Tokens
# are a few characters long, so the cut almost always keeps more than `max_length` tokens.
MAX_CHARS_PER_TOKEN = 8
# Tokens kept per text for local model directories; presets set their own.
DEFAULT_MAX_LENGTH = 128


@dataclass(frozen=True)
class OnnxPreset:
    repo: str
    revision: str
    onnx_file: str
    onnx_sha256: str
    license: str
    description: str
    # Tokens kept per text. Compute grows with it; past this, longer inputs stopped changing
    # accuracy on WiLI-2018 paragraphs (see docs/benchmarks.md).
    max_length: int = DEFAULT_MAX_LENGTH

    def remote(self, filename: str, sha256: str | None = None) -> RemoteFile:
        return RemoteFile.huggingface(self.repo, filename, self.revision, sha256)


PRESETS: dict[str, OnnxPreset] = {
    "xlm-roberta-lid": OnnxPreset(
        repo="onnx-community/xlm-roberta-base-language-detection-ONNX",
        revision="919c87aa2749131ae1ab709931a16bf1cc9774ea",
        onnx_file="onnx/model_quantized.onnx",
        onnx_sha256="e3a2f1b44ea6a76683e4655127531b05c6b568fe643bd39bddf7f7c62ab182c9",
        license="MIT",
        description="papluca XLM-RoBERTa-base, 20 languages, int8 (279 MB). Accurate but slow.",
    ),
    "bert-openlid": OnnxPreset(
        repo="onnx-community/language_detection-ONNX",
        revision="859bc93be10a6f8ad4a070911a4caaa092cdcd2e",
        onnx_file="onnx/model_quantized.onnx",
        onnx_sha256="ee8599c721d24f678ef10cee403f4deda3a40e7e237e9983b3df11bf3f34ec2e",
        license="MIT",
        description="alexneakameni 4-layer BERT trained on OpenLID, 201 labels, int8 (25 MB).",
        max_length=96,
    ),
}


class OnnxClassifierModel(BaseModel):
    """A Hugging Face ``*ForSequenceClassification`` model exported to ONNX.

    ``model`` is a preset name (see ``PRESETS``) or a local directory holding ``config.json``,
    ``tokenizer.json`` and either ``model.onnx`` or ``onnx/model_quantized.onnx``.

    Speed: texts are truncated to ``max_length`` tokens (``None``: the preset's, or 128 for a
    local directory), batches are formed from texts of similar length and padded only to their
    longest member, and onnxruntime runs with full graph optimisation. ``num_threads=None`` lets
    onnxruntime use every physical core. Long texts are cut at a space before tokenizing, far
    enough out that the kept tokens don't change.
    """

    sort_batches_by_length = True

    def __init__(
        self,
        model: str | Path = "bert-openlid",
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
        max_length: int | None = None,
        num_threads: int | None = None,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        # Checked before downloading weights that couldn't be run.
        tokenizers = require("tokenizers", ONNX_EXTRA)
        require("onnxruntime", ONNX_EXTRA)
        onnx_path, tokenizer_path, config_path = self._resolve(
            model, only_local_files, cache_dir
        )
        if max_length is None:
            preset = PRESETS.get(model) if isinstance(model, str) else None
            max_length = preset.max_length if preset else DEFAULT_MAX_LENGTH

        config = json.loads(config_path.read_text(encoding="utf-8"))
        id2label = config["id2label"]
        self.native_labels = [id2label[str(i)] for i in range(len(id2label))]
        self._mapper = self._label_mapper(self.native_labels)
        self._pad_id = int(config.get("pad_token_id") or 0)

        self._tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(max_length=max_length)
        self._max_chars = max_length * MAX_CHARS_PER_TOKEN
        self._session = create_session(onnx_path, num_threads)
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    @staticmethod
    def remote_files(preset: str) -> list[RemoteFile]:
        """The files a preset downloads: the ONNX graph, tokenizer and config."""
        if preset not in PRESETS:
            raise ValueError(
                f"unknown ONNX preset {preset!r}; presets: {', '.join(PRESETS)}"
            )
        spec = PRESETS[preset]
        return [
            spec.remote(spec.onnx_file, spec.onnx_sha256),
            spec.remote("tokenizer.json"),
            spec.remote("config.json"),
        ]

    @staticmethod
    def _resolve(
        model: str | Path, only_local_files: bool, cache_dir: str | Path | None
    ) -> tuple[Path, Path, Path]:
        if isinstance(model, str) and model in PRESETS:
            onnx, tokenizer, config = (
                fetch(remote, only_local_files=only_local_files, cache_dir=cache_dir)
                for remote in OnnxClassifierModel.remote_files(model)
            )
            return onnx, tokenizer, config
        root = Path(model)
        if not root.is_dir():
            raise FileNotFoundError(
                f"{model!r} is neither an ONNX preset ({', '.join(PRESETS)}) nor a directory"
            )
        onnx_path = next(
            (
                p
                for p in (root / "model.onnx", root / "onnx" / "model_quantized.onnx")
                if p.is_file()
            ),
            None,
        )
        if onnx_path is None:
            raise FileNotFoundError(
                f"no model.onnx or onnx/model_quantized.onnx in {root}"
            )
        return onnx_path, root / "tokenizer.json", root / "config.json"

    def token_ids(self, texts: list[str]) -> list[list[int]]:
        """Token ids of each text, truncated to ``max_length``.

        A text longer than ``max_length * MAX_CHARS_PER_TOKEN`` characters is first cut at a
        space: tokens never span whitespace, so the cut text's tokens are a prefix of the
        whole text's. When the cut text still overflows ``max_length``, its truncated tokens
        are exactly the whole text's; otherwise the whole text is tokenized instead.
        """
        cut = [_cut_at_space(text, self._max_chars) for text in texts]
        encodings = self._tokenizer.encode_batch(cut)
        short = [
            i
            for i, (text, part, enc) in enumerate(
                zip(texts, cut, encodings, strict=True)
            )
            if part is not text and not enc.overflowing
        ]
        if short:
            redone = self._tokenizer.encode_batch([texts[i] for i in short])
            for i, enc in zip(short, redone, strict=True):
                encodings[i] = enc
        return [enc.ids for enc in encodings]

    def logits(self, texts: list[str]) -> np.ndarray:
        input_ids, attention_mask = pad(self.token_ids(texts), self._pad_id)
        feeds: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        outputs = self._session.run(
            None, {k: v for k, v in feeds.items() if k in self._input_names}
        )
        return cast("np.ndarray", outputs[0])

    def _scores(self, texts: list[str]) -> np.ndarray:
        return self._mapper.reduce(_softmax(self.logits(texts)))

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        return self._mapper.top_label(self._scores(texts).argmax(axis=1))

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        return self._mapper.to_dicts(self._scores(texts), top_k)


def _cut_at_space(text: str, limit: int) -> str:
    """``text`` up to its last space before ``limit`` characters (``text`` itself if short)."""
    if len(text) <= limit:
        return text
    space = text.rfind(" ", limit // 2, limit)
    return text if space < 0 else text[:space]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)
