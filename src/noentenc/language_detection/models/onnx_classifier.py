"""Transformer sequence classifiers for language identification, run with onnxruntime."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from noentenc.language_detection._download import RemoteFile, fetch
from noentenc.language_detection._optional import require
from noentenc.language_detection.labels import LabelMapper
from noentenc.language_detection.models.base import BaseModel

ONNX_EXTRA = "onnx"


@dataclass(frozen=True)
class OnnxPreset:
    repo: str
    revision: str
    onnx_file: str
    onnx_sha256: str
    license: str
    description: str

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
    ),
}


class OnnxClassifierModel(BaseModel):
    """A Hugging Face ``*ForSequenceClassification`` model exported to ONNX.

    ``model`` is a preset name (see ``PRESETS``) or a local directory holding ``config.json``,
    ``tokenizer.json`` and either ``model.onnx`` or ``onnx/model_quantized.onnx``.

    Speed: texts are truncated to ``max_length`` tokens, batches are formed from texts of similar
    length and padded only to their longest member, and onnxruntime runs with full graph
    optimisation. ``num_threads=None`` lets onnxruntime use every physical core.
    """

    sort_batches_by_length = True

    def __init__(
        self,
        model: str | Path = "bert-openlid",
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
        max_length: int = 128,
        num_threads: int | None = None,
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        ort = require("onnxruntime", ONNX_EXTRA)
        tokenizers = require("tokenizers", ONNX_EXTRA)
        onnx_path, tokenizer_path, config_path = self._resolve(model, only_local_files)

        config = json.loads(config_path.read_text(encoding="utf-8"))
        id2label = config["id2label"]
        self.native_labels = [id2label[str(i)] for i in range(len(id2label))]
        self._mapper = LabelMapper(
            self.native_labels,
            normalize=normalize_labels,
            collapse_macrolanguages=collapse_macrolanguages,
        )
        self._pad_id = int(config.get("pad_token_id") or 0)

        self._tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(max_length=max_length)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = num_threads or 0
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    @staticmethod
    def _resolve(model: str | Path, only_local_files: bool) -> tuple[Path, Path, Path]:
        if isinstance(model, str) and model in PRESETS:
            preset = PRESETS[model]
            return (
                fetch(
                    preset.remote(preset.onnx_file, preset.onnx_sha256),
                    only_local_files=only_local_files,
                ),
                fetch(
                    preset.remote("tokenizer.json"), only_local_files=only_local_files
                ),
                fetch(preset.remote("config.json"), only_local_files=only_local_files),
            )
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

    @property
    def labels(self) -> list[str]:
        return list(self._mapper.labels)

    def logits(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        width = max(len(enc.ids) for enc in encodings)
        input_ids = np.full((len(texts), width), self._pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(texts), width), dtype=np.int64)
        for row, enc in enumerate(encodings):
            input_ids[row, : len(enc.ids)] = enc.ids
            attention_mask[row, : len(enc.ids)] = 1
        feeds: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        return self._session.run(
            None, {k: v for k, v in feeds.items() if k in self._input_names}
        )[0]

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        logits = self.logits(texts)
        if self._mapper.identity:
            return self._mapper.top_label(logits.argmax(axis=1))
        return self._mapper.top_label(
            self._mapper.reduce(_softmax(logits)).argmax(axis=1)
        )

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        return self._mapper.to_dicts(
            self._mapper.reduce(_softmax(self.logits(texts))), top_k
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)
