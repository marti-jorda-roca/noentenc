"""What a profile would load, what it costs, and the caller's limits on licence and size.

Nothing here loads or downloads a model: sizes and licences come from `_catalog.py`, and
cache state from looking at the cache directory.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from noentenc.languages import UnsupportedLanguageError

MB = 1 << 20

# Resident memory measured after loading a model and translating a sentence
# (docs/benchmarks.md#memory), per model and precision.
_MEASURED_MEMORY: dict[tuple[str, str], int] = {
    ("OpusMTModel", "q4"): 1084 * MB,
    ("OpusMTModel", "int8"): 630 * MB,
    ("OpusMTModel", "fp32"): 1312 * MB,
    ("SMaLL100Model", "int8"): 1151 * MB,
    ("M2M100Model", "int8"): 2375 * MB,
    ("NLLBModel", "int8"): 4057 * MB,
    ("NLLBModel", "fp32"): 3810 * MB,
}
# Opus-MT was measured on en→es; every pair has the same architecture.
_MEASURED_ON = {"OpusMTModel": "Opus-MT en→es"}
# Without a measurement, memory is estimated as this many times the download, the median
# of the measured translation models.
_ESTIMATE_FACTOR = 4

# Resident memory measured after loading each detection preset and labelling 2,000
# FLORES-200 sentences in 200 languages (docs/benchmarks.md#detection-memory). The large
# fastText models are memory-mapped; this counts the pages those sentences touched.
_MEASURED_DETECTION_MEMORY: dict[str, int] = {
    "lid176": 22 * MB,
    "langid": 112 * MB,
    "bert-openlid": 249 * MB,
    "openlid-v3": 1142 * MB,
    "glotlid": 1270 * MB,
}


@dataclass(frozen=True)
class ModelPlan:
    """A model a profile would load, and what it costs. Nothing is downloaded to know it."""

    task: Literal["detection", "translation"]
    # The model as `Translation.model` names it, e.g. "OpusMTModel(Xenova/opus-mt-en-es)".
    name: str
    precision: str | None
    # SPDX identifier of the weights' licence.
    license: str
    # Bytes of the model's files, cached or not.
    download_bytes: int
    # Resident memory once loaded, and whether it was measured or estimated.
    memory_bytes: int
    memory_basis: str
    # Whether every file is already in the cache.
    cached: bool
    files: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class Plan:
    """The models a profile would load for a set of tasks."""

    models: tuple[ModelPlan, ...]

    @property
    def download_bytes(self) -> int:
        """Bytes of every model's files."""
        return sum(model.download_bytes for model in self.models)

    @property
    def missing_bytes(self) -> int:
        """Bytes still to download: the models that aren't cached yet."""
        return sum(m.download_bytes for m in self.models if not m.cached)

    @property
    def memory_bytes(self) -> int:
        """Resident memory with every model loaded at once."""
        return sum(model.memory_bytes for model in self.models)

    @property
    def licenses(self) -> frozenset[str]:
        return frozenset(model.license for model in self.models)


class ModelConstraintError(UnsupportedLanguageError):
    """No model the profile would use for a task is allowed by the caller's constraints."""


@dataclass(frozen=True)
class Constraints:
    """Limits on which models a profile may pick. `None` means no limit."""

    # SPDX identifiers, compared case-insensitively.
    allowed_licenses: frozenset[str] | None = None
    max_download_bytes: int | None = None

    @classmethod
    def of(
        cls,
        allowed_licenses: Iterable[str] | None = None,
        max_download_bytes: int | None = None,
    ) -> Constraints:
        if isinstance(allowed_licenses, str):
            raise TypeError("allowed_licenses must be a list of SPDX identifiers")
        licenses = None
        if allowed_licenses is not None:
            licenses = frozenset(licence.lower() for licence in allowed_licenses)
        if max_download_bytes is not None and (
            isinstance(max_download_bytes, bool)
            or not isinstance(max_download_bytes, int)
            or max_download_bytes < 0
        ):
            raise ValueError(
                f"max_download_bytes must be a non-negative int, got {max_download_bytes!r}"
            )
        return cls(licenses, max_download_bytes)

    @property
    def active(self) -> bool:
        return self.allowed_licenses is not None or self.max_download_bytes is not None

    def rejection(self, plan: ModelPlan) -> str | None:
        """Why `plan` isn't allowed, or None."""
        if (
            self.allowed_licenses is not None
            and plan.license.lower() not in self.allowed_licenses
        ):
            return f"licence {plan.license} is not allowed"
        if (
            self.max_download_bytes is not None
            and plan.download_bytes > self.max_download_bytes
        ):
            return (
                f"{plan.download_bytes / MB:.0f} MB is over max_download_bytes "
                f"({self.max_download_bytes / MB:.0f} MB)"
            )
        return None


def translation_memory(
    model_class: str, precision: str, download_bytes: int
) -> tuple[int, str]:
    measured = _MEASURED_MEMORY.get((model_class, precision))
    if measured is not None:
        on = _MEASURED_ON.get(model_class)
        return measured, f"measured on {on}" if on else "measured"
    return (
        _ESTIMATE_FACTOR * download_bytes,
        f"estimated as {_ESTIMATE_FACTOR}x the download",
    )


def detection_memory(preset: str, download_bytes: int) -> tuple[int, str]:
    measured = _MEASURED_DETECTION_MEMORY.get(preset)
    if measured is not None:
        return measured, "measured after labelling 2,000 sentences"
    # Memory-mapped weights take at most their file size once every page is read.
    return download_bytes, "estimated as the download size"


def hf_cached(
    repo: str, revision: str, filenames: Iterable[str], cache_dir: Path | None
) -> bool:
    """Whether every file is in the Hugging Face cache at `revision` (a commit)."""
    root = cache_dir or _default_hf_cache()
    snapshot = root / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    return all((snapshot / name).is_file() for name in filenames)


def _default_hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "huggingface" / "hub"
