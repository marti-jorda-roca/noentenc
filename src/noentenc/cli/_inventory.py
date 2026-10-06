"""The models noentenc can use, and what the cache holds, for `noentenc list` and `info`.

Nothing here downloads or loads a model: sizes and licences come from the catalog, and
cache state from looking at the cache directory.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from noentenc._cache import CACHE_ENV, cache_root, translation_cache
from noentenc._catalog import DETECTION_FILE_SIZES
from noentenc.language_detection._download import RemoteFile, is_cached
from noentenc.language_detection.base import PROFILE_PRESETS
from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.language_detection.models.fasttext.model import (
    PRESETS as FASTTEXT_PRESETS,
)
from noentenc.language_detection.models.langid_model import LangidModel
from noentenc.language_detection.models.onnx_classifier import (
    PRESETS as ONNX_PRESETS,
)
from noentenc.language_detection.models.onnx_classifier import OnnxClassifierModel
from noentenc.languages import Language
from noentenc.profiles import Profile
from noentenc.translation._routing import ModelChoice, auto_choices
from noentenc.translation.models.m2m100 import M2M100Model

Task = Literal["detection", "translation"]

# The modules each extra installs, to tell whether it is installed.
EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "translation": ("huggingface_hub", "onnxruntime", "tokenizers"),
    "onnx": ("onnxruntime", "tokenizers"),
    "pandas": ("pandas",),
    "polars": ("polars",),
    "lingua": ("lingua",),
    "cld3": ("gcld3",),
    "heliport": ("heliport",),
    "cli": ("click",),
}

# Detection backends whose weights ship inside their package: (name, licence, extra).
_PACKAGED_DETECTION = (
    ("LinguaModel", "Apache-2.0", "lingua"),
    ("Cld3Model", "Apache-2.0", "cld3"),
    ("HeliportModel", "GPL-3.0", "heliport"),
)
_LANGID_LICENSE = "BSD-2-Clause"


@dataclass(frozen=True)
class CatalogModel:
    """A model noentenc can use: what it downloads, who uses it, and whether it's ready."""

    task: Task
    # As `ModelPlan.name` and `Translation.model` name it, e.g. "FastTextModel(lid176)".
    name: str
    precision: str | None
    license: str
    # None when the weights ship inside the backend's package.
    download_bytes: int | None
    # The profiles that load it.
    profiles: tuple[Profile, ...]
    # Whether every file is in the cache; None when there is nothing to download.
    cached: bool | None
    # The extra it needs (None: the core install), and whether that extra is installed.
    extra: str | None
    installed: bool


@dataclass(frozen=True)
class CachedModel:
    """Weights found in the cache, with the bytes they take on disk."""

    task: Task
    # A detection model's name, or a translation model's Hugging Face repo id. Files
    # that no detection model downloads are named by their path in the cache.
    name: str
    path: Path
    size_bytes: int


def extra_installed(extra: str) -> bool:
    return all(importlib.util.find_spec(module) for module in EXTRA_MODULES[extra])


def catalog(cache_dir: str | Path | None = None) -> list[CatalogModel]:
    """Every model a profile or a preset can load, detection first."""
    return [*_detection_catalog(cache_dir), *_translation_catalog(cache_dir)]


def _detection_remotes() -> Iterator[tuple[str, list[RemoteFile], str, str | None]]:
    """(name, files, licence, extra) of each detection model noentenc downloads."""
    for preset, spec in FASTTEXT_PRESETS.items():
        yield (
            f"FastTextModel({preset})",
            FastTextModel.remote_files(preset),
            spec.license,
            None,
        )
    for preset, spec in ONNX_PRESETS.items():
        yield (
            f"OnnxClassifierModel({preset})",
            OnnxClassifierModel.remote_files(preset),
            spec.license,
            "onnx",
        )
    yield "LangidModel", LangidModel.remote_files(), _LANGID_LICENSE, None


def _detection_catalog(cache_dir: str | Path | None) -> Iterator[CatalogModel]:
    profiles: dict[str, list[Profile]] = {}
    for profile, preset in PROFILE_PRESETS.items():
        profiles.setdefault(f"FastTextModel({preset})", []).append(profile)
    for name, remotes, licence, extra in _detection_remotes():
        yield CatalogModel(
            task="detection",
            name=name,
            precision=None,
            license=licence,
            download_bytes=sum(DETECTION_FILE_SIZES[r.cache_path] for r in remotes),
            profiles=tuple(profiles.get(name, ())),
            cached=all(is_cached(remote, cache_dir) for remote in remotes),
            extra=extra,
            installed=extra is None or extra_installed(extra),
        )
    for name, licence, extra in _PACKAGED_DETECTION:
        yield CatalogModel(
            task="detection",
            name=name,
            precision=None,
            license=licence,
            download_bytes=None,
            profiles=(),
            cached=None,
            extra=extra,
            installed=extra_installed(extra),
        )


def _translation_catalog(cache_dir: str | Path | None) -> list[CatalogModel]:
    """The models any profile routes a pair to, and M2M-100, which none does."""
    profiles: dict[ModelChoice, list[Profile]] = {}
    for profile in Profile:
        for target in Language:
            for choice in auto_choices(profile, target):
                used_by = profiles.setdefault(choice, [])
                if profile not in used_by:
                    used_by.append(profile)
    m2m100 = ModelChoice(
        M2M100Model, M2M100Model.default_model, M2M100Model.default_revision
    )
    profiles.setdefault(m2m100, [])
    installed = extra_installed("translation")
    models = []
    for choice, used_by in profiles.items():
        plan = choice.plan(cache_dir)
        models.append(
            CatalogModel(
                task="translation",
                name=plan.name,
                precision=plan.precision,
                license=plan.license,
                download_bytes=plan.download_bytes,
                profiles=tuple(used_by),
                cached=plan.cached,
                extra="translation",
                installed=installed,
            )
        )
    # The multilingual models first, then the Opus-MT pairs.
    return sorted(
        models,
        key=lambda m: (m.name.startswith("OpusMTModel"), m.name, m.precision),
    )


def cache_source(cache_dir: str | Path | None = None) -> str:
    """Where the cache root comes from: the option, the environment, or the default."""
    if cache_dir is not None:
        return "--cache-dir"
    return CACHE_ENV if os.environ.get(CACHE_ENV) else "default"


def cached_models(cache_dir: str | Path | None = None) -> list[CachedModel]:
    """The detection and translation weights in the cache. Reads file sizes only."""
    return [
        *_cached_detection(cache_root(cache_dir), translation_cache(cache_dir)),
        *_cached_translation(translation_cache(cache_dir)),
    ]


def _cached_detection(root: Path, hub: Path) -> list[CachedModel]:
    owners = {
        remote.cache_path: name
        for name, remotes, _, _ in _detection_remotes()
        for remote in remotes
    }
    files: dict[str, list[Path]] = {}
    for directory, subdirectories, filenames in os.walk(root):
        if Path(directory) == root and hub.name in subdirectories:
            subdirectories.remove(hub.name)
        for filename in filenames:
            path = Path(directory) / filename
            # Downloads in progress are written to a ".part" file first.
            if path.suffix == ".part":
                continue
            relative = path.relative_to(root).as_posix()
            files.setdefault(owners.get(relative, relative), []).append(path)
    return [
        CachedModel(
            task="detection",
            name=name,
            path=Path(os.path.commonpath(paths)) if len(paths) > 1 else paths[0],
            size_bytes=sum(path.stat().st_size for path in paths),
        )
        for name, paths in sorted(files.items())
    ]


def _cached_translation(hub: Path) -> list[CachedModel]:
    models = []
    for repo in sorted(hub.glob("models--*")):
        # Snapshot files link to blobs, which recent huggingface-hub versions may keep
        # outside the repo's folder; on Windows without symlinks they are the files.
        # Count each file a snapshot resolves to once.
        files = {path.resolve() for path in repo.glob("snapshots/**/*")}
        size = sum(path.stat().st_size for path in files if path.is_file())
        if size:
            name = repo.name.removeprefix("models--").replace("--", "/")
            models.append(CachedModel("translation", name, repo, size))
    return models
