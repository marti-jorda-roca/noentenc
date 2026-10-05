"""Which model each profile uses for a language pair, decided without loading anything."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from noentenc._cache import translation_cache
from noentenc._catalog import OPUS_MT_LICENSES, TRANSLATION_FILE_SIZES
from noentenc._plan import (
    Constraints,
    ModelConstraintError,
    ModelPlan,
    hf_cached,
    translation_memory,
)
from noentenc.languages import Language, UnsupportedLanguageError
from noentenc.profiles import Profile
from noentenc.translation.models._seq2seq import Precision, Seq2SeqModel
from noentenc.translation.models.nllb import NLLBModel
from noentenc.translation.models.opus_mt import OPUS_MT_PAIRS, OpusMTModel, pair_repo
from noentenc.translation.models.small100 import SMaLL100Model

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ModelChoice:
    """A model a profile picks for a pair: what to download and how to load it."""

    model_class: type[Seq2SeqModel]
    repo: str
    revision: str
    # None is the class's default precision.
    precision: Precision | None = None

    @property
    def name(self) -> str:
        """The model as `Translation.model` names it."""
        return f"{self.model_class.__name__}({self.repo})"

    def plan(
        self, cache_dir: str | Path | None = None, *, check_cache: bool = True
    ) -> ModelPlan:
        """What loading this model costs, from the catalog; downloads nothing.

        Without `check_cache`, `cached` is False instead of looking at the cache.
        """
        precision = self.model_class._check_precision(self.precision)
        files = self.model_class.filenames(precision)
        sizes = TRANSLATION_FILE_SIZES[f"{self.repo}@{self.revision}"]
        download = sum(sizes[name] for name in files)
        memory, basis = translation_memory(
            self.model_class.__name__, str(precision), download
        )
        licence = OPUS_MT_LICENSES.get(self.repo) or self.model_class.weights_license
        return ModelPlan(
            task="translation",
            name=self.name,
            precision=str(precision),
            license=licence or "unknown",
            download_bytes=download,
            memory_bytes=memory,
            memory_basis=basis,
            cached=check_cache
            and hf_cached(
                self.repo, self.revision, files, translation_cache(cache_dir)
            ),
            files=tuple(files),
        )

    def build(
        self, *, only_local_files: bool = False, cache_dir: str | Path | None = None
    ) -> Seq2SeqModel:
        return self.model_class(
            self.repo,
            only_local_files,
            revision=self.revision,
            precision=self.precision,
            cache_dir=cache_dir,
        )

    def download(
        self, *, cache_dir: str | Path | None = None, force: bool = False
    ) -> dict[str, Path]:
        return self.model_class.download(
            self.repo,
            revision=self.revision,
            precision=self.precision,
            cache_dir=cache_dir,
            force=force,
        )


# A candidate returns the model it would use for a pair, or None when it can't translate it.
_Candidate = Callable[[Language | None, Language], ModelChoice | None]


def _opus_mt(source: Language | None, target: Language) -> ModelChoice | None:
    if source is None or (source, target) not in OPUS_MT_PAIRS:
        return None
    return ModelChoice(OpusMTModel, *pair_repo(source, target))


def _opus_mt_int8(source: Language | None, target: Language) -> ModelChoice | None:
    # Only reached when constraints reject the default q4 export: int8 scores the same
    # on FLORES-200 and downloads 2.7x less, but is slower one sentence at a time.
    choice = _opus_mt(source, target)
    return (
        None
        if choice is None
        else ModelChoice(OpusMTModel, choice.repo, choice.revision, Precision.INT8)
    )


def _small100(source: Language | None, target: Language) -> ModelChoice | None:
    return _multilingual(SMaLL100Model, source, target)


def _nllb_int8(source: Language | None, target: Language) -> ModelChoice | None:
    # Explicit, so changing the model's default precision doesn't change the profile.
    return _multilingual(NLLBModel, source, target, Precision.INT8)


def _nllb_fp32(source: Language | None, target: Language) -> ModelChoice | None:
    return _multilingual(NLLBModel, source, target, Precision.FP32)


def _multilingual(
    model: type[Seq2SeqModel],
    source: Language | None,
    target: Language,
    precision: Precision | None = None,
) -> ModelChoice | None:
    if not model.schema.supports(source, target):
        return None
    return ModelChoice(model, model.default_model, model.default_revision, precision)


# Models tried in order for each profile; the first that translates the pair is used.
# Where Opus-MT has a model it scores as well as NLLB-200 on average and is ~8x faster
# (FLORES-200 chrF++, docs/benchmarks.md), so every profile tries it first.
CANDIDATES: dict[Profile, tuple[_Candidate, ...]] = {
    Profile.SPEED: (_opus_mt, _opus_mt_int8, _small100),
    Profile.BALANCE: (_opus_mt, _opus_mt_int8, _nllb_int8, _small100),
    Profile.QUALITY: (_opus_mt, _opus_mt_int8, _nllb_fp32, _small100),
}


def choose(
    profile: Profile,
    source: Language | None,
    target: Language,
    constraints: Constraints | None = None,
) -> ModelChoice:
    """The model `profile` uses for `source` -> `target` (`source=None`: unknown).

    A candidate that `constraints` rejects is skipped for the next one; when every
    candidate is rejected, `ModelConstraintError` says why.
    """
    rejected: list[str] = []
    for candidate in CANDIDATES[profile]:
        choice = candidate(source, target)
        if choice is None:
            continue
        reason = None
        if constraints is not None and constraints.active:
            reason = constraints.rejection(choice.plan(check_cache=False))
        if reason is None:
            return choice
        rejected.append(f"{choice.name}: {reason}")
    pair = f"into {target}" if source is None else f"{source}->{target}"
    if rejected:
        raise ModelConstraintError(
            f"No {profile} model for {pair} meets the constraints ({'; '.join(rejected)})"
        )
    raise UnsupportedLanguageError(
        f"No {profile} model translates {pair}; pass a model explicitly"
    )
