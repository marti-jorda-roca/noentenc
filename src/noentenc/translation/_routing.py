"""Which model each profile uses for a language pair, decided without loading anything."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    Profile.SPEED: (_opus_mt, _small100),
    Profile.BALANCE: (_opus_mt, _nllb_int8, _small100),
    Profile.QUALITY: (_opus_mt, _nllb_fp32, _small100),
}


def choose(profile: Profile, source: Language | None, target: Language) -> ModelChoice:
    """The model `profile` uses for `source` -> `target` (`source=None`: unknown)."""
    for candidate in CANDIDATES[profile]:
        choice = candidate(source, target)
        if choice is not None:
            return choice
    pair = f"into {target}" if source is None else f"{source}->{target}"
    raise UnsupportedLanguageError(
        f"No {profile} model translates {pair}; pass a model explicitly"
    )
