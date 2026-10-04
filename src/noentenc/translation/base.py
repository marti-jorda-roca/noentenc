from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, overload

from tqdm import tqdm

from noentenc._dataframe import column_values, with_column
from noentenc.languages import Language, UnsupportedLanguageError
from noentenc.profiles import Profile
from noentenc.translation.models._seq2seq import Precision, Seq2SeqModel
from noentenc.translation.models.base import BaseModel
from noentenc.translation.models.nllb import NLLBModel
from noentenc.translation.models.opus_mt import OPUS_MT_PAIRS, OpusMTModel
from noentenc.translation.models.small100 import SMaLL100Model

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# What a candidate returns for a language pair: the cache key and loader of the model
# it would use, or None when that model can't translate the pair.
_Pick = tuple[Hashable, Callable[[], BaseModel]] | None


def _opus_mt(source: Language | None, target: Language) -> _Pick:
    if source is None or (source, target) not in OPUS_MT_PAIRS:
        return None
    return (source, target), lambda: OpusMTModel.from_pair(source, target)


def _small100(source: Language | None, target: Language) -> _Pick:
    return _multilingual(SMaLL100Model, source, target)


def _nllb_int8(source: Language | None, target: Language) -> _Pick:
    # Explicit, so changing the model's default precision doesn't change the profile.
    return _multilingual(NLLBModel, source, target, Precision.INT8)


def _nllb_fp32(source: Language | None, target: Language) -> _Pick:
    return _multilingual(NLLBModel, source, target, Precision.FP32)


def _multilingual(
    model: type[Seq2SeqModel],
    source: Language | None,
    target: Language,
    precision: Precision | None = None,
) -> _Pick:
    if not model.schema.supports(source, target):
        return None
    return (model, precision), lambda: model(precision=precision)


# Models tried in order for each profile; the first that translates the pair is used.
# Where Opus-MT has a model it scores as well as NLLB-200 on average and is ~8x faster
# (FLORES-200 chrF++, docs/benchmarks.md), so every profile tries it first.
_CANDIDATES: dict[Profile, tuple[Callable[[Language | None, Language], _Pick], ...]] = {
    Profile.SPEED: (_opus_mt, _small100),
    Profile.BALANCE: (_opus_mt, _nllb_int8, _small100),
    Profile.QUALITY: (_opus_mt, _nllb_fp32, _small100),
}


class Translator:
    """Translate text, batches or dataset columns.

    `model` is a model, or a `Profile` that picks one per language pair (`None` is
    `"speed"`). Models are loaded lazily on first use and then reused.

    Every profile uses a direct Opus-MT model when one exists for the pair. For the
    other pairs:

    - `speed`: SMaLL-100 (int8, 595 MB).
    - `balance`: NLLB-200 600M (int8, 860 MB), about 7x slower than SMaLL-100.
    - `quality`: NLLB-200 600M (fp32, 3.5 GB).

    Without a source language, or for a language NLLB-200 lacks, they fall back to
    SMaLL-100. NLLB-200 is licensed CC-BY-NC-4.0 (non-commercial) and warns when loaded.
    """

    def __init__(self, model: BaseModel | Profile | str | None = None) -> None:
        if isinstance(model, BaseModel):
            self.model: BaseModel | None = model
            self.profile = Profile.SPEED  # unused: `model` translates every pair
        else:
            self.model = None
            self.profile = Profile(model or Profile.SPEED)
        self._default_models: dict[Hashable, BaseModel] = {}

    def translate(
        self,
        text: str,
        target_language: Language,
        source_language: Language | None = None,
    ) -> str:
        return self.translate_batch([text], target_language, source_language)[0]

    def translate_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        model = self._model_for(source_language, target_language)
        return model.predict_batch(texts, target_language, source_language, batch_size)

    @overload
    def translate_dataset(
        self,
        dataset: pl.DataFrame,
        target_column: str,
        result_column: str,
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> pl.DataFrame: ...

    @overload
    def translate_dataset(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        result_column: str,
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> pd.DataFrame: ...

    def translate_dataset(
        self,
        dataset: pl.DataFrame | pd.DataFrame,
        target_column: str,
        result_column: str,
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> pl.DataFrame | pd.DataFrame:
        """Return `dataset` with `result_column` holding the translation of `target_column`.

        Works with polars and pandas frames; null texts stay null.
        With `show_progress` a tqdm bar tracks the rows translated so far.
        """
        values = column_values(dataset, target_column)
        rows = [i for i, value in enumerate(values) if isinstance(value, str)]
        model = self._model_for(source_language, target_language)
        texts = [values[i] for i in rows]
        translations: list[str] = []
        with tqdm(
            total=len(texts), desc="Translating", disable=not show_progress
        ) as progress:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                translations.extend(
                    model.predict_batch(
                        chunk, target_language, source_language, batch_size
                    )
                )
                progress.update(len(chunk))
        results: list[str | None] = [None] * len(values)
        for i, translation in zip(rows, translations, strict=True):
            results[i] = translation
        return with_column(dataset, result_column, results, strings=True)

    def _model_for(self, source: Language | None, target: Language) -> BaseModel:
        if self.model is not None:
            return self.model
        source = None if source is None else Language(source)
        target = Language(target)
        for candidate in _CANDIDATES[self.profile]:
            picked = candidate(source, target)
            if picked is None:
                continue
            key, load = picked
            if key not in self._default_models:
                self._default_models[key] = load()
            return self._default_models[key]
        pair = f"into {target}" if source is None else f"{source}->{target}"
        raise UnsupportedLanguageError(
            f"No {self.profile} model translates {pair}; pass a model explicitly"
        )
