from typing import TYPE_CHECKING, overload

from tqdm import tqdm

from noentenc._dataframe import column_values, with_column
from noentenc.languages import Language, UnsupportedLanguageError
from noentenc.translation.models.base import BaseModel
from noentenc.translation.models.opus_mt import OPUS_MT_PAIRS, OpusMTModel
from noentenc.translation.models.small100 import SMaLL100Model

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class Translator:
    """Translate text, batches or dataset columns.

    With `model=None`, each language pair uses the lightest and fastest model that
    supports it: a direct Opus-MT model when one exists, otherwise SMaLL-100.
    Models are loaded lazily on first use and then reused.
    """

    def __init__(self, model: BaseModel | None = None) -> None:
        self.model = model
        self._default_models: dict[tuple[Language, Language] | None, BaseModel] = {}

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
        if source is not None and (source, target) in OPUS_MT_PAIRS:
            key = (source, target)
            if key not in self._default_models:
                self._default_models[key] = OpusMTModel.from_pair(source, target)
            return self._default_models[key]
        if not SMaLL100Model.schema.supports(source, target):
            pair = f"into {target}" if source is None else f"{source}->{target}"
            raise UnsupportedLanguageError(
                f"No default model translates {pair}; pass a model explicitly"
            )
        if None not in self._default_models:
            self._default_models[None] = SMaLL100Model()
        return self._default_models[None]
