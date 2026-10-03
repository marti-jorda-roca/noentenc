from typing import TYPE_CHECKING, cast, overload

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
    ) -> pd.DataFrame: ...

    def translate_dataset(
        self,
        dataset: pl.DataFrame | pd.DataFrame,
        target_column: str,
        result_column: str,
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> pl.DataFrame | pd.DataFrame:
        """Return `dataset` with `result_column` holding the translation of `target_column`.

        Works with polars and pandas frames; null texts stay null.
        """
        module = type(dataset).__module__.split(".", 1)[0]
        if module not in {"polars", "pandas"}:
            raise TypeError(
                f"expected a polars or pandas DataFrame, got {type(dataset).__name__}"
            )
        values = dataset[target_column].to_list()
        rows = [i for i, value in enumerate(values) if isinstance(value, str)]
        translations = self.translate_batch(
            [values[i] for i in rows], target_language, source_language, batch_size
        )
        results: list[str | None] = [None] * len(values)
        for i, translation in zip(rows, translations, strict=True):
            results[i] = translation

        if module == "polars":
            # The caller passed a polars frame, so polars is installed.
            import polars

            frame = cast("pl.DataFrame", dataset)
            return frame.with_columns(
                polars.Series(result_column, results, polars.String)
            )
        return cast("pd.DataFrame", dataset).assign(**{result_column: results})

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
            raise UnsupportedLanguageError(
                f"No default model translates {source}->{target}; pass a model explicitly"
            )
        if None not in self._default_models:
            self._default_models[None] = SMaLL100Model()
        return self._default_models[None]
