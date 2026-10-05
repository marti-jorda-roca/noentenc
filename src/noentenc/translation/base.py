from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Literal, overload

from tqdm import tqdm

from noentenc._batching import check_batch_size, check_texts, is_blank
from noentenc._dataframe import column_values, with_column
from noentenc.languages import Language
from noentenc.profiles import Profile
from noentenc.translation import _routing
from noentenc.translation.models.base import (
    BaseModel,
    Translation,
    TranslationStatus,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import polars as pl

# What to do when translating a text raises: raise it, or record it in the result.
ErrorPolicy = Literal["raise", "record"]

# Models a profile-based `Translator` keeps loaded at once. Two cover a pair model plus the
# SMaLL-100 fallback; see docs/benchmarks.md#memory for the RAM each model takes.
DEFAULT_MAX_LOADED_MODELS = 2


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

    A profile keeps at most `max_loaded_models` models loaded (`None`: no limit). When a
    pair needs another one, the least recently used is dropped first, and loaded again if a
    later call needs it. A higher limit saves reloads when calls alternate between many
    pairs, at 0.6 to 1.2 GB of RAM per Opus-MT or SMaLL-100 model (docs/benchmarks.md).
    `unload()` drops them all.

    A profile downloads models into `cache_dir` (see `noentenc._cache` for the default),
    or with `only_local_files` only reads them from there, failing at once when a model
    is missing. `noentenc.prepare` downloads them ahead of time. For a `model` you pass,
    give these options to the model instead.
    """

    def __init__(
        self,
        model: BaseModel | Profile | str | None = None,
        *,
        max_loaded_models: int | None = DEFAULT_MAX_LOADED_MODELS,
        only_local_files: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        if isinstance(model, BaseModel):
            if only_local_files or cache_dir is not None:
                raise ValueError(
                    "only_local_files and cache_dir apply to the models a profile "
                    "loads; pass them to your model instead"
                )
            self.model: BaseModel | None = model
            self.profile = Profile.SPEED  # unused: `model` translates every pair
        else:
            self.model = None
            self.profile = Profile(model or Profile.SPEED)
        self.only_local_files = only_local_files
        self.cache_dir = cache_dir
        self.max_loaded_models = _check_max_loaded_models(max_loaded_models)
        # Loaded profile models, least recently used first.
        self._default_models: OrderedDict[Hashable, BaseModel] = OrderedDict()
        self._models_lock = threading.Lock()

    @property
    def loaded_models(self) -> list[BaseModel]:
        """The models this translator loaded and still holds, least recently used first.

        Empty when it was given a `model`: that one belongs to the caller.
        """
        with self._models_lock:
            return list(self._default_models.values())

    def unload(self) -> None:
        """Drop every model this translator loaded, so their memory can be freed.

        Releases the ONNX sessions and tokenizers once nothing else references them; a
        call already translating keeps its model until it returns. The translator stays
        usable and loads models again when needed. A `model` passed to the constructor
        belongs to the caller and is kept.
        """
        with self._models_lock:
            self._default_models.clear()

    @overload
    def translate(
        self,
        text: str,
        target_language: Language,
        source_language: Language | None = None,
        *,
        truncate: bool = False,
        detailed: Literal[False] = False,
    ) -> str: ...

    @overload
    def translate(
        self,
        text: str,
        target_language: Language,
        source_language: Language | None = None,
        *,
        truncate: bool = False,
        detailed: Literal[True],
    ) -> Translation: ...

    def translate(
        self,
        text: str,
        target_language: Language,
        source_language: Language | None = None,
        *,
        truncate: bool = False,
        detailed: bool = False,
    ) -> str | Translation:
        """Translate `text`, sentence by sentence, keeping its whitespace and line breaks.

        A sentence longer than the model reads raises `InputTooLongError`; with
        `truncate` its end is dropped instead. With `detailed` the result is a
        `Translation`, which says whether input was dropped or output cut short.

        Blank text, and text whose `source_language` is `target_language`, come back
        unchanged without loading a model.
        """
        (translation,) = self._translate(
            [text], target_language, source_language, 32, truncate, "raise"
        )
        return translation if detailed else translation.text

    @overload
    def translate_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
        detailed: Literal[False] = False,
        errors: ErrorPolicy = "raise",
    ) -> list[str]: ...

    @overload
    def translate_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
        detailed: Literal[True],
        errors: ErrorPolicy = "raise",
    ) -> list[Translation]: ...

    def translate_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
        detailed: bool = False,
        errors: ErrorPolicy = "raise",
    ) -> list[str] | list[Translation]:
        """Translate each of `texts`, like `translate`.

        An empty list returns at once. With `errors="record"` a text whose translation
        raises comes back as given, and its detailed result has status `FAILED` and
        the error; the other texts are still translated. Arguments that apply to the
        whole call (batch size, languages, an unsupported pair) always raise.
        """
        translations = self._translate(
            texts, target_language, source_language, batch_size, truncate, errors
        )
        return translations if detailed else [t.text for t in translations]

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
        *,
        truncate: bool = False,
        errors: ErrorPolicy = "raise",
        error_column: str | None = None,
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
        *,
        truncate: bool = False,
        errors: ErrorPolicy = "raise",
        error_column: str | None = None,
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
        *,
        truncate: bool = False,
        errors: ErrorPolicy = "raise",
        error_column: str | None = None,
    ) -> pl.DataFrame | pd.DataFrame:
        """Return `dataset` with `result_column` holding the translation of `target_column`.

        Works with polars and pandas frames; null texts stay null. Texts are translated
        like `translate_batch`, including `truncate` and `errors`. With
        `errors="record"` a row whose translation raises gets a null translation, and
        its error in `error_column` if given (null for the other rows).
        With `show_progress` a tqdm bar tracks the rows translated so far.
        """
        _check_arguments(target_language, source_language, batch_size, errors)
        values = column_values(dataset, target_column)
        rows = [i for i, value in enumerate(values) if isinstance(value, str)]
        texts = [values[i] for i in rows]
        translations: list[Translation] = []
        with tqdm(
            total=len(texts), desc="Translating", disable=not show_progress
        ) as progress:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                translations += self._translate(
                    chunk,
                    target_language,
                    source_language,
                    batch_size,
                    truncate,
                    errors,
                )
                progress.update(len(chunk))
        results: list[str | None] = [None] * len(values)
        failures: list[str | None] = [None] * len(values)
        for i, translation in zip(rows, translations, strict=True):
            if translation.status is TranslationStatus.FAILED:
                failures[i] = translation.error
            else:
                results[i] = translation.text
        dataset = with_column(dataset, result_column, results, strings=True)
        if error_column is not None:
            dataset = with_column(dataset, error_column, failures, strings=True)
        return dataset

    def _translate(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None,
        batch_size: int,
        truncate: bool,
        errors: ErrorPolicy,
    ) -> list[Translation]:
        """Translate `texts`, answering what needs no model before loading one."""
        check_texts(texts)
        target, source = _check_arguments(
            target_language, source_language, batch_size, errors
        )
        # Blank texts and same-language requests stay as given; translate the rest.
        results = [Translation(t, status=TranslationStatus.UNCHANGED) for t in texts]
        pending = [
            i for i, text in enumerate(texts) if not is_blank(text) and source != target
        ]
        if not pending:
            return results
        model = self._model_for(source, target)

        def run(batch: list[str]) -> list[Translation]:
            return model.predict_batch_detailed(
                batch, target, source, batch_size, truncate=truncate
            )

        batch = [texts[i] for i in pending]
        translated = run(batch) if errors == "raise" else _record_failures(batch, run)
        for i, translation in zip(pending, translated, strict=True):
            results[i] = translation
        return results

    def _model_for(self, source: Language | None, target: Language) -> BaseModel:
        if self.model is not None:
            # Checked here so an unsupported pair fails the call even when
            # `errors="record"` keeps per-text failures.
            self.model.schema.validate(source, target, type(self.model).__name__)
            return self.model
        choice = _routing.choose(self.profile, source, target)
        return self._cached(
            choice,
            lambda: choice.build(
                only_local_files=self.only_local_files, cache_dir=self.cache_dir
            ),
        )

    def _cached(self, key: Hashable, load: Callable[[], BaseModel]) -> BaseModel:
        """The loaded model for `key`, loading it after evicting down to the limit.

        Eviction only drops the cache's reference: a call that is translating with an
        evicted model keeps it alive until it returns.
        """
        with self._models_lock:
            model = self._default_models.get(key)
            if model is not None:
                self._default_models.move_to_end(key)
                return model
            limit = self.max_loaded_models
            # Evict before loading, so the old models can be freed first.
            while limit is not None and len(self._default_models) >= limit:
                self._default_models.popitem(last=False)
        model = load()
        with self._models_lock:
            self._default_models[key] = model
            self._default_models.move_to_end(key)
            while limit is not None and len(self._default_models) > limit:
                self._default_models.popitem(last=False)
        return model


def _check_max_loaded_models(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"max_loaded_models must be an int or None, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"max_loaded_models must be >= 1, got {value}")
    return value


def _check_arguments(
    target: Language,
    source: Language | None,
    batch_size: int,
    errors: ErrorPolicy,
) -> tuple[Language, Language | None]:
    """Validate what applies to a whole call; return the languages as `Language`s."""
    check_batch_size(batch_size)
    if errors not in ("raise", "record"):
        raise ValueError(f"errors must be 'raise' or 'record', got {errors!r}")
    return Language(target), None if source is None else Language(source)


def _record_failures(
    texts: list[str], run: Callable[[list[str]], list[Translation]]
) -> list[Translation]:
    """`run(texts)`, with each text whose translation raises returned as a failure.

    A batch that raises is split in half until the failing texts are alone, so the
    others are still translated in batches.
    """
    try:
        return run(texts)
    except Exception as error:
        if len(texts) == 1:
            return [
                Translation(
                    texts[0],
                    status=TranslationStatus.FAILED,
                    error=f"{type(error).__name__}: {error}",
                )
            ]
    middle = len(texts) // 2
    return _record_failures(texts[:middle], run) + _record_failures(texts[middle:], run)
