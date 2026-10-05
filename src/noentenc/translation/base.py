from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, overload

from tqdm import tqdm

from noentenc._batching import check_batch_size, check_texts, is_blank
from noentenc._dataframe import column_values, with_column
from noentenc._plan import Constraints, Plan
from noentenc.languages import (
    ANY_LANGUAGE,
    Language,
    LanguageSchema,
    UnsupportedLanguageError,
    to_language,
)
from noentenc.profiles import Profile
from noentenc.translation import _routing
from noentenc.translation.models.base import (
    BaseModel,
    Translation,
    TranslationStatus,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import pandas as pd
    import polars as pl

    from noentenc.language_detection import LanguageDetector

# What to do when translating a text raises: raise it, or record it in the result.
ErrorPolicy = Literal["raise", "record"]

# With `source_language="auto"`, what to do with a text whose language the detector can't
# tell, or that no model translates: keep it as given (with a status), translate it
# without a source language, or fail the call.
UnknownSourcePolicy = Literal["keep", "fallback", "raise"]

# Pass as `source_language` to detect each text's language.
AUTO: Final = "auto"

# Models a profile-based `Translator` keeps loaded at once. Two cover a pair model plus the
# SMaLL-100 fallback; see docs/benchmarks.md#memory for the RAM each model takes.
DEFAULT_MAX_LOADED_MODELS = 2

# The detector `source_language="auto"` builds when none is given: the profile's model,
# abstaining like the setting docs/benchmarks.md#conservative-detection tested on lid176.
AUTO_DETECTION_SETTINGS: dict[str, float] = {"min_letters": 4, "min_score": 0.5}

_NO_LINGUISTIC_CONTENT = "zxx"


class SourceLanguageError(UnsupportedLanguageError):
    """With `unknown_source="raise"`, a text's language couldn't be used."""


@dataclass(frozen=True)
class _Group:
    """Texts that share a source language, and the model that translates them."""

    source: Language | None
    indices: list[int]
    key: Hashable
    load: Callable[[], BaseModel]


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

    `source_language="auto"` detects each text's language with `detector` (by default
    the profile's `LanguageDetector`, abstaining on short or uncertain text), then
    translates each language with the model the profile picks for it. Texts already in
    the target language, and texts without linguistic content, come back unchanged.
    `unknown_source` decides what happens to texts whose language the detector can't
    tell or no model translates: `"keep"` returns them as given with status
    `UNKNOWN_SOURCE` or `UNSUPPORTED_SOURCE`, `"fallback"` translates them without a
    source language, and `"raise"` fails the call with `SourceLanguageError` before
    anything is translated.

    A profile keeps at most `max_loaded_models` models loaded (`None`: no limit). When a
    pair needs another one, the least recently used is dropped first, and loaded again if a
    later call needs it. A higher limit saves reloads when calls alternate between many
    pairs, at 0.6 to 1.2 GB of RAM per Opus-MT or SMaLL-100 model (docs/benchmarks.md).
    `unload()` drops them all.

    A profile downloads models into `cache_dir` (see `noentenc._cache` for the default),
    or with `only_local_files` only reads them from there, failing at once when a model
    is missing. `noentenc.prepare` downloads them ahead of time. For a `model` you pass,
    give these options to the model instead.

    `allowed_licenses` (SPDX identifiers, e.g. `["MIT", "Apache-2.0", "CC-BY-4.0"]`)
    and `max_download_bytes` restrict what a profile may pick. A model that breaks them
    is skipped for the profile's next choice, and when none is left the call raises
    `ModelConstraintError` before downloading anything (with `"auto"`, those texts get
    `UNSUPPORTED_SOURCE`). `plan()`, `supports()` and `supported_languages()` answer
    what a profile would do without loading or downloading a model.

    Languages can be `Language` members or codes and names that `to_language` accepts,
    such as `"es"`, `"spa"`, `"es-ES"` or `"spanish"`.
    """

    def __init__(
        self,
        model: BaseModel | Profile | str | None = None,
        *,
        max_loaded_models: int | None = DEFAULT_MAX_LOADED_MODELS,
        only_local_files: bool = False,
        cache_dir: str | Path | None = None,
        detector: LanguageDetector | None = None,
        allowed_licenses: Iterable[str] | None = None,
        max_download_bytes: int | None = None,
    ) -> None:
        self.constraints = Constraints.of(allowed_licenses, max_download_bytes)
        if isinstance(model, BaseModel):
            if only_local_files or cache_dir is not None or self.constraints.active:
                raise ValueError(
                    "only_local_files, cache_dir and the licence and size constraints "
                    "apply to the models a profile loads; pass a model that meets them"
                )
            self.model: BaseModel | None = model
            self.profile = Profile.SPEED  # only picks the default detector
        else:
            self.model = None
            self.profile = Profile(model or Profile.SPEED)
        self.only_local_files = only_local_files
        self.cache_dir = cache_dir
        self.max_loaded_models = _check_max_loaded_models(max_loaded_models)
        self._detector = detector
        # Loaded profile models, least recently used first.
        self._default_models: OrderedDict[Hashable, BaseModel] = OrderedDict()
        self._models_lock = threading.Lock()

    @property
    def detector(self) -> LanguageDetector:
        """The detector `source_language="auto"` uses, built on first use."""
        with self._models_lock:
            if self._detector is None:
                from noentenc.language_detection import LanguageDetector

                self._detector = LanguageDetector(
                    self.profile,
                    only_local_files=self.only_local_files,
                    cache_dir=self.cache_dir,
                    allowed_licenses=self.constraints.allowed_licenses,
                    max_download_bytes=self.constraints.max_download_bytes,
                    **AUTO_DETECTION_SETTINGS,  # ty: ignore[invalid-argument-type]
                )
            return self._detector

    def plan(
        self,
        target_language: Language | str,
        source_language: Language | str | None = None,
    ) -> Plan:
        """The models translating into `target_language` would load, and their costs.

        Loads and downloads nothing. With `source_language="auto"` the plan holds the
        detector and every model a detected language could be routed to. Raises like
        `translate` when no allowed model translates the pair.
        """
        from noentenc.language_detection.base import check_profile_plan

        if self.model is not None:
            raise ValueError(
                "plan() describes profile models; this translator was given one"
            )
        target, source, auto = _check_arguments(
            target_language, source_language, 32, "raise"
        )
        if not auto:
            choice = _routing.choose(self.profile, source, target, self.constraints)
            return Plan((choice.plan(self.cache_dir),))
        choices = dict.fromkeys(
            choice
            for candidate in (None, *Language)
            if candidate != target
            and (choice := self._choice_or_none(candidate, target)) is not None
        )
        detector = check_profile_plan(self.profile, self.constraints, self.cache_dir)
        return Plan((detector, *(choice.plan(self.cache_dir) for choice in choices)))

    def supports(
        self,
        target_language: Language | str,
        source_language: Language | str | None = None,
    ) -> bool:
        """Whether a model (allowed by the constraints) translates the pair. Loads nothing."""
        target, source, auto = _check_arguments(
            target_language, source_language, 32, "raise"
        )
        if self.model is not None:
            return auto or self.model.schema.supports(source, target)
        if auto:
            return self._choice_or_none(None, target) is not None or any(
                self._choice_or_none(candidate, target) is not None
                for candidate in Language
                if candidate != target
            )
        return self._choice_or_none(source, target) is not None

    def supported_languages(self) -> LanguageSchema:
        """The languages this translator reads and writes, under its constraints.

        `source` is `ANY_LANGUAGE` when some target is reachable without a source
        language. Not every source pairs with every target; `supports()` checks a pair.
        """
        if self.model is not None:
            return self.model.schema
        sources: set[Language] = set()
        targets: set[Language] = set()
        any_source = False
        for target in Language:
            if self._choice_or_none(None, target) is not None:
                any_source = True
                targets.add(target)
            for source in Language:
                if source != target and self._choice_or_none(source, target):
                    sources.add(source)
                    targets.add(target)
        return LanguageSchema(
            source=ANY_LANGUAGE if any_source else frozenset(sources),
            target=frozenset(targets),
        )

    def _choice_or_none(
        self, source: Language | None, target: Language
    ) -> _routing.ModelChoice | None:
        try:
            return _routing.choose(self.profile, source, target, self.constraints)
        except UnsupportedLanguageError:
            return None

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
        target_language: Language | str,
        source_language: Language | str | None = None,
        *,
        truncate: bool = False,
        detailed: Literal[False] = False,
        unknown_source: UnknownSourcePolicy = "keep",
    ) -> str: ...

    @overload
    def translate(
        self,
        text: str,
        target_language: Language | str,
        source_language: Language | str | None = None,
        *,
        truncate: bool = False,
        detailed: Literal[True],
        unknown_source: UnknownSourcePolicy = "keep",
    ) -> Translation: ...

    def translate(
        self,
        text: str,
        target_language: Language | str,
        source_language: Language | str | None = None,
        *,
        truncate: bool = False,
        detailed: bool = False,
        unknown_source: UnknownSourcePolicy = "keep",
    ) -> str | Translation:
        """Translate `text`, sentence by sentence, keeping its whitespace and line breaks.

        A sentence longer than the model reads raises `InputTooLongError`; with
        `truncate` its end is dropped instead. With `detailed` the result is a
        `Translation`, which says whether input was dropped or output cut short, which
        source language and model were used, and what the detector found.

        Blank text, and text whose `source_language` is `target_language`, come back
        unchanged without loading a model. `source_language="auto"` detects it.
        """
        (translation,) = self._translate(
            [text],
            target_language,
            source_language,
            32,
            truncate,
            "raise",
            unknown_source,
        )
        return translation if detailed else translation.text

    @overload
    def translate_batch(
        self,
        texts: list[str],
        target_language: Language | str,
        source_language: Language | str | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
        detailed: Literal[False] = False,
        errors: ErrorPolicy = "raise",
        unknown_source: UnknownSourcePolicy = "keep",
    ) -> list[str]: ...

    @overload
    def translate_batch(
        self,
        texts: list[str],
        target_language: Language | str,
        source_language: Language | str | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
        detailed: Literal[True],
        errors: ErrorPolicy = "raise",
        unknown_source: UnknownSourcePolicy = "keep",
    ) -> list[Translation]: ...

    def translate_batch(
        self,
        texts: list[str],
        target_language: Language | str,
        source_language: Language | str | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
        detailed: bool = False,
        errors: ErrorPolicy = "raise",
        unknown_source: UnknownSourcePolicy = "keep",
    ) -> list[str] | list[Translation]:
        """Translate each of `texts`, like `translate`, in input order.

        An empty list returns at once. With `errors="record"` a text whose translation
        raises comes back as given, and its detailed result has status `FAILED` and
        the error; the other texts are still translated. Arguments that apply to the
        whole call (batch size, languages, an unsupported pair) always raise.

        With `source_language="auto"` the texts are grouped by detected language, so
        each model loads once per call and translates its texts in batches.
        """
        translations = self._translate(
            texts,
            target_language,
            source_language,
            batch_size,
            truncate,
            errors,
            unknown_source,
        )
        return translations if detailed else [t.text for t in translations]

    @overload
    def translate_dataset(
        self,
        dataset: pl.DataFrame,
        target_column: str,
        result_column: str,
        target_language: Language | str,
        source_language: Language | str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        *,
        truncate: bool = False,
        errors: ErrorPolicy = "raise",
        error_column: str | None = None,
        unknown_source: UnknownSourcePolicy = "keep",
        status_column: str | None = None,
        source_column: str | None = None,
    ) -> pl.DataFrame: ...

    @overload
    def translate_dataset(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        result_column: str,
        target_language: Language | str,
        source_language: Language | str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        *,
        truncate: bool = False,
        errors: ErrorPolicy = "raise",
        error_column: str | None = None,
        unknown_source: UnknownSourcePolicy = "keep",
        status_column: str | None = None,
        source_column: str | None = None,
    ) -> pd.DataFrame: ...

    def translate_dataset(
        self,
        dataset: pl.DataFrame | pd.DataFrame,
        target_column: str,
        result_column: str,
        target_language: Language | str,
        source_language: Language | str | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        *,
        truncate: bool = False,
        errors: ErrorPolicy = "raise",
        error_column: str | None = None,
        unknown_source: UnknownSourcePolicy = "keep",
        status_column: str | None = None,
        source_column: str | None = None,
    ) -> pl.DataFrame | pd.DataFrame:
        """Return `dataset` with `result_column` holding the translation of `target_column`.

        Works with polars and pandas frames; rows keep their order and null texts stay
        null. Texts are translated like `translate_batch`, including `truncate`,
        `errors`, `source_language="auto"` and `unknown_source`. With
        `errors="record"` a row whose translation raises gets a null translation, and
        its error in `error_column` if given (null for the other rows).

        `status_column` adds each row's `TranslationStatus` and `source_column` the
        source language used (detected, with `"auto"`); both are null for null texts.
        With `show_progress` a tqdm bar tracks the rows translated so far.
        """
        _check_arguments(
            target_language, source_language, batch_size, errors, unknown_source
        )
        values = column_values(dataset, target_column)
        rows = [i for i, value in enumerate(values) if isinstance(value, str)]
        with tqdm(
            total=len(rows), desc="Translating", disable=not show_progress
        ) as progress:
            translations = self._translate(
                [values[i] for i in rows],
                target_language,
                source_language,
                batch_size,
                truncate,
                errors,
                unknown_source,
                progress=progress.update,
            )
        results: list[str | None] = [None] * len(values)
        failures: list[str | None] = [None] * len(values)
        statuses: list[str | None] = [None] * len(values)
        sources: list[str | None] = [None] * len(values)
        for i, translation in zip(rows, translations, strict=True):
            if translation.status is TranslationStatus.FAILED:
                failures[i] = translation.error
            else:
                results[i] = translation.text
            statuses[i] = str(translation.status)
            if translation.source_language is not None:
                sources[i] = str(translation.source_language)
        dataset = with_column(dataset, result_column, results, strings=True)
        for column, extra in (
            (error_column, failures),
            (status_column, statuses),
            (source_column, sources),
        ):
            if column is not None:
                dataset = with_column(dataset, column, extra, strings=True)
        return dataset

    def _translate(
        self,
        texts: list[str],
        target_language: Language | str,
        source_language: Language | str | None,
        batch_size: int,
        truncate: bool,
        errors: ErrorPolicy,
        unknown_source: UnknownSourcePolicy,
        progress: Callable[[int], object] | None = None,
    ) -> list[Translation]:
        """Translate `texts`, answering what needs no model before loading one."""
        check_texts(texts)
        target, source, auto = _check_arguments(
            target_language, source_language, batch_size, errors, unknown_source
        )
        # Blank texts and same-language requests stay as given; translate the rest.
        results = [
            Translation(t, status=TranslationStatus.UNCHANGED, source_language=source)
            for t in texts
        ]
        pending = [i for i, text in enumerate(texts) if not is_blank(text)]
        if auto:
            groups = self._auto_groups(
                texts, pending, target, batch_size, unknown_source, results
            )
        elif pending and source != target:
            key, load = self._route(source, target)
            groups = [_Group(source, pending, key, load)]
        else:
            groups = []
        if progress is not None:
            progress(len(texts) - sum(len(group.indices) for group in groups))
        # Texts translated by the same model go together, so it loads once per call.
        order = {group.key: n for n, group in reversed(list(enumerate(groups)))}
        for group in sorted(groups, key=lambda g: order[g.key]):
            model = group.load()
            name = _model_name(model)
            chunk = len(group.indices) if progress is None else batch_size
            for start in range(0, len(group.indices), chunk):
                indices = group.indices[start : start + chunk]
                translated = self._run(
                    model,
                    [texts[i] for i in indices],
                    target,
                    group.source,
                    batch_size,
                    truncate,
                    errors,
                )
                for i, translation in zip(indices, translated, strict=True):
                    results[i] = replace(
                        translation,
                        source_language=group.source,
                        detected_language=results[i].detected_language,
                        detection_score=results[i].detection_score,
                        model=name,
                    )
                if progress is not None:
                    progress(len(indices))
        return results

    @staticmethod
    def _run(
        model: BaseModel,
        texts: list[str],
        target: Language,
        source: Language | None,
        batch_size: int,
        truncate: bool,
        errors: ErrorPolicy,
    ) -> list[Translation]:
        def run(batch: list[str]) -> list[Translation]:
            return model.predict_batch_detailed(
                batch, target, source, batch_size, truncate=truncate
            )

        return run(texts) if errors == "raise" else _record_failures(texts, run)

    def _auto_groups(
        self,
        texts: list[str],
        pending: list[int],
        target: Language,
        batch_size: int,
        unknown_source: UnknownSourcePolicy,
        results: list[Translation],
    ) -> list[_Group]:
        """Detect the language of each pending text and group them by source.

        Fills `results` for the texts that won't be translated, and raises before any
        model loads when `unknown_source="raise"` meets a text it can't translate.
        """
        detections = self.detector.detect_batch(
            [texts[i] for i in pending], batch_size=batch_size, detailed=True
        )
        by_source: dict[Language | None, list[int]] = {}
        unusable: list[tuple[int, TranslationStatus]] = []
        for i, detection in zip(pending, detections, strict=True):
            results[i] = replace(
                results[i],
                detected_language=detection.language,
                detection_score=detection.score,
            )
            if detection.language == _NO_LINGUISTIC_CONTENT:
                continue
            language = to_language(detection.language, None)
            if language == target:
                results[i] = replace(results[i], source_language=language)
            elif language is None:
                status = (
                    TranslationStatus.UNKNOWN_SOURCE
                    if detection.language == "und"
                    else TranslationStatus.UNSUPPORTED_SOURCE
                )
                unusable.append((i, status))
            else:
                by_source.setdefault(language, []).append(i)
        groups: list[_Group] = []
        for language, indices in by_source.items():
            try:
                key, load = self._route(language, target)
            except UnsupportedLanguageError:
                unusable += [(i, TranslationStatus.UNSUPPORTED_SOURCE) for i in indices]
                continue
            groups.append(_Group(language, indices, key, load))
        return groups + self._unusable(texts, target, unusable, unknown_source, results)

    def _unusable(
        self,
        texts: list[str],
        target: Language,
        unusable: list[tuple[int, TranslationStatus]],
        policy: UnknownSourcePolicy,
        results: list[Translation],
    ) -> list[_Group]:
        """Apply `unknown_source` to texts that can't be translated from their language."""
        if not unusable:
            return []
        if policy == "raise":
            i, status = min(unusable)
            raise SourceLanguageError(
                f"texts[{i}] ({status}, detected {results[i].detected_language!r}): "
                f"{texts[i][:60]!r}; pass unknown_source='keep' or 'fallback' to "
                "translate the other texts"
            )
        if policy == "fallback":
            try:
                key, load = self._route(None, target)
            except UnsupportedLanguageError:
                pass
            else:
                return [_Group(None, sorted(i for i, _ in unusable), key, load)]
        for i, status in unusable:
            results[i] = replace(results[i], status=status)
        return []

    def _route(
        self, source: Language | None, target: Language
    ) -> tuple[Hashable, Callable[[], BaseModel]]:
        """The cache key and loader of the model for a pair; raises if there is none."""
        if self.model is not None:
            # Checked here so an unsupported pair fails the call even when
            # `errors="record"` keeps per-text failures.
            self.model.schema.validate(source, target, type(self.model).__name__)
            model = self.model
            return id(model), lambda: model
        choice = _routing.choose(self.profile, source, target, self.constraints)
        return choice, lambda: self._cached(
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


def _model_name(model: BaseModel) -> str:
    return f"{type(model).__name__}({getattr(model, 'model', '')})"


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
    target: Language | str,
    source: Language | str | None,
    batch_size: int,
    errors: ErrorPolicy,
    unknown_source: UnknownSourcePolicy = "keep",
) -> tuple[Language, Language | None, bool]:
    """Validate what applies to a whole call.

    Returns the target and source as `Language`s (the source None when not given or
    `"auto"`), and whether the source is `"auto"`.
    """
    check_batch_size(batch_size)
    if errors not in ("raise", "record"):
        raise ValueError(f"errors must be 'raise' or 'record', got {errors!r}")
    if unknown_source not in ("keep", "fallback", "raise"):
        raise ValueError(
            "unknown_source must be 'keep', 'fallback' or 'raise', "
            f"got {unknown_source!r}"
        )
    auto = source == AUTO
    if source is None or auto:
        return to_language(target), None, auto
    return to_language(target), to_language(source), False


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
