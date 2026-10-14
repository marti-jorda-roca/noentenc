from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from noentenc._plan import ModelPlan
from noentenc.languages import Language, LanguageSchema


class InputTooLongError(ValueError):
    """A sentence is longer than the model can read, and truncation wasn't requested."""


class TranslationStatus(StrEnum):
    TRANSLATED = "translated"
    # Returned as given without calling a model: blank, already in the target language,
    # or (with `source_language="auto"`) without linguistic content, such as a bare link.
    UNCHANGED = "unchanged"
    # The model raised, and `errors="record"` kept the original text.
    FAILED = "failed"
    # `source_language="auto"` couldn't tell the language, so the text was kept as given.
    UNKNOWN_SOURCE = "unknown_source"
    # `source_language="auto"` detected a language that no model translates into the
    # target, so the text was kept as given.
    UNSUPPORTED_SOURCE = "unsupported_source"


@dataclass(frozen=True)
class Translation:
    """A translated text, whether any of it may be missing, and how it was produced."""

    text: str
    # A sentence was longer than the model reads and its end was dropped
    # (only with `truncate=True`).
    input_truncated: bool = False
    # Generation hit its token limit before the model finished a sentence, so the
    # output may be cut short or end in repetition.
    output_limit_reached: bool = False
    status: TranslationStatus = TranslationStatus.TRANSLATED
    # Why the translation failed, as "ErrorType: message", when `status` is FAILED.
    error: str | None = None
    # The source language used: the one passed, or the detected one.
    source_language: Language | None = None
    # With `source_language="auto"`: the detector's label (ISO 639-3, "und" when it
    # abstained, "zxx" for no linguistic content) and the score of its top label.
    detected_language: str | None = None
    detection_score: float | None = None
    # The model that translated the text, e.g. "OpusMTModel(Xenova/opus-mt-de-en)".
    model: str | None = None
    # How URLs, emails, code, placeholders, tags and numbers were kept unchanged:
    # "placeholders" (swapped out and back), "segments" (the prose between them was
    # translated piece by piece, because the text had a fenced code block or placeholders
    # didn't survive), or None when the text had none or `preserve=False`.
    preservation: str | None = None
    # A sentence was translated in pieces (only with `split_long=True`): it was longer
    # than the model reads, or its output hit the length limit and its halves didn't.
    input_split: bool = False


class BaseModel(ABC):
    # Languages the model reads and writes; every model sets it.
    schema: LanguageSchema

    def __init__(self, model: str | Path, only_local_files: bool = False) -> None:
        self.model = model
        self.only_local_files = only_local_files

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={str(self.model)!r})"

    def plan(self) -> ModelPlan:
        """What this loaded model is and costs, as `Translator.plan()` reports it.

        This default knows only the model's name; built-in models override it with
        their precision, licence, size and memory.
        """
        return ModelPlan(
            task="translation",
            name=f"{type(self).__name__}({getattr(self, 'model', '')})",
            precision=None,
            license="unknown",
            download_bytes=0,
            memory_bytes=0,
            memory_basis="unknown",
            cached=True,
        )

    def predict(
        self,
        text: str,
        target_language: Language,
        source_language: Language | None = None,
    ) -> str:
        return self.predict_batch([text], target_language, source_language)[0]

    @abstractmethod
    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> list[str]: ...

    def predict_batch_detailed(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
        *,
        truncate: bool = False,
        split_long: bool = False,
    ) -> list[Translation]:
        """Like `predict_batch`, with what was dropped on the way in or out.

        This default ignores `truncate` and `split_long` and reports nothing missing;
        models that know their length limits override it. `Translator` only passes
        `split_long` when it's True, so overrides written without it keep working.
        """
        return [
            Translation(text)
            for text in self.predict_batch(
                texts, target_language, source_language, batch_size
            )
        ]
