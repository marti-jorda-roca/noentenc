from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from noentenc.languages import Language, LanguageSchema


class InputTooLongError(ValueError):
    """A sentence is longer than the model can read, and truncation wasn't requested."""


@dataclass(frozen=True)
class Translation:
    """A translated text and whether any of it may be missing."""

    text: str
    # A sentence was longer than the model reads and its end was dropped
    # (only with `truncate=True`).
    input_truncated: bool = False
    # Generation hit its token limit before the model finished a sentence, so the
    # output may be cut short or end in repetition.
    output_limit_reached: bool = False


class BaseModel(ABC):
    # Languages the model reads and writes; every model sets it.
    schema: LanguageSchema

    def __init__(self, model: str | Path, only_local_files: bool = False) -> None:
        self.model = model
        self.only_local_files = only_local_files

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
    ) -> list[Translation]:
        """Like `predict_batch`, with what was dropped on the way in or out.

        This default ignores `truncate` and reports nothing missing; models that know
        their length limits override it.
        """
        return [
            Translation(text)
            for text in self.predict_batch(
                texts, target_language, source_language, batch_size
            )
        ]
