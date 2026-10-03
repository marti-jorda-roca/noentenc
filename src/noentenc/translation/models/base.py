from abc import ABC
from pathlib import Path

from noentenc.languages import Language, LanguageSchema


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
        raise NotImplementedError

    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        raise NotImplementedError
