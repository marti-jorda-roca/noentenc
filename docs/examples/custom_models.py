"""Plug your own models into `LanguageDetector` and `Translator`.

See docs/custom-models.md for the full guide.
Run: uv run python docs/examples/custom_models.py
"""

import unicodedata
from collections import Counter

from noentenc import Language
from noentenc.language_detection import BaseModel as DetectionModel
from noentenc.language_detection import LanguageDetector
from noentenc.translation import OpusMTModel, Translator
from noentenc.translation.models.base import BaseModel as TranslationModel

# --- Language detection ---------------------------------------------------------------

# Unicode script name prefix -> the ISO 639-3 code it identifies on its own.
SCRIPTS = {"GREEK": "ell", "HEBREW": "heb", "THAI": "tha", "HANGUL": "kor"}


class ScriptModel(DetectionModel):
    """Identifies languages that have a script of their own, from character names alone.

    Subclasses only implement `labels`, `_predict_chunk` and `_predict_score_chunk`.
    The base class handles batching and returns "und" for empty texts.
    """

    def __init__(self) -> None:
        super().__init__("unicode-scripts")

    @property
    def labels(self) -> list[str]:
        return [*SCRIPTS.values(), "und"]

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        return [next(iter(scores)) for scores in self._predict_score_chunk(texts, 1)]

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        return [self._scores(text, top_k) for text in texts]

    def _scores(self, text: str, top_k: int | None) -> dict[str, float]:
        counts = Counter(
            code
            for char in text
            if char.isalpha()
            for prefix, code in SCRIPTS.items()
            if unicodedata.name(char, "").startswith(prefix)
        )
        total = sum(counts.values())
        if not total:
            return {"und": 1.0}
        # Sorted descending, as the base class expects.
        return {code: n / total for code, n in counts.most_common(top_k)}


detector = LanguageDetector(ScriptModel())
print(detector.detect_batch(["Καλημέρα", "שלום עולם", "Hello", ""]))
# ['ell', 'heb', 'und', 'und']

# --- Translation ----------------------------------------------------------------------


class TranslationMemoryModel(TranslationModel):
    """Returns human-approved translations when it has one; asks `fallback` otherwise."""

    def __init__(
        self,
        memory: dict[tuple[str, Language], str],
        fallback: TranslationModel,
    ) -> None:
        super().__init__("translation-memory")
        self.memory = memory
        self.fallback = fallback
        # Every model declares a schema; this one handles what its fallback handles.
        self.schema = fallback.schema

    def predict(
        self,
        text: str,
        target_language: Language,
        source_language: Language | None = None,
    ) -> str:
        return self.predict_batch([text], target_language, source_language)[0]

    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        self.schema.validate(source_language, target_language, type(self).__name__)
        misses = [t for t in texts if (t, target_language) not in self.memory]
        translated = dict(
            zip(
                misses,
                self.fallback.predict_batch(
                    misses, target_language, source_language, batch_size
                ),
                strict=True,
            )
        )
        return [
            self.memory.get((text, target_language)) or translated[text]
            for text in texts
        ]


memory = {("Sign in", Language.SPANISH): "Iniciar sesión"}
model = TranslationMemoryModel(
    memory, OpusMTModel.from_pair(Language.ENGLISH, Language.SPANISH)
)
translator = Translator(model)
print(
    translator.translate_batch(["Sign in", "Forgot your password?"], Language.SPANISH)
)
# ['Iniciar sesión', '¿Olvidaste tu contraseña?']
