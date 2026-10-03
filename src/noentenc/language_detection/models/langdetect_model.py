"""langdetect backend: a port of Google's language-detection library (55 languages). Benchmark baseline."""

from noentenc.language_detection._optional import require
from noentenc.language_detection.labels import UNDETERMINED
from noentenc.language_detection.models.base import BaseModel


class LangdetectModel(BaseModel):
    """Wraps ``langdetect`` (``uv add 'noentenc[langdetect]'``).

    langdetect is randomised; ``seed`` makes it deterministic (it sets the library-global
    ``DetectorFactory.seed``). Texts without usable features return ``"und"``.
    """

    def __init__(
        self,
        model: str = "langdetect",
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
        seed: int | None = 0,
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        langdetect = require("langdetect", "langdetect")
        detector_factory = require("langdetect.detector_factory", "langdetect")
        langdetect.DetectorFactory.seed = seed
        detector_factory.init_factory()
        self._langdetect = langdetect
        self._mapper = self._label_mapper(
            detector_factory._factory.get_lang_list()  # noqa: SLF001 - no public accessor
        )

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        out = []
        for text in texts:
            try:
                out.append(self._label(self._langdetect.detect(text)))
            except self._langdetect.LangDetectException:
                out.append(UNDETERMINED)
        return out

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        out = []
        for text in texts:
            try:
                ranked = self._langdetect.detect_langs(text)  # sorted descending
            except self._langdetect.LangDetectException:
                out.append({UNDETERMINED: 1.0})
                continue
            out.append(self._normalize_scores({r.lang: r.prob for r in ranked[:top_k]}))
        return out
