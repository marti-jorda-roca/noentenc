"""Google CLD3 backend: a tiny feed-forward network over character n-grams (107 languages)."""

from noentenc.language_detection._optional import require
from noentenc.language_detection.labels import (
    UNDETERMINED,
    LabelMapper,
    normalize_scores,
    to_iso639_3,
)
from noentenc.language_detection.models.base import BaseModel

CLD3_LANGUAGES = (  # noqa: SIM905 - a word list reads better than 107 quoted strings
    "af am ar bg bg-Latn bn bs ca ceb co cs cy da de el el-Latn en eo es et eu fa fi fil fr "
    "fy ga gd gl gu ha haw hi hi-Latn hmn hr ht hu hy id ig is it iw ja ja-Latn jv ka kk km "
    "kn ko ku ky la lb lo lt lv mg mi mk ml mn mr ms mt my ne nl no ny pa pl ps pt ro ru "
    "ru-Latn sd si sk sl sm sn so sq sr st su sv sw ta te tg th tr uk ur uz vi xh yi yo zh "
    "zh-Latn zu"
).split()
DEFAULT_TOP_K = 3


class Cld3Model(BaseModel):
    """Wraps ``cld3-py`` (imported as ``gcld3``; ``uv add 'noentenc[cld3]'``).

    ``predict_score`` uses CLD3's ``FindTopNMostFreqLangs``: it splits the text into
    single-language spans and reports each span language's own probability, so the values are
    not a distribution over all languages and don't sum to 1.
    """

    def __init__(
        self,
        model: str = "cld3",
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
        max_num_bytes: int = 1000,
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        gcld3 = require("gcld3", "cld3")
        self._identifier = gcld3.NNetLanguageIdentifier(
            min_num_bytes=0, max_num_bytes=max_num_bytes
        )
        self._mapper = LabelMapper(
            CLD3_LANGUAGES,
            normalize=normalize_labels,
            collapse_macrolanguages=collapse_macrolanguages,
        )

    @property
    def labels(self) -> list[str]:
        return list(self._mapper.labels)

    def _label(self, code: str) -> str:
        if code == UNDETERMINED:
            return code
        if not self.normalize_labels:
            return code
        return to_iso639_3(code, collapse_macrolanguages=self.collapse_macrolanguages)

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        return [
            self._label(self._identifier.FindLanguage(text=text).language)
            for text in texts
        ]

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        results = []
        for text in texts:
            found = self._identifier.FindTopNMostFreqLangs(
                text=text, num_langs=top_k or DEFAULT_TOP_K
            )
            raw = {
                r.language: r.probability for r in found if r.language != UNDETERMINED
            }
            scores = normalize_scores(
                raw,
                normalize=self.normalize_labels,
                collapse_macrolanguages=self.collapse_macrolanguages,
            )
            results.append(
                dict(sorted(scores.items(), key=lambda kv: -kv[1]))
                or {UNDETERMINED: 1.0}
            )
        return results
