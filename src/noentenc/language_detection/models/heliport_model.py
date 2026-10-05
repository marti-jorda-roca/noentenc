"""heliport backend: a Rust port of HeLI-OTS (220 languages). GPL-3.0; no Windows wheels."""

from noentenc._optional import require
from noentenc.language_detection.labels import NO_LINGUISTIC_CONTENT, UNDETERMINED
from noentenc.language_detection.models.base import BaseModel

_ISO_LENGTH = 3


def heliport_code(raw: str) -> str:
    """The ISO code in one of heliport's sub-labels.

    A macrolanguage followed by one of its languages (``hbsbos``, ``msazsm``) is that
    language; any other sub-label (``finl``, ``msamalay``, ``undhtml``) is its prefix.
    """
    if len(raw) == 2 * _ISO_LENGTH:
        return raw[_ISO_LENGTH:]
    return raw[:_ISO_LENGTH]


class HeliportModel(BaseModel):
    """Wraps ``heliport`` (``uv add 'noentenc[heliport]'``).

    heliport reports confidence scores, not probabilities. ``predict_score`` therefore returns
    only the top language with its confidence (higher is more confident; not bounded by 1).
    Batches run in parallel in Rust.
    """

    scores_every_label = False

    def __init__(
        self,
        model: str = "heliport",
        only_local_files: bool = False,
        normalize_labels: bool = True,
        collapse_macrolanguages: bool = False,
    ) -> None:
        super().__init__(
            model, only_local_files, normalize_labels, collapse_macrolanguages
        )
        heliport = require("heliport", "heliport")
        self._identifier = heliport.Identifier()
        native = sorted(
            {heliport_code(c) for c in self._identifier.get_confidence_all()}
        )
        self._mapper = self._label_mapper(
            [*native, UNDETERMINED, NO_LINGUISTIC_CONTENT]
        )

    def _label(self, native: str) -> str:
        return super()._label(heliport_code(native))

    def _scored(self, texts: list[str]) -> list[tuple[str, float]]:
        if len(texts) == 1:
            return [self._identifier.identify_with_score(texts[0])]
        return self._identifier.par_identify_with_score(texts)

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        if len(texts) == 1:
            return [self._label(self._identifier.identify(texts[0]))]
        return [self._label(code) for code in self._identifier.par_identify(texts)]

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:  # noqa: ARG002 - only top-1 exists
        return [
            {self._label(code): float(score)} for code, score in self._scored(texts)
        ]
