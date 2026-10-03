"""Language-identification backends.

``FastTextModel`` and ``OnnxClassifierModel`` use core dependencies only; the other backends
import their optional dependency lazily, when they are constructed.
"""

from noentenc.language_detection.models.base import BaseModel
from noentenc.language_detection.models.cld3_model import Cld3Model
from noentenc.language_detection.models.fasttext import FastTextModel
from noentenc.language_detection.models.heliport_model import HeliportModel
from noentenc.language_detection.models.langdetect_model import LangdetectModel
from noentenc.language_detection.models.langid_model import LangidModel
from noentenc.language_detection.models.lingua_model import LinguaModel
from noentenc.language_detection.models.onnx_classifier import OnnxClassifierModel

__all__ = [
    "BaseModel",
    "Cld3Model",
    "FastTextModel",
    "HeliportModel",
    "LangdetectModel",
    "LangidModel",
    "LinguaModel",
    "OnnxClassifierModel",
]
