from typing import ClassVar

from noentenc.languages import ANY_LANGUAGE, Language, LanguageSchema
from noentenc.translation.models._seq2seq import Precision
from noentenc.translation.models.m2m100 import M2M100_CODES, M2M100Model


class SMaLL100Model(M2M100Model):
    """SMaLL-100: M2M100 distilled to 333M params with a 3-layer decoder.

    Unlike M2M100, the target-language token goes on the source side and the
    encoder is not conditioned on the source language, so none is needed.
    """

    schema = LanguageSchema(source=ANY_LANGUAGE, target=frozenset(M2M100_CODES))
    default_model: ClassVar[str] = "casawolice/small100-onnx"
    default_revision: ClassVar[str] = "5c2c73ac70bee9c58f5a7ac5e84a36bee25db8ee"
    default_precision: ClassVar[Precision] = Precision.INT8
    # This export only ships int8 weights, under the unsuffixed file names.
    onnx_files: ClassVar[dict[Precision, tuple[str, str]]] = {
        Precision.INT8: ("onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"),
    }

    def _frame(
        self, ids: list[int], source: Language | None, target: Language
    ) -> list[int]:
        return [self._language_id(target), *ids, self.config["eos_token_id"]]

    def _forced_first_id(self, target: Language) -> int | None:
        return None
