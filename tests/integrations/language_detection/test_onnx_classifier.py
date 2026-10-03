import pytest

from noentenc.language_detection.models.onnx_classifier import (
    PRESETS,
    OnnxClassifierModel,
)
from tests.integrations.language_detection.helpers import skip_unless_available

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_real_onnx_models(preset: str) -> None:
    pytest.importorskip("onnxruntime")
    remote = PRESETS[preset]
    skip_unless_available(
        remote.remote(remote.onnx_file), remote.remote("tokenizer.json")
    )
    model = OnnxClassifierModel(preset)
    assert model.predict_batch(
        [
            "The weather is lovely today.",
            "",
            "Hola, ¿cómo estás? Espero que todo vaya bien.",
        ]
    ) == [
        "eng",
        "und",
        "spa",
    ]
