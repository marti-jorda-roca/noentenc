"""Real fastText weights against predictions recorded from the reference ``fasttext`` package."""

import pytest

from noentenc.language_detection.models.fasttext import PRESETS, FastTextModel
from tests.integrations.language_detection.helpers import skip_unless_available
from tests.unit.language_detection.helpers import (
    FASTTEXT_EXPECTED,
    assert_fasttext_parity,
    load_sentences,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "preset", sorted(name for name in FASTTEXT_EXPECTED if name in PRESETS)
)
def test_real_models_match_fasttext(preset: str) -> None:
    skip_unless_available(PRESETS[preset].remote)
    model = FastTextModel(preset, normalize_labels=False)
    assert_fasttext_parity(model, load_sentences(), FASTTEXT_EXPECTED[preset])
