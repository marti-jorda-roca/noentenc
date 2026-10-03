"""End-to-end translations with real weights.

They run when NOENTENC_RUN_INTEGRATION=1 (downloading weights) or when the weights are
already in the Hugging Face cache; otherwise they are skipped.
"""

import os
import warnings
from collections.abc import Callable

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from noentenc.languages import Language
from noentenc.translation import (
    M2M100Model,
    NLLBModel,
    OpusMTModel,
    SMaLL100Model,
    Translator,
)
from noentenc.translation.models._seq2seq import Seq2SeqModel

RUN_INTEGRATION = os.environ.get("NOENTENC_RUN_INTEGRATION") == "1"
EN, ES, FR, DE = Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.GERMAN
SENTENCES = [
    "The weather is nice today.",
    "Hi!",
    "The European Central Bank kept interest rates unchanged on Thursday, citing "
    "persistent uncertainty about inflation.",
    "Where is the train station?",
] * 4

pytestmark = pytest.mark.integration


def load(factory: Callable[[bool], Seq2SeqModel]) -> Seq2SeqModel:
    if RUN_INTEGRATION:
        return factory(False)
    try:
        return factory(True)
    except LocalEntryNotFoundError:
        pytest.skip("model weights not cached; set NOENTENC_RUN_INTEGRATION=1")


def assert_batch_matches_single(
    model: Seq2SeqModel, target: Language, source: Language
) -> None:
    batch = model.predict_batch(SENTENCES, target, source, batch_size=5)
    singles = {text: model.predict(text, target, source) for text in set(SENTENCES)}
    assert batch == [singles[text] for text in SENTENCES]


def test_opus_mt_en_es() -> None:
    model = load(lambda local: OpusMTModel.from_pair(EN, ES, only_local_files=local))
    assert "tiempo" in model.predict("The weather is nice today.", ES).lower()
    assert_batch_matches_single(model, ES, EN)


def test_opus_mt_target_prefix_en_zh() -> None:
    model = load(
        lambda local: OpusMTModel.from_pair(
            EN, Language.CHINESE, only_local_files=local
        )
    )
    assert "天气" in model.predict("The weather is nice today.", Language.CHINESE)


def test_small100_en_fr_without_source() -> None:
    model = load(lambda local: SMaLL100Model(only_local_files=local))
    assert "temps" in model.predict("The weather is nice today.", FR).lower()
    assert_batch_matches_single(model, FR, EN)


def test_m2m100_en_de() -> None:
    model = load(lambda local: M2M100Model(only_local_files=local))
    assert "wetter" in model.predict("The weather is nice today.", DE, EN).lower()


def test_nllb_en_es() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = load(lambda local: NLLBModel(only_local_files=local))
    assert "tiempo" in model.predict("The weather is nice today.", ES, EN).lower()


def test_default_translator() -> None:
    load(lambda local: OpusMTModel.from_pair(EN, ES, only_local_files=local))
    assert (
        "tiempo" in Translator().translate("The weather is nice today.", ES, EN).lower()
    )
