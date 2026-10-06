"""End-to-end translations with real weights.

They run when NOENTENC_RUN_INTEGRATION=1 (downloading weights) or when the weights are
already in the noentenc cache; otherwise they are skipped.
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


def test_long_text_keeps_every_sentence_and_line_break() -> None:
    model = load(lambda local: OpusMTModel.from_pair(EN, ES, only_local_files=local))
    paragraph = " ".join(
        f"Sentence number {i} talks about the weather in a small town near the sea."
        for i in range(40)
    )
    text = f"{paragraph}\n\n{paragraph}\nThe last line is about Barcelona."
    result = Translator(model).translate(text, ES, EN, detailed=True)
    first, blank, second, last = result.text.split("\n")
    assert blank == ""
    assert "39" in first
    assert "39" in second
    assert "Barcelona" in last
    assert not result.input_truncated
    assert not result.output_limit_reached


def test_stream_with_capped_threads_matches_batch() -> None:
    load(lambda local: OpusMTModel.from_pair(EN, ES, only_local_files=local))
    translator = Translator(num_threads=1)
    streamed = translator.translate_stream(
        iter(SENTENCES), ES, EN, batch_size=5, chunk_size=6
    )
    assert list(streamed) == translator.translate_batch(SENTENCES, ES, EN, batch_size=5)
    (model,) = translator.loaded_models
    assert isinstance(model, OpusMTModel)
    for session in (model._engine._encoder, model._engine._decoder):
        assert session.get_session_options().intra_op_num_threads == 1
