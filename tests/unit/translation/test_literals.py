"""URLs, emails, code, placeholders, tags and numbers survive translation byte-for-byte."""

import re

import polars as pl
import pytest

from noentenc import Language
from noentenc.translation import Translation, TranslationStatus, Translator
from noentenc.translation._literals import (
    find_literals,
    mask,
    split_at_literals,
    translate_preserving,
)
from tests.unit.translation.test_translator import UpperModel

ES = Language.SPANISH


def literals(text: str) -> list[str]:
    return [text[a:b] for a, b in find_literals(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Visit https://example.com/reset?id=42 or email support@acme.io",
            ["https://example.com/reset?id=42", "support@acme.io"],
        ),
        # Sentence punctuation after a URL or domain isn't part of it.
        ("Go to https://acme.io/a?b=1&c=2.", ["https://acme.io/a?b=1&c=2"]),
        ("Go to www.acme.io, then acme.io/help!", ["www.acme.io", "acme.io/help"]),
        (
            "(see https://en.wikipedia.org/wiki/Foo_(bar))",
            ["https://en.wikipedia.org/wiki/Foo_(bar)"],
        ),
        ("ping @ana in #support", ["@ana", "#support"]),
        ("Run `pip install noentenc` now", ["`pip install noentenc`"]),
        (
            "Hi {name}, {0} {{order_id}} ${amount} %s %(count)d",
            ["{name}", "{0}", "{{order_id}}", "${amount}", "%s", "%(count)d"],
        ),
        (
            '<a href="https://x.io">here</a> &amp; &#39;',
            ['<a href="https://x.io">', "</a>", "&amp;", "&#39;"],
        ),
        (
            "Read [the guide](https://docs.acme.io/start).",
            ["](https://docs.acme.io/start)"],
        ),
        (
            "1,299.99 EUR at 14:30 on 12/05/2026, ticket 42",
            ["1,299.99", "14:30", "12/05/2026", "42"],
        ),
        # Not literals: a title before a name, abbreviations, numbers inside words.
        ("Mr.Smith said e.g. this at 3pm", []),
    ],
)
def test_find_literals(text: str, expected: list[str]) -> None:
    assert literals(text) == expected


def test_mask_and_restore_round_trip() -> None:
    text = "Visit https://a.io or https://a.io, then email x@y.io"
    masked = mask(text)
    assert masked is not None
    assert masked.text == "Visit ZXQ0 or ZXQ1, then email ZXQ2"
    assert masked.literals == ("https://a.io", "https://a.io", "x@y.io")
    assert masked.restore("Visite ZXQ0 o ZXQ1, y escriba a ZXQ2") == (
        "Visite https://a.io o https://a.io, y escriba a x@y.io"
    )
    # Placeholders may move, but each must come back exactly once.
    assert masked.restore("ZXQ2 ZXQ1 ZXQ0") == "x@y.io https://a.io https://a.io"
    assert masked.restore("Visite ZXQ0 o ZXQ1") is None  # dropped
    assert masked.restore("ZXQ0 ZXQ0 ZXQ1 ZXQ2") is None  # repeated
    assert masked.restore("ZXQ0 Zxq1 ZXQ2") is None  # altered
    assert masked.restore("ZXQ0 ZXQ1 ZXQ2 ZXQ3") is None  # invented


def test_many_literals_do_not_confuse_placeholder_numbers() -> None:
    text = " ".join(f"#{i}" for i in range(12))
    masked = mask(text)
    assert masked is not None
    assert masked.restore(masked.text) == text
    assert masked.restore(masked.text.replace("ZXQ10", "ZXQ1")) is None


def test_placeholders_next_to_words_get_padding() -> None:
    masked = mask("Click <a>here</a> now")
    assert masked is not None
    assert masked.text == "Click ZXQ0 here ZXQ1 now"
    assert (
        masked.restore("Haga clic ZXQ0 aquí ZXQ1 ahora")
        == "Haga clic <a>aquí</a> ahora"
    )
    # The padding is optional on the way back.
    assert masked.restore("ZXQ0aquíZXQ1") == "<a>aquí</a>"


def test_placeholder_collisions_pick_another_stem() -> None:
    masked = mask("ZXQ0 is a ticket, see https://a.io")
    assert masked is not None
    assert masked.stem == "QZX"
    assert masked.text == "ZXQ0 is a ticket, see QZX0"
    assert mask("zxq and qzx, see https://a.io") is None  # no free stem


def test_split_at_literals() -> None:
    assert split_at_literals("Visit https://a.io now") == [
        ("Visit ", False),
        ("https://a.io", True),
        (" now", False),
    ]


def upper(texts: list[str]) -> list[Translation]:
    return [Translation(text.upper()) for text in texts]


def test_translate_preserving_keeps_literals_and_translates_prose() -> None:
    texts = ["visit https://a.io/x?q=1 or mail x@y.io", "no literals", ""]
    results = translate_preserving(texts, upper)
    assert [r.text for r in results] == [
        "VISIT https://a.io/x?q=1 OR MAIL x@y.io",
        "NO LITERALS",
        "",
    ]
    assert [r.preservation for r in results] == ["placeholders", None, None]


def test_lost_placeholders_fall_back_to_pieces() -> None:
    calls: list[list[str]] = []

    def drops_placeholders(texts: list[str]) -> list[Translation]:
        calls.append(texts)
        return [Translation(re.sub(r"ZXQ\d+", "", text).upper()) for text in texts]

    (result,) = translate_preserving(
        ["visit https://a.io or mail x@y.io now"], drops_placeholders
    )
    assert result.text == "VISIT https://a.io OR MAIL x@y.io NOW"
    assert result.preservation == "segments"
    # One batch with placeholders, then one with the pieces of every failed text.
    assert calls == [["visit ZXQ0 or mail ZXQ1 now"], ["visit", "or mail", "now"]]


def test_texts_without_a_free_stem_go_to_pieces() -> None:
    (result,) = translate_preserving(["zxq qzx see https://a.io"], upper)
    assert result.text == "ZXQ QZX SEE https://a.io"
    assert result.preservation == "segments"


def test_piece_failures_fail_the_text() -> None:
    def fails(texts: list[str]) -> list[Translation]:
        return [
            Translation(t, status=TranslationStatus.FAILED, error="E: x")
            if "or" in t
            else Translation(re.sub(r"ZXQ\d", "", t))
            for t in texts
        ]

    (result,) = translate_preserving(["visit https://a.io or mail x@y.io"], fails)
    assert result.status is TranslationStatus.FAILED
    assert result.text == "visit https://a.io or mail x@y.io"


@pytest.fixture
def translator() -> Translator:
    return Translator(UpperModel())


TEXT = "visit https://example.com/reset?id=42 or email support@acme.io"
KEPT = "VISIT https://example.com/reset?id=42 OR EMAIL support@acme.io:es"


def test_single_batch_and_dataset_preserve_literals(translator: Translator) -> None:
    assert translator.translate(TEXT, ES, "en") == KEPT
    assert translator.translate_batch([TEXT, "hola"], ES, "en") == [KEPT, "HOLA:es"]
    frame = pl.DataFrame({"text": [TEXT, None]})
    out = translator.translate_dataset(frame, "text", "es", ES, "en")
    assert out["es"].to_list() == [KEPT, None]
    detailed = translator.translate(TEXT, ES, "en", detailed=True)
    assert detailed.preservation == "placeholders"


def test_preserve_false_sends_the_text_as_is(translator: Translator) -> None:
    assert translator.translate(TEXT, ES, "en", preserve=False) == (
        TEXT.upper() + ":es"
    )
    model = translator.model
    assert isinstance(model, UpperModel)
    assert model.batches[-1] == [TEXT]
    translator.translate(TEXT, ES, "en")
    assert model.batches[-1] == ["visit ZXQ0 or email ZXQ1"]
    frame = pl.DataFrame({"text": [TEXT]})
    out = translator.translate_dataset(frame, "text", "es", ES, "en", preserve=False)
    assert out["es"].to_list() == [TEXT.upper() + ":es"]


class FailsOnTwo(UpperModel):
    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        if any("two" in text for text in texts):
            raise ValueError("boom")
        return super().predict_batch(texts, target_language, source_language)


def test_recorded_failures_return_the_original_text() -> None:
    translator = Translator(FailsOnTwo())
    texts = ["one https://a.io", "two https://b.io"]
    ok, failed = translator.translate_batch(
        texts, ES, "en", detailed=True, errors="record"
    )
    assert ok.text == "ONE https://a.io:es"
    assert failed.status is TranslationStatus.FAILED
    assert failed.text == "two https://b.io"
