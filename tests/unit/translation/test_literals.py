"""URLs, emails, code, placeholders, tags and numbers survive translation byte-for-byte."""

import re
from collections.abc import Iterator

import polars as pl
import pytest

from noentenc import Language
from noentenc.translation import (
    LiteralKind,
    LiteralSpan,
    Translation,
    TranslationStatus,
    Translator,
)
from noentenc.translation._literals import (
    check_kinds,
    compile_keep,
    find_literals,
    mask,
    split_at_literals,
    translate_preserving,
)
from tests.unit.translation.test_translator import UpperModel

ES = Language.SPANISH


def literals(text: str, keep: tuple[re.Pattern[str], ...] = ()) -> list[str]:
    return [span.text for span in find_literals(text, keep)]


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Whole words only, case included.
        ("Nike and Nike's shoes", ["Nike", "Nike"]),
        ("Nikes, nike, NIKE, PreNike", []),
        # The longest term wins where terms overlap.
        ("New Nike Air Max today", ["Nike Air Max"]),
        # Terms starting or ending in punctuation still match.
        ("Built with C++ and .NET.", ["C++", ".NET"]),
    ],
)
def test_keep_terms(text: str, expected: list[str]) -> None:
    keep = compile_keep(["Nike", "Nike Air Max", "C++", ".NET"])
    assert literals(text, keep) == expected


def test_keep_patterns_match_as_given() -> None:
    keep = compile_keep([re.compile(r"sku-\d+", re.IGNORECASE), re.compile(r"x*")])
    # Empty matches are ignored.
    assert literals("Order SKU-12 and sku-7", keep) == ["SKU-12", "sku-7"]


def test_keep_merges_overlaps_with_built_in_literals() -> None:
    keep = compile_keep(["Nike Air 90", "acme"])
    # "90" is a number literal inside the term; "acme" is inside a URL.
    assert literals("Buy Nike Air 90 at https://acme.io/shop", keep) == [
        "Nike Air 90",
        "https://acme.io/shop",
    ]
    masked = mask("Buy Nike Air 90 at https://acme.io/shop", keep)
    assert masked is not None
    assert masked.text == "Buy ZXQ0 at ZXQ1"


@pytest.mark.parametrize(
    ("keep", "error"),
    [
        ("Nike", TypeError),
        (re.compile("Nike"), TypeError),
        ([""], ValueError),
        (["  "], ValueError),
        ([42], TypeError),
        ([re.compile(b"Nike")], TypeError),
    ],
)
def test_compile_keep_rejects_bad_terms(keep: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        compile_keep(keep)  # ty: ignore[invalid-argument-type]


def test_keep_terms_survive_both_paths() -> None:
    keep = compile_keep(["Nike"])
    (kept,) = translate_preserving(["i love nike and Nike"], upper, keep)
    assert kept.text == "I LOVE NIKE AND Nike"
    assert kept.preservation == "placeholders"

    def drops_placeholders(texts: list[str]) -> list[Translation]:
        return [Translation(re.sub(r"ZXQ\d+", "", text).upper()) for text in texts]

    (pieces,) = translate_preserving(["buy Nike now"], drops_placeholders, keep)
    assert pieces.text == "BUY Nike NOW"
    assert pieces.preservation == "segments"


def test_translator_keep() -> None:
    translator = Translator(UpperModel(), keep=["Nike", re.compile(r"SKU-\d+")])
    assert translator.translate("buy Nike SKU-42 at acme.io", ES, "en") == (
        "BUY Nike SKU-42 AT acme.io:es"
    )
    assert translator.translate_batch(["nike", "Nike"], ES, "en") == [
        "NIKE:es",
        "Nike:es",
    ]
    # preserve=False sends the terms to the model too.
    assert translator.translate("Nike", ES, "en", preserve=False) == "NIKE:es"


def test_find_literals_reports_kinds_and_positions() -> None:
    text = "Ping @ana at 14:30 about https://a.io/x."
    assert find_literals(text) == [
        LiteralSpan(LiteralKind.MENTION, 5, 9, "@ana"),
        LiteralSpan(LiteralKind.NUMBER, 13, 18, "14:30"),
        LiteralSpan(LiteralKind.URL, 25, 39, "https://a.io/x"),
    ]


def test_kinds_select_the_literals_found() -> None:
    text = "Mix 200 g at 14:30, see https://a.io or #recipes"
    no_numbers = frozenset(LiteralKind) - {LiteralKind.NUMBER}
    assert [s.text for s in find_literals(text, kinds=no_numbers)] == [
        "https://a.io",
        "#recipes",
    ]
    assert find_literals(text, kinds=frozenset()) == []
    # Without the number pattern, nothing else claims the digits.
    assert [s.text for s in find_literals("v 1.2.3", kinds=no_numbers)] == []


def test_keep_matches_have_no_kind_and_merge_into_the_first_literal() -> None:
    keep = compile_keep(["Nike Air 90", "acme"])
    spans = find_literals("Buy Nike Air 90 at https://acme.io/shop", keep)
    assert [(s.kind, s.text) for s in spans] == [
        (None, "Nike Air 90"),
        (LiteralKind.URL, "https://acme.io/shop"),
    ]


@pytest.mark.parametrize(
    ("preserve", "expected"),
    [
        (True, frozenset(LiteralKind)),
        (False, None),
        ([], frozenset()),
        (["url", LiteralKind.EMAIL], {LiteralKind.URL, LiteralKind.EMAIL}),
    ],
)
def test_check_kinds(preserve: object, expected: object) -> None:
    assert check_kinds(preserve) == expected  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("preserve", "error"),
    [("url", TypeError), (["urls"], ValueError), ([None], ValueError)],
)
def test_check_kinds_rejects_bad_values(
    preserve: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        check_kinds(preserve)  # ty: ignore[invalid-argument-type]


def test_preserve_kinds_keep_only_those_kinds(translator: Translator) -> None:
    text = "see #deals at https://a.io"
    assert translator.translate(text, ES, "en", preserve=["url"]) == (
        "SEE #DEALS AT https://a.io:es"
    )
    assert translator.translate_batch([text], ES, "en", preserve=["hashtag"]) == [
        "SEE #deals AT HTTPS://A.IO:es"
    ]
    assert list(translator.translate_stream([text], ES, "en", preserve=[])) == [
        text.upper() + ":es"
    ]
    frame = pl.DataFrame({"text": [text]})
    out = translator.translate_dataset(frame, "text", "es", ES, "en", preserve=["url"])
    assert out["es"].to_list() == ["SEE #DEALS AT https://a.io:es"]


def test_empty_preserve_still_keeps_keep_terms() -> None:
    translator = Translator(UpperModel(), keep=["Nike"])
    assert translator.translate("buy Nike at acme.io", ES, "en", preserve=[]) == (
        "BUY Nike AT ACME.IO:es"
    )


def test_bad_preserve_fails_before_reading_or_translating(
    translator: Translator,
) -> None:
    def never_read() -> Iterator[str]:
        raise AssertionError("read")
        yield ""

    with pytest.raises(ValueError, match="unknown literal kind"):
        translator.translate_stream(never_read(), ES, "en", preserve=["links"])
    with pytest.raises(TypeError, match="wrap it in a list"):
        translator.translate("hola", ES, "en", preserve="url")
    model = translator.model
    assert isinstance(model, UpperModel)
    assert model.batches == []


def test_translator_find_literals() -> None:
    translator = Translator(UpperModel(), keep=["Nike"])
    text = "Nike costs 90 at acme.io"
    assert [(s.kind, s.text) for s in translator.find_literals(text)] == [
        (None, "Nike"),
        (LiteralKind.NUMBER, "90"),
        (LiteralKind.URL, "acme.io"),
    ]
    assert [s.text for s in translator.find_literals(text, ["url"])] == [
        "Nike",
        "acme.io",
    ]
    assert translator.find_literals(text, preserve=False) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Fix:\n```python\nprint('hi')\n```\nThanks",
            [(LiteralKind.CODE_BLOCK, "```python\nprint('hi')\n```")],
        ),
        # Never closed: the block runs to the end of the text.
        ("Run:\n~~~\nmake all", [(LiteralKind.CODE_BLOCK, "~~~\nmake all")]),
        # A fence must open a line; mid-line backticks stay inline code.
        ("Inline ```x``` here", [(LiteralKind.CODE, "`x`")]),
        (
            "Edit /etc/app/config.yaml, ~/notes.txt, ./run.sh or C:\\Users\\ana\\a.txt.",
            [
                (LiteralKind.PATH, "/etc/app/config.yaml"),
                (LiteralKind.PATH, "~/notes.txt"),
                (LiteralKind.PATH, "./run.sh"),
                (LiteralKind.PATH, "C:\\Users\\ana\\a.txt"),
            ],
        ),
        # Not paths: words around a slash, one-segment absolute paths.
        ("and/or km/h TCP/IP /s", []),
        (
            "Call getUserName with user_id on x86_64",
            [
                (LiteralKind.IDENTIFIER, "getUserName"),
                (LiteralKind.IDENTIFIER, "user_id"),
                (LiteralKind.IDENTIFIER, "x86_64"),
            ],
        ),
        # Not identifiers: names with one hump, Markdown emphasis.
        ("iPhone JavaScript PhD McDonald __bold__ _it_", []),
        (
            "sha 964670f15c6b47f9b8f9340732720473 id "
            "964670f1-5c6b-47f9-b8f9-340732720473 b64 aGVsbG8gd29ybGQgaG93",
            [
                (LiteralKind.ENCODED, "964670f15c6b47f9b8f9340732720473"),
                (LiteralKind.ENCODED, "964670f1-5c6b-47f9-b8f9-340732720473"),
                (LiteralKind.ENCODED, "aGVsbG8gd29ybGQgaG93"),
            ],
        ),
        # Not encoded: long words without digits, hyphenated words with a number.
        ("antidisestablishmentarianism COVID-19-related", [(LiteralKind.NUMBER, "19")]),
    ],
)
def test_find_code_paths_identifiers_and_encoded(
    text: str, expected: list[tuple[LiteralKind, str]]
) -> None:
    assert [(span.kind, span.text) for span in find_literals(text)] == expected


BLOCK_TEXT = "fix it:\n\n```\nprint(x)  # all names\n```\n\nthen run it."


def test_code_blocks_never_reach_the_model() -> None:
    seen: list[str] = []

    def record(texts: list[str]) -> list[Translation]:
        seen.extend(texts)
        return upper(texts)

    (result,) = translate_preserving([BLOCK_TEXT], record)
    assert result.text == "FIX IT:\n\n```\nprint(x)  # all names\n```\n\nTHEN RUN IT."
    assert result.preservation == "segments"
    assert seen == ["fix it:", "then run it."]


def test_code_blocks_mixed_with_other_literals_and_plain_texts() -> None:
    texts = [BLOCK_TEXT + " see https://a.io", "plain text", "```\nonly code\n```"]
    results = translate_preserving(texts, upper)
    assert [r.text for r in results] == [
        "FIX IT:\n\n```\nprint(x)  # all names\n```\n\nTHEN RUN IT. SEE https://a.io",
        "PLAIN TEXT",
        "```\nonly code\n```",
    ]
    assert [r.preservation for r in results] == ["segments", None, "segments"]


def test_code_blocks_go_to_the_model_when_not_preserved() -> None:
    kinds = frozenset(LiteralKind) - {LiteralKind.CODE_BLOCK}
    (result,) = translate_preserving([BLOCK_TEXT], upper, kinds=kinds)
    assert result.text == BLOCK_TEXT.upper()


def test_a_failed_prose_piece_fails_the_text_with_code_blocks() -> None:
    def fails_on_run(texts: list[str]) -> list[Translation]:
        return [
            Translation(t, status=TranslationStatus.FAILED, error="ValueError: boom")
            if "run" in t
            else Translation(t.upper())
            for t in texts
        ]

    (result,) = translate_preserving([BLOCK_TEXT], fails_on_run)
    assert result.status is TranslationStatus.FAILED
    assert result.text == BLOCK_TEXT
