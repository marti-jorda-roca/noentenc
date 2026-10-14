import re

import pytest

from noentenc.translation._segment import split_sentences, split_to_fit


@pytest.mark.parametrize(
    ("text", "sentences", "separators"),
    [
        ("", [], [""]),
        (" \n ", [], [" \n "]),
        ("Hi.", ["Hi."], ["", ""]),
        (" Hi there.  ", ["Hi there."], [" ", "  "]),
        (
            "The cat sat. It was sunny!\n\nThe next day? Rain.",
            ["The cat sat.", "It was sunny!", "The next day?", "Rain."],
            ["", " ", "\n\n", " ", ""],
        ),
        ("one line\r\nanother", ["one line", "another"], ["", "\r\n", ""]),
        (
            'He said "Stop." Then left.',
            ['He said "Stop."', "Then left."],
            ["", " ", ""],
        ),
        (
            "Wait... what? Yes, e.g. this.",
            ["Wait... what?", "Yes, e.g. this."],
            ["", " ", ""],
        ),
        (
            "Dr. Smith met J. Doe. Then.",
            ["Dr. Smith met J. Doe.", "Then."],
            ["", " ", ""],
        ),
        ("Version 2.5 is out. Go", ["Version 2.5 is out.", "Go"], ["", " ", ""]),
        ("你好。再见！", ["你好。", "再见！"], ["", "", ""]),
        ("他说：“你好。”他走了", ["他说：“你好。”", "他走了"], ["", "", ""]),
        ("नमस्ते। आप कैसे हैं?", ["नमस्ते।", "आप कैसे हैं?"], ["", " ", ""]),
    ],
)
def test_split_sentences(
    text: str, sentences: list[str], separators: list[str]
) -> None:
    segments = split_sentences(text)
    assert segments.sentences == sentences
    assert segments.separators == separators
    assert segments.join(segments.sentences) == text


def test_join_spaces_translations_of_unspaced_sentences() -> None:
    segments = split_sentences("你好。再见。\n谢谢。")
    assert segments.join(["Hello.", "Bye.", "Thanks."]) == "Hello. Bye.\nThanks."
    assert segments.join(["你好。", "再见。", "谢谢。"]) == "你好。再见。\n谢谢。"


def word_offsets(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in re.finditer(r"\S+", text)]


@pytest.mark.parametrize(
    ("text", "window", "pieces"),
    [
        ("a b c", 3, [("a b c", "")]),
        ("a b c d e f g h", 3, [("a b c", " "), ("d e f", " "), ("g h", "")]),
        # A clause end wins over the window's last space...
        (
            "one two, three four five six; seven eight",
            4,
            [("one two,", " "), ("three four five six;", " "), ("seven eight", "")],
        ),
        # ...unless it would leave less than a third of the window.
        (
            "one, two three four five six",
            5,
            [("one, two three four five", " "), ("six", "")],
        ),
        # Line breaks and runs of spaces are kept as the separator.
        ("x  y\n z", 1, [("x", "  "), ("y", "\n "), ("z", "")]),
    ],
)
def test_split_to_fit(text: str, window: int, pieces: list[tuple[str, str]]) -> None:
    assert split_to_fit(text, word_offsets(text), window) == pieces
    assert "".join(piece + gap for piece, gap in pieces) == text


def test_split_to_fit_cuts_at_tokens_without_whitespace() -> None:
    text = "abcdefghij"
    offsets = [(i, i + 1) for i in range(len(text))]
    assert split_to_fit(text, offsets, 4) == [("abcd", ""), ("efgh", ""), ("ij", "")]
    with pytest.raises(ValueError, match="window"):
        split_to_fit(text, offsets, 0)
