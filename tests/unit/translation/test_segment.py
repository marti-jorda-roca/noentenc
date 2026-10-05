import pytest

from noentenc.translation._segment import split_sentences


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
