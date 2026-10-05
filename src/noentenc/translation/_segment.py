import re
from dataclasses import dataclass
from itertools import pairwise

# Characters that end a sentence, and closing quotes/brackets that may follow them.
_SENTENCE_END = frozenset(".!?…。！？।؟۔")
_CLOSERS = frozenset("\"'”’»)]）」』")
_FULL_WIDTH_END = frozenset("。！？”’）」』")
# Words whose full stop is usually not the end of a sentence (titles before a name).
_ABBREVIATIONS = frozenset({"Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Sra", "Jr", "St"})
# Whitespace runs, plus the gap after full-width punctuation, which needs no space:
# "你好。再见。" is two sentences.
_GAP = re.compile(
    r"\s+"
    r"|(?<=[。！？])(?![\s。！？”’）」』])"
    r"|(?<=[。！？][”’）」』])(?![\s”’）」』])"
)


@dataclass(frozen=True)
class Segments:
    """A text cut into sentences, keeping the whitespace between them.

    `separators` has one more item than `sentences`: the text is `separators[0]`,
    `sentences[0]`, `separators[1]`, ..., `sentences[-1]`, `separators[-1]`.
    """

    sentences: list[str]
    separators: list[str]

    def join(self, sentences: list[str]) -> str:
        """Put `sentences` (e.g. translations of `self.sentences`) back in place."""
        parts = [self.separators[0]]
        inner = len(sentences) - 1
        for i, (sentence, separator) in enumerate(
            zip(sentences, self.separators[1:], strict=True)
        ):
            # Sentences after full-width punctuation have no separator; a translation
            # into a language that isn't written that way needs a space between them.
            if i < inner and not separator and sentence[-1:] not in _FULL_WIDTH_END:
                separator = " "
            parts += [sentence, separator]
        return "".join(parts)


def split_sentences(text: str) -> Segments:
    """Split `text` at line breaks and after sentence-ending punctuation.

    A full stop doesn't end a sentence before a lowercase letter ("e.g. this"), or
    after an initial or a title ("J. Smith", "Dr. Smith").
    """
    # (start, end) of each separator; the first and last may be empty.
    cuts = [(0, 0)]
    for gap in _GAP.finditer(text):
        if gap.start() == 0:
            cuts[0] = gap.span()
        elif gap.end() == len(text) or _ends_sentence(text, gap):
            cuts.append(gap.span())
    if cuts[-1][1] != len(text):
        cuts.append((len(text), len(text)))
    return Segments(
        sentences=[text[end:start] for (_, end), (start, _) in pairwise(cuts)],
        separators=[text[start:end] for start, end in cuts],
    )


def _ends_sentence(text: str, gap: re.Match[str]) -> bool:
    whitespace = gap.group()
    # Empty gaps only match after full-width punctuation.
    if not whitespace or "\n" in whitespace:
        return True
    end = gap.start() - 1
    while end > 0 and text[end] in _CLOSERS:
        end -= 1
    if text[end] not in _SENTENCE_END or text[gap.end()].islower():
        return False
    word = text[text.rfind(" ", 0, end) + 1 : end]
    return text[end] != "." or not (len(word) == 1 or word in _ABBREVIATIONS)
