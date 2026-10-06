"""Helpers shared by the detection and translation tests."""

from collections.abc import Iterator


class CountingTexts:
    """Texts read one at a time, like lines of a file, counting how many were read."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.read = 0

    def __iter__(self) -> Iterator[str]:
        for text in self.texts:
            self.read += 1
            yield text
