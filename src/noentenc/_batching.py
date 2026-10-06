from collections.abc import Iterable, Iterator, Sequence
from itertools import islice

# Texts the streaming methods read and process together, unless told otherwise.
DEFAULT_CHUNK_SIZE = 1024


def check_int(
    name: str, value: object, *, optional: bool = False, minimum: int = 1
) -> int | None:
    """`value` if it is an int of at least `minimum` (or None when `optional`), else raise."""
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        kind = "an int or None" if optional else "an int"
        raise TypeError(f"{name} must be {kind}, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def check_batch_size(batch_size: object) -> None:
    """Reject a `batch_size` that isn't a positive int, before any model is loaded."""
    check_int("batch_size", batch_size)


def check_texts(texts: object) -> None:
    """Reject anything but a sequence of strings, e.g. a single string or a null."""
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise TypeError(
            f"texts must be a list of strings, got {type(texts).__name__}; "
            "pass a single text to the non-batch method"
        )
    _check_strings(texts)


def _check_strings(texts: Sequence[object], start: int = 0) -> None:
    """Reject a text that isn't a string, numbering `texts` from `start`."""
    for i, text in enumerate(texts, start):
        if not isinstance(text, str):
            raise TypeError(f"texts[{i}] must be a string, got {type(text).__name__}")


def stream_chunks(texts: Iterable[str], chunk_size: int) -> Iterator[list[str]]:
    """`texts` in lists of up to `chunk_size`, each read from `texts` only when needed.

    Rejects a single string, a non-iterable and a bad `chunk_size` at once. A text that
    isn't a string raises when its chunk is read.
    """
    check_int("chunk_size", chunk_size)
    if isinstance(texts, str | bytes) or not isinstance(texts, Iterable):
        raise TypeError(
            f"texts must be an iterable of strings, got {type(texts).__name__}; "
            "pass a single text to the non-batch method"
        )
    return _chunks(iter(texts), chunk_size)


def _chunks(texts: Iterator[str], size: int) -> Iterator[list[str]]:
    start = 0
    while chunk := list(islice(texts, size)):
        _check_strings(chunk, start)
        start += len(chunk)
        yield chunk


def is_blank(text: str) -> bool:
    return not text or text.isspace()
