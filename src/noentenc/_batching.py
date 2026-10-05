from collections.abc import Sequence


def check_batch_size(batch_size: object) -> None:
    """Reject a `batch_size` that isn't a positive int, before any model is loaded."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an int, got {type(batch_size).__name__}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")


def check_texts(texts: object) -> None:
    """Reject anything but a sequence of strings, e.g. a single string or a null."""
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise TypeError(
            f"texts must be a list of strings, got {type(texts).__name__}; "
            "pass a single text to the non-batch method"
        )
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{i}] must be a string, got {type(text).__name__}")


def is_blank(text: str) -> bool:
    return not text or text.isspace()
