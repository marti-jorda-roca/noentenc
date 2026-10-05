from collections.abc import Sequence


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
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{i}] must be a string, got {type(text).__name__}")


def is_blank(text: str) -> bool:
    return not text or text.isspace()
