"""Which texts carry no language at all: digits, emoji, punctuation, bare URLs and emails."""

import re

# A whitespace-separated token that is a URL, a bare domain or an email address, possibly
# wrapped in brackets or quotes and followed by punctuation. A bare domain needs a lowercase
# top-level label ("foo.com", not "Mr.Smith").
_LITERAL_TOKEN = re.compile(
    r"""[(<\["'«]*"""
    r"(?:"
    r"[A-Za-z][A-Za-z0-9+.-]*://\S+"
    r"|[Ww]{3}\.\S+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|[\w-]+(?:\.[\w-]+)*\.[a-z]{2,24}(?:[/?#:]\S*)?"
    r")"
    r"""[)>\]"'».,;:!?]*"""
)


def _token_letters(token: str) -> int:
    """Letters in `token`, or 0 when it is a URL, domain or email address."""
    letters = sum(map(str.isalpha, token))
    if letters and _LITERAL_TOKEN.fullmatch(token):
        return 0
    return letters


def has_linguistic_content(text: str) -> bool:
    """Whether `text` has a letter outside URLs, domains and email addresses.

    Stops at the first word with letters, so ordinary text costs one token check.
    """
    return any(_token_letters(token) for token in text.split())


def count_letters(text: str) -> int:
    """Letters in `text`, not counting those in URLs, domains and email addresses."""
    return sum(_token_letters(token) for token in text.split())
