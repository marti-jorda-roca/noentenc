"""Keep URLs, emails, code, placeholders, tags and numbers byte-for-byte through translation.

Each literal is swapped for a placeholder (`ZXQ0`, `ZXQ1`...) that the models copy through,
the text is translated, and the placeholders are swapped back. Placeholders were chosen by
translating templates with every model: rare uppercase tokens like `ZXQ0` came back intact
in 100% of Opus-MT and SMaLL-100 outputs, and 93% of NLLB-200 and M2M100 ones, more than
`{0}`, `__0__`, `<x0>` or numbers did.

When a translation drops, repeats or alters a placeholder, the text is translated again
piece by piece instead: the prose between literals is translated on its own and the
literals are kept between the pieces, so they are preserved whatever the model does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from noentenc.translation.models.base import Translation, TranslationStatus

if TYPE_CHECKING:
    from collections.abc import Callable

# `Translation.preservation`: how the literals were kept.
PLACEHOLDERS = "placeholders"
SEGMENTS = "segments"

# One alternative per kind of literal, tried left to right at each position.
_LITERAL = re.compile(
    "|".join(
        (
            # Inline code, and a Markdown link's target with the brackets around it.
            r"`[^`\n]+`",
            r"\]\([^\s()]*(?:\([^\s()]*\)[^\s()]*)*\)",
            # HTML/XML tags and character references.
            r"</?[A-Za-z][\w:.-]*(?:\s+[^<>]*?)?/?>",
            r"&(?:[A-Za-z][A-Za-z0-9]*|#\d+|#x[0-9A-Fa-f]+);",
            # Template placeholders: {{name}}, {name}, {0}, ${name}, %s, %(name)s.
            r"\{\{[^{}\n]*\}\}",
            r"\$\{[^{}\n]*\}",
            r"\{[A-Za-z_0-9][\w.:\-]*\}",
            r"%\(\w+\)[sdifr]|%[sdif]",
            # URLs, up to whitespace, quotes or angle brackets. Trailing punctuation and
            # an unmatched closing bracket are trimmed afterwards.
            r"(?P<url>(?:[A-Za-z][A-Za-z0-9+.-]*://|[Ww]{3}\.)[^\s<>\"'`]+)",
            # Email addresses, mentions and hashtags.
            r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
            r"(?<![\w@])@\w+",
            r"(?<![\w&#])#\w+",
            # Bare domains need a lowercase top-level label: "acme.io", not "Mr.Smith".
            r"(?P<domain>(?<![\w.@])[\w-]+(?:\.[\w-]+)*\.[a-z]{2,24}"
            r"(?:/[^\s<>\"'`]*)?(?![\w@]))",
            # Numbers, with decimal, thousands, time and date separators.
            r"(?<![\w.,])\d+(?:[.,:/-]\d+)*(?!\w)",
        )
    )
)
_TRAILING = ".,;:!?'\""
_BRACKETS = {")": "(", "]": "[", "}": "{"}
# Placeholder stems, tried in order; a stem the text already contains is skipped.
_STEMS = ("ZXQ", "QZX")


@dataclass(frozen=True)
class Masked:
    """A text with its literals swapped for placeholders.

    A placeholder that would touch a letter or digit (`<b>bold</b>`) gets a space on that
    side, so the word around it still reads as a word; `restore` removes the space.
    """

    text: str
    literals: tuple[str, ...]
    stem: str
    # Whether placeholder i got a space before it, and after it.
    padded: tuple[tuple[bool, bool], ...]

    def restore(self, translated: str) -> str | None:
        """`translated` with the literals back, or None if a placeholder went missing.

        Every placeholder must appear exactly once, unaltered, and no other.
        """
        found = re.findall(rf"{self.stem}(\d+)", translated)
        if sorted(found) != sorted(str(i) for i in range(len(self.literals))):
            return None
        for i, literal in enumerate(self.literals):
            before, after = self.padded[i]
            pattern = (
                (" ?" if before else "")
                + rf"{self.stem}{i}(?!\d)"
                + (" ?" if after else "")
            )
            translated = re.sub(
                pattern, lambda _, value=literal: value, translated, count=1
            )
        return translated


def find_literals(text: str) -> list[tuple[int, int]]:
    """`(start, end)` of every literal in `text`, in order."""
    spans: list[tuple[int, int]] = []
    for match in _LITERAL.finditer(text):
        start, end = match.span()
        if match.lastgroup in ("url", "domain"):
            end = _trim(text, start, end)
        spans.append((start, end))
    return spans


def mask(text: str) -> Masked | None:
    """`text` with its literals replaced, or None when it has none or no stem is free."""
    spans = find_literals(text)
    if not spans:
        return None
    stem = next((s for s in _STEMS if s.lower() not in text.lower()), None)
    if stem is None:
        return None
    parts: list[str] = []
    padded: list[tuple[bool, bool]] = []
    position = 0
    for i, (start, end) in enumerate(spans):
        before = start > 0 and text[start - 1].isalnum()
        after = end < len(text) and text[end].isalnum()
        padded.append((before, after))
        token = (" " if before else "") + f"{stem}{i}" + (" " if after else "")
        parts += [text[position:start], token]
        position = end
    parts.append(text[position:])
    return Masked(
        "".join(parts), tuple(text[a:b] for a, b in spans), stem, tuple(padded)
    )


def split_at_literals(text: str) -> list[tuple[str, bool]]:
    """`text` as `(piece, is_literal)` pieces, literals and the prose between them."""
    pieces: list[tuple[str, bool]] = []
    position = 0
    for start, end in find_literals(text):
        if start > position:
            pieces.append((text[position:start], False))
        pieces.append((text[start:end], True))
        position = end
    if position < len(text):
        pieces.append((text[position:], False))
    return pieces


def _trim(text: str, start: int, end: int) -> int:
    """Drop sentence punctuation and unmatched closing brackets from a URL's end."""
    while end > start:
        last = text[end - 1]
        if (
            last in _TRAILING
            or last in _BRACKETS
            and text.count(_BRACKETS[last], start, end) < text.count(last, start, end)
        ):
            end -= 1
        else:
            break
    return end


def translate_preserving(
    texts: list[str], translate: Callable[[list[str]], list[Translation]]
) -> list[Translation]:
    """`translate(texts)`, with every literal in the texts kept byte-for-byte.

    Texts are translated with placeholders first. Those whose placeholders don't come
    back intact, or that have no free placeholder stem, are translated piece by piece.
    """
    masks = [mask(text) for text in texts]
    # Texts with literals but no free placeholder stem go straight to piecewise.
    piecewise = [
        i
        for i, (text, masked) in enumerate(zip(texts, masks, strict=True))
        if masked is None and find_literals(text)
    ]
    skipped = set(piecewise)
    first = [i for i in range(len(texts)) if i not in skipped]
    results: list[Translation | None] = [None] * len(texts)
    translated = translate([_model_input(texts[i], masks[i]) for i in first])
    for i, result in zip(first, translated, strict=True):
        masked = masks[i]
        if masked is None:
            results[i] = result
            continue
        if result.status is TranslationStatus.FAILED:
            # A failed text comes back as given: the original, not the masked one.
            results[i] = replace(result, text=texts[i])
            continue
        restored = masked.restore(result.text)
        if restored is None:
            piecewise.append(i)
        else:
            results[i] = replace(result, text=restored, preservation=PLACEHOLDERS)
    for i, result in zip(
        piecewise,
        _translate_pieces([texts[i] for i in piecewise], translate),
        strict=True,
    ):
        results[i] = result
    return results  # ty: ignore[invalid-return-type] - every slot is filled above


def _model_input(text: str, masked: Masked | None) -> str:
    return text if masked is None else masked.text


def _translate_pieces(
    texts: list[str], translate: Callable[[list[str]], list[Translation]]
) -> list[Translation]:
    """Translate the prose between each text's literals, in one batch, and join it back."""
    if not texts:
        return []
    split = [split_at_literals(text) for text in texts]
    # (text, piece) of every piece to translate, with its surrounding whitespace removed.
    jobs = [
        (t, p)
        for t, pieces in enumerate(split)
        for p, (piece, literal) in enumerate(pieces)
        if not literal and any(char.isalpha() for char in piece)
    ]
    outputs = translate([split[t][p][0].strip() for t, p in jobs])
    by_piece = dict(zip(jobs, outputs, strict=True))
    results: list[Translation] = []
    for t, pieces in enumerate(split):
        parts: list[str] = []
        done = [by_piece[t, p] for p in range(len(pieces)) if (t, p) in by_piece]
        failed = next((d for d in done if d.status is TranslationStatus.FAILED), None)
        if failed is not None:
            results.append(replace(failed, text=texts[t]))
            continue
        for p, (piece, _literal) in enumerate(pieces):
            if (t, p) in by_piece:
                stripped = piece.strip()
                lead = piece[: len(piece) - len(piece.lstrip())]
                trail = piece[len(lead) + len(stripped) :]
                parts += [lead, by_piece[t, p].text, trail]
            else:
                parts.append(piece)
        results.append(
            Translation(
                "".join(parts),
                input_truncated=any(d.input_truncated for d in done),
                output_limit_reached=any(d.output_limit_reached for d in done),
                preservation=SEGMENTS,
            )
        )
    return results
