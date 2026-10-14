"""Keep URLs, emails, code, paths, placeholders, tags and numbers byte-for-byte in translation.

Each literal is swapped for a placeholder (`ZXQ0`, `ZXQ1`...) that the models copy through,
the text is translated, and the placeholders are swapped back. Placeholders were chosen by
translating templates with every model: rare uppercase tokens like `ZXQ0` came back intact
in 100% of Opus-MT and SMaLL-100 outputs, and 93% of NLLB-200 and M2M100 ones, more than
`{0}`, `__0__`, `<x0>` or numbers did.

When a translation drops, repeats or alters a placeholder, the text is translated again
piece by piece instead: the prose between literals is translated on its own and the
literals are kept between the pieces, so they are preserved whatever the model does.
Fenced code blocks always go that way: a block alone on its lines would leave the model a
sentence that is nothing but a placeholder.

Each literal has a `LiteralKind`, and callers choose which kinds to keep
(`preserve=` on the translate methods). Callers can add their own literals
(`Translator(keep=...)`): terms such as brand names, and regular expressions. Their matches
are kept like the built-in ones; where matches overlap, the union of them is kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING

from noentenc.translation.models.base import Translation, TranslationStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# `Translation.preservation`: how the literals were kept.
PLACEHOLDERS = "placeholders"
SEGMENTS = "segments"


class LiteralKind(StrEnum):
    """A kind of text that translation keeps byte-for-byte."""

    # Fenced Markdown code blocks (``` or ~~~). They never reach the model: the prose
    # around them is translated on its own.
    CODE_BLOCK = "code_block"
    # Inline code in backticks.
    CODE = "code"
    # A Markdown link's target, with the brackets around it: `](https://…)`.
    LINK_TARGET = "link_target"
    # HTML/XML tags and character references.
    TAG = "tag"
    # Template placeholders: {{name}}, {name}, {0}, ${name}, %s, %(name)s.
    PLACEHOLDER = "placeholder"
    # URLs, and bare domains with a lowercase top-level label (acme.io/help).
    URL = "url"
    EMAIL = "email"
    MENTION = "mention"
    HASHTAG = "hashtag"
    # File paths: /usr/local/bin, ~/notes.txt, ./run.sh, C:\Users\ana.
    PATH = "path"
    # Hashes, keys, UUIDs and base64: 16+ characters mixing letters and digits.
    ENCODED = "encoded"
    # Code identifiers: snake_case, and camelCase or PascalCase with two or more humps
    # (getUserName, but not iPhone or JavaScript).
    IDENTIFIER = "identifier"
    # Numbers, with decimal, thousands, time and date separators.
    NUMBER = "number"


# Every kind: what `preserve=True` keeps.
ALL_KINDS: frozenset[LiteralKind] = frozenset(LiteralKind)


@dataclass(frozen=True)
class LiteralSpan:
    """A literal found in a text: what it is and where.

    `kind` is None for a match of the translator's `keep` terms. Where literals overlap
    they are merged into one span, with the kind of the one that starts first.
    """

    kind: LiteralKind | None
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Pattern:
    kind: LiteralKind
    regex: str
    # Drop sentence punctuation and unmatched closing brackets from the end.
    trim: bool = False


# One alternative per kind of literal, tried left to right at each position.
_PATTERNS: tuple[_Pattern, ...] = (
    # An opening fence at the start of a line, up to the same fence on a line of its own,
    # or the end of the text when it's never closed.
    _Pattern(
        LiteralKind.CODE_BLOCK,
        r"(?ms:^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*$.*?"
        r"(?:^[ \t]{0,3}(?P=fence)[ \t]*$|\Z))",
    ),
    _Pattern(LiteralKind.CODE, r"`[^`\n]+`"),
    _Pattern(LiteralKind.LINK_TARGET, r"\]\([^\s()]*(?:\([^\s()]*\)[^\s()]*)*\)"),
    _Pattern(LiteralKind.TAG, r"</?[A-Za-z][\w:.-]*(?:\s+[^<>]*?)?/?>"),
    _Pattern(LiteralKind.TAG, r"&(?:[A-Za-z][A-Za-z0-9]*|#\d+|#x[0-9A-Fa-f]+);"),
    _Pattern(LiteralKind.PLACEHOLDER, r"\{\{[^{}\n]*\}\}"),
    _Pattern(LiteralKind.PLACEHOLDER, r"\$\{[^{}\n]*\}"),
    _Pattern(LiteralKind.PLACEHOLDER, r"\{[A-Za-z_0-9][\w.:\-]*\}"),
    _Pattern(LiteralKind.PLACEHOLDER, r"%\(\w+\)[sdifr]|%[sdif]"),
    # Up to whitespace, quotes or angle brackets.
    _Pattern(
        LiteralKind.URL,
        r"(?:[A-Za-z][A-Za-z0-9+.-]*://|[Ww]{3}\.)[^\s<>\"'`]+",
        trim=True,
    ),
    _Pattern(LiteralKind.EMAIL, r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    _Pattern(LiteralKind.MENTION, r"(?<![\w@])@\w+"),
    _Pattern(LiteralKind.HASHTAG, r"(?<![\w&#])#\w+"),
    # Bare domains need a lowercase top-level label: "acme.io", not "Mr.Smith".
    _Pattern(
        LiteralKind.URL,
        r"(?<![\w.@])[\w-]+(?:\.[\w-]+)*\.[a-z]{2,24}(?:/[^\s<>\"'`]*)?(?![\w@])",
        trim=True,
    ),
    # Absolute paths need two segments, so "/s" or "and/or" aren't paths.
    _Pattern(
        LiteralKind.PATH,
        r"(?<![\w/.~\\-])(?:(?:~|\.\.?)(?:/[\w.@+-]+)+|(?:/[\w.@+-]+){2,})/?"
        r"|(?<![\w\\])[A-Za-z]:\\(?:[\w.@+-]+\\)*[\w.@+-]*",
        trim=True,
    ),
    _Pattern(
        LiteralKind.ENCODED,
        r"(?<![\w+/=-])(?:"
        r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}"
        r"|(?=[A-Za-z0-9+/]*\d)(?=[A-Za-z0-9+/]*[A-Za-z])[A-Za-z0-9+/]{16,}={0,2}"
        r")(?![\w+/=-])",
    ),
    _Pattern(
        LiteralKind.IDENTIFIER,
        r"(?<![\w$])(?:"
        r"[A-Za-z0-9]*[a-z0-9][A-Z][A-Za-z0-9]*[a-z0-9][A-Z][A-Za-z0-9]*"
        r"|_*[A-Za-z0-9]+(?:_+[A-Za-z0-9]+)+_*"
        r")(?![\w$])",
    ),
    _Pattern(LiteralKind.NUMBER, r"(?<![\w.,])\d+(?:[.,:/-]\d+)*(?!\w)"),
)
_CODE_BLOCKS = frozenset({LiteralKind.CODE_BLOCK})
_TRAILING = ".,;:!?'\""
_BRACKETS = {")": "(", "]": "[", "}": "{"}
# Placeholder stems, tried in order; a stem the text already contains is skipped.
_STEMS = ("ZXQ", "QZX")


def check_kinds(
    preserve: bool | Iterable[LiteralKind | str],
) -> frozenset[LiteralKind] | None:
    """The kinds `preserve` keeps: all for True, None (nothing, not even `keep`) for False."""
    if preserve is True:
        return ALL_KINDS
    if preserve is False:
        return None
    if isinstance(preserve, str):
        raise TypeError(
            f"preserve takes True, False or a list of literal kinds, got {preserve!r}; "
            "wrap it in a list"
        )
    kinds: set[LiteralKind] = set()
    for kind in preserve:
        try:
            kinds.add(LiteralKind(kind))
        except ValueError:
            raise ValueError(
                f"unknown literal kind {kind!r}; "
                f"expected one of {sorted(str(k) for k in LiteralKind)}"
            ) from None
    return frozenset(kinds)


@cache
def _literal_regex(
    kinds: frozenset[LiteralKind],
) -> tuple[re.Pattern[str], tuple[_Pattern, ...]] | None:
    """One regex trying the patterns of `kinds` in order, each in group `p<index>`."""
    patterns = tuple(p for p in _PATTERNS if p.kind in kinds)
    if not patterns:
        return None
    regex = "|".join(f"(?P<p{i}>{p.regex})" for i, p in enumerate(patterns))
    return re.compile(regex), patterns


def compile_keep(
    terms: Iterable[str | re.Pattern[str]],
) -> tuple[re.Pattern[str], ...]:
    """The patterns matching `terms`: strings as whole words, compiled patterns as given.

    A string matches exactly, case included, and not inside a longer word: `"Nike"`
    matches in "Nike's" but not in "Nikes". All strings share one pattern, longest
    first, so a text is scanned once however many there are.
    """
    if isinstance(terms, (str, re.Pattern)):
        raise TypeError(
            f"keep takes a list of strings and compiled patterns, got {terms!r}; "
            f"wrap it in a list"
        )
    words: set[str] = set()
    patterns: list[re.Pattern[str]] = []
    for term in terms:
        if isinstance(term, re.Pattern) and isinstance(term.pattern, str):
            patterns.append(term)
        elif isinstance(term, str):
            if not term.strip():
                raise ValueError("keep terms can't be empty or whitespace")
            words.add(term)
        else:
            raise TypeError(
                f"keep terms must be strings or re.compile() str patterns, got {term!r}"
            )
    if words:
        ordered = sorted(words, key=lambda word: (-len(word), word))
        patterns.insert(0, re.compile("|".join(map(_whole_word, ordered))))
    return tuple(patterns)


def _whole_word(term: str) -> str:
    """`term` escaped, not matching where a word character continues it."""
    before = r"(?<!\w)" if re.match(r"\w", term[0]) else ""
    after = r"(?!\w)" if re.match(r"\w", term[-1]) else ""
    return before + re.escape(term) + after


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


def find_literals(
    text: str,
    keep: tuple[re.Pattern[str], ...] = (),
    kinds: frozenset[LiteralKind] = ALL_KINDS,
) -> list[LiteralSpan]:
    """Every literal of `kinds` in `text`, and every match of `keep`, in order.

    Overlapping literals are merged into one.
    """
    spans: list[tuple[int, int, LiteralKind | None]] = []
    compiled = _literal_regex(kinds)
    if compiled is not None:
        regex, patterns = compiled
        for match in regex.finditer(text):
            pattern = patterns[int(match.lastgroup[1:])]  # ty: ignore[not-subscriptable]
            start, end = match.span()
            if pattern.trim:
                end = _trim(text, start, end)
            spans.append((start, end, pattern.kind))
    for keep_pattern in keep:
        spans += [
            (m.start(), m.end(), None)
            for m in keep_pattern.finditer(text)
            if m.end() > m.start()
        ]
    # A keep match at the same start as a built-in literal sorts after it.
    spans.sort(key=lambda span: (span[0], span[2] is None))
    merged: list[tuple[int, int, LiteralKind | None]] = []
    for start, end, kind in spans:
        if merged and start < merged[-1][1]:
            first, last, first_kind = merged[-1]
            merged[-1] = (first, max(last, end), first_kind)
        else:
            merged.append((start, end, kind))
    return [LiteralSpan(kind, a, b, text[a:b]) for a, b, kind in merged]


def mask(
    text: str,
    keep: tuple[re.Pattern[str], ...] = (),
    kinds: frozenset[LiteralKind] = ALL_KINDS,
) -> Masked | None:
    """`text` with its literals replaced, or None when it has none or no stem is free."""
    spans = [(span.start, span.end) for span in find_literals(text, keep, kinds)]
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


def split_at_literals(
    text: str,
    keep: tuple[re.Pattern[str], ...] = (),
    kinds: frozenset[LiteralKind] = ALL_KINDS,
) -> list[tuple[str, bool]]:
    """`text` as `(piece, is_literal)` pieces, literals and the prose between them."""
    pieces: list[tuple[str, bool]] = []
    position = 0
    for span in find_literals(text, keep, kinds):
        if span.start > position:
            pieces.append((text[position : span.start], False))
        pieces.append((span.text, True))
        position = span.end
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
    texts: list[str],
    translate: Callable[[list[str]], list[Translation]],
    keep: tuple[re.Pattern[str], ...] = (),
    kinds: frozenset[LiteralKind] = ALL_KINDS,
) -> list[Translation]:
    """`translate(texts)`, with the literals of `kinds`, and `keep`'s matches, kept.

    Fenced code blocks are cut out first, and the prose between them is translated on
    its own. Texts are translated with placeholders next. Those whose placeholders don't
    come back intact, or that have no free placeholder stem, are translated piece by
    piece.
    """
    if LiteralKind.CODE_BLOCK not in kinds:
        return _translate_masked(texts, translate, keep, kinds)
    inner = kinds - _CODE_BLOCKS
    blocks = [
        i for i, text in enumerate(texts) if find_literals(text, (), _CODE_BLOCKS)
    ]
    if not blocks:
        return _translate_masked(texts, translate, keep, inner)
    with_blocks = set(blocks)
    plain = [i for i in range(len(texts)) if i not in with_blocks]
    results: list[Translation | None] = [None] * len(texts)
    for i, result in zip(
        plain,
        _translate_masked([texts[i] for i in plain], translate, keep, inner),
        strict=True,
    ):
        results[i] = result
    for i, result in zip(
        blocks,
        _translate_pieces(
            [texts[i] for i in blocks],
            lambda prose: _translate_masked(prose, translate, keep, inner),
            lambda text: split_at_literals(text, (), _CODE_BLOCKS),
        ),
        strict=True,
    ):
        results[i] = result
    return results  # ty: ignore[invalid-return-type] - every slot is filled above


def _translate_masked(
    texts: list[str],
    translate: Callable[[list[str]], list[Translation]],
    keep: tuple[re.Pattern[str], ...],
    kinds: frozenset[LiteralKind],
) -> list[Translation]:
    """`translate_preserving` once code blocks are out of the way."""
    if not texts:
        return []
    masks = [mask(text, keep, kinds) for text in texts]
    # Texts with literals but no free placeholder stem go straight to piecewise.
    piecewise = [
        i
        for i, (text, masked) in enumerate(zip(texts, masks, strict=True))
        if masked is None and find_literals(text, keep, kinds)
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
        _translate_pieces(
            [texts[i] for i in piecewise],
            translate,
            lambda text: split_at_literals(text, keep, kinds),
        ),
        strict=True,
    ):
        results[i] = result
    return results  # ty: ignore[invalid-return-type] - every slot is filled above


def _model_input(text: str, masked: Masked | None) -> str:
    return text if masked is None else masked.text


def _translate_pieces(
    texts: list[str],
    translate: Callable[[list[str]], list[Translation]],
    split_text: Callable[[str], list[tuple[str, bool]]],
) -> list[Translation]:
    """Translate the prose between each text's literals, in one batch, and join it back.

    `split_text` cuts a text into `(piece, is_literal)` pieces.
    """
    if not texts:
        return []
    split = [split_text(text) for text in texts]
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
