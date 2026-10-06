"""Draw the README's benchmark chart, in light and dark versions.

The chart is pixel art in the style of the logo: a 5x7 bitmap font, bars with a
highlight, a shade and an outline, and the logo's monster in the corner. The numbers
come from docs/benchmarks.md, "Against the original implementations". Update ROWS when
those change, then run:

    uv run python scripts/make_benchmark_chart.py
"""

import re
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

# (title, subtitle, noentenc value, its label, original value, its label, ratio, verdict)
ROWS = [
    (
        "INSTALL SIZE",
        "VS TORCH + TRANSFORMERS",
        140,
        "140 MB",
        770,
        "770 MB",
        "5.5×",
        "SMALLER",
    ),
    (
        "START-UP TIME",
        "OPUS-MT VS TRANSFORMERS",
        0.69,
        "0.69 S",
        2.8,
        "2.8 S",
        "4×",
        "FASTER",
    ),
    (
        "TRANSLATION SPEED",
        "OPUS-MT VS TRANSFORMERS",
        122,
        "122/S",
        66,
        "66/S",
        "1.9×",
        "FASTER",
    ),
    (
        "DETECTION SPEED",
        "LANGID VS LANGID.PY",
        745_000,
        "745K/S",
        1_600,
        "1.6K/S",
        "470×",
        "FASTER",
    ),
]

THEMES = {
    "light": {
        "surface": "#f2ede4",
        "border": "#1b1b24",
        "grid": "#c9c2b4",
        "ink": "#1b1b24",
        "muted": "#6b6577",
        "shadow": None,
    },
    "dark": {
        "surface": "#1b1b24",
        "border": "#34343f",
        "grid": "#34343f",
        "ink": "#f2ede4",
        "muted": "#9a97a6",
        "shadow": "#0d0d12",
    },
}

OUTLINE = "#1f0c10"
# (highlight, fill, shade): the monster's red, and its teeth.
OURS = ("#ff6b57", "#e3352b", "#a51d1f")
THEIRS = ("#ffffff", "#c9ced8", "#8b93a1")

# 5x7 glyphs, one string per row; "#" is a lit pixel.
GLYPHS = {
    "A": ".###. #...# #...# ##### #...# #...# #...#",
    "B": "####. #...# #...# ####. #...# #...# ####.",
    "C": ".###. #...# #.... #.... #.... #...# .###.",
    "D": "####. #...# #...# #...# #...# #...# ####.",
    "E": "##### #.... #.... ####. #.... #.... #####",
    "F": "##### #.... #.... ####. #.... #.... #....",
    "G": ".###. #...# #.... #.### #...# #...# .####",
    "H": "#...# #...# #...# ##### #...# #...# #...#",
    "I": "### .#. .#. .#. .#. .#. ###",
    "J": "..### ...#. ...#. ...#. #..#. #..#. .##..",
    "K": "#...# #..#. #.#.. ##... #.#.. #..#. #...#",
    "L": "#.... #.... #.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #.#.# #...# #...# #...#",
    "N": "#...# #...# ##..# #.#.# #..## #...# #...#",
    "O": ".###. #...# #...# #...# #...# #...# .###.",
    "P": "####. #...# #...# ####. #.... #.... #....",
    "Q": ".###. #...# #...# #...# #.#.# #..#. .##.#",
    "R": "####. #...# #...# ####. #.#.. #..#. #...#",
    "S": ".#### #.... #.... .###. ....# ....# ####.",
    "T": "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# #...# #...# .###.",
    "V": "#...# #...# #...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #...# #.#.# #.#.# #.#.# .#.#.",
    "X": "#...# #...# .#.#. ..#.. .#.#. #...# #...#",
    "Y": "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..",
    "Z": "##### ....# ...#. ..#.. .#... #.... #####",
    "0": ".###. #...# #..## #.#.# ##..# #...# .###.",
    "1": "..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.",
    "2": ".###. #...# ....# ...#. ..#.. .#... #####",
    "3": "##### ...#. ..#.. ...#. ....# #...# .###.",
    "4": "...#. ..##. .#.#. #..#. ##### ...#. ...#.",
    "5": "##### #.... ####. ....# ....# #...# .###.",
    "6": "..##. .#... #.... ####. #...# #...# .###.",
    "7": "##### ....# ...#. ..#.. .#... .#... .#...",
    "8": ".###. #...# #...# .###. #...# #...# .###.",
    "9": ".###. #...# #...# .#### ....# ...#. .##..",
    ".": ".. .. .. .. .. ## ##",
    ",": ".. .. .. .. .# .# #.",
    "'": "# # . . . . .",
    "/": "....# ....# ...#. ..#.. .#... #.... #....",
    "+": "..... ..#.. ..#.. ##### ..#.. ..#.. .....",
    "-": ".... .... .... #### .... .... ....",
    "×": "..... #...# .#.#. ..#.. .#.#. #...# .....",
    " ": "... ... ... ... ... ... ...",
}
GLYPHS = {char: rows.split() for char, rows in GLYPHS.items()}

WIDTH = 960
PAD = 32
U = 2  # one pixel of the art, in SVG units
HEADER = 118
ROW_HEIGHT = 84
FOOTER = 8
BAR_X = 352
BAR_MAX = 300
BAR_HEIGHT = 9 * U


def text_width(content: str, size: int) -> int:
    return (sum(len(GLYPHS[c][0]) + 1 for c in content) - 1) * size


def pixel_text(
    x: float, y: float, content: str, color: str, size: int, shadow: str | None
) -> str:
    """Draw text in the bitmap font, one horizontal run of lit pixels per path segment."""

    def runs(dx: float, dy: float) -> str:
        d = []
        cursor = x + dx
        for char in content:
            glyph = GLYPHS[char]
            for row, line in enumerate(glyph):
                for match in re.finditer(r"#+", line):
                    px = cursor + match.start() * size
                    py = y + dy + row * size
                    d.append(
                        f"M{px:g} {py:g}h{len(match.group()) * size}v{size}h-{len(match.group()) * size}z"
                    )
            cursor += (len(glyph[0]) + 1) * size
        return "".join(d)

    parts = []
    if shadow:
        parts.append(f'<path fill="{shadow}" d="{runs(size, size)}"/>')
    parts.append(f'<path fill="{color}" d="{runs(0, 0)}"/>')
    return "".join(parts)


def rect(x: float, y: float, w: float, h: float, color: str) -> str:
    return f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{color}"/>'


def pixel_bar(x: float, y: float, width: float, colors: tuple[str, str, str]) -> str:
    """A bar like the monster's body: outline with cut corners, highlight on top, shade below and at the end."""
    highlight, fill, shade = colors
    width = max(round(width / U) * U, 6 * U)
    h = BAR_HEIGHT
    return "".join(
        (
            rect(x + U, y, width - 2 * U, h, OUTLINE),
            rect(x, y + U, width, h - 2 * U, OUTLINE),
            rect(x + U, y + U, width - 2 * U, h - 2 * U, fill),
            rect(x + U, y + U, width - 3 * U, U, highlight),
            rect(x + U, y + h - 2 * U, width - 2 * U, U, shade),
            rect(x + width - 2 * U, y + 2 * U, U, h - 3 * U, shade),
        )
    )


def dotted_line(y: float, color: str) -> str:
    d = "".join(f"M{x} {y}h{U}v{U}h-{U}z" for x in range(PAD, WIDTH - PAD, 3 * U))
    return f'<path fill="{color}" d="{d}"/>'


def card(width: int, height: int, theme: dict[str, str | None]) -> str:
    """The background: a rectangle with stepped pixel corners and a 1-pixel border."""
    s = 2 * U

    def stepped(inset: float) -> str:
        x0, y0, x1, y1 = inset, inset, width - inset, height - inset
        return (
            f"M{x0 + 2 * s} {y0}H{x1 - 2 * s}v{s}h{s}v{s}h{s}V{y1 - 2 * s}h-{s}v{s}h-{s}v{s}"
            f"H{x0 + 2 * s}v-{s}h-{s}v-{s}h-{s}V{y0 + 2 * s}h{s}v-{s}h{s}z"
        )

    return f'<path fill="{theme["border"]}" d="{stepped(0)}"/><path fill="{theme["surface"]}" d="{stepped(U)}"/>'


def monster(theme_name: str, x: float, y: float, scale: float) -> str:
    """The logo's monster, without the wordmark below it."""
    logo = (ASSETS / f"logo-{theme_name}.svg").read_text(encoding="utf-8")
    # The monster spans x 63-91 and y 4-29 of the logo's 96x48 grid. The wordmark starts
    # at row 28, where only the monster's outline and feet belong to it.
    feet = ("#1f0c10", "#f6b9a9", "#d48a7a")
    paths = []
    for fill, d in re.findall(r'<path fill="([^"]+)" d="([^"]+)"', logo):
        pixels = [
            p
            for p in re.findall(r"M[^M]+", d)
            if int(p.split()[1].split("h")[0]) < 28 or fill in feet
        ]
        if pixels:
            paths.append(f'<path fill="{fill}" d="{"".join(pixels)}"/>')
    return f'<g transform="translate({x - 63 * scale} {y - 4 * scale}) scale({scale})">{"".join(paths)}</g>'


def render(name: str, theme: dict[str, str | None]) -> str:
    ink, muted, shadow = theme["ink"], theme["muted"], theme["shadow"]
    height = HEADER + ROW_HEIGHT * len(ROWS) + FOOTER
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" shape-rendering="crispEdges" role="img" '
        'aria-label="noentenc against the original packages: 5.5 times smaller install, '
        "4 times faster start-up, 1.9 times the translation throughput and 470 times "
        'the langid throughput">',
        card(WIDTH, height, theme),
        monster(name, WIDTH - PAD - 29 * 3, 18, 3),
        pixel_text(PAD, 32, "SAME MODELS, LIGHTER AND FASTER", ink, 3, shadow),
        rect(PAD, 62, text_width("SAME MODELS, LIGHTER AND FASTER", 3), U, "#f7c21b"),
    ]

    legend_x = PAD
    for label, colors in (("NOENTENC", OURS), ("ORIGINAL PACKAGE", THEIRS)):
        parts.append(pixel_bar(legend_x, 80, 10 * U, colors))
        parts.append(pixel_text(legend_x + 14 * U, 82, label, ink, U, shadow))
        legend_x += 14 * U + text_width(label, U) + 16 * U

    for i, (
        title,
        subtitle,
        ours,
        ours_label,
        theirs,
        theirs_label,
        ratio,
        verdict,
    ) in enumerate(ROWS):
        top = HEADER + i * ROW_HEIGHT
        parts.append(dotted_line(top, theme["grid"]))
        parts.append(pixel_text(PAD, top + 22, title, ink, U, shadow))
        parts.append(pixel_text(PAD, top + 44, subtitle, muted, U, None))

        scale = BAR_MAX / max(ours, theirs)
        for j, (value, label, colors) in enumerate(
            ((ours, ours_label, OURS), (theirs, theirs_label, THEIRS))
        ):
            y = top + 20 + j * (BAR_HEIGHT + 2 * U)
            width = max(round(value * scale / U) * U, 6 * U)
            parts.append(pixel_bar(BAR_X, y, width, colors))
            parts.append(
                pixel_text(
                    BAR_X + width + 5 * U,
                    y + U,
                    label,
                    ink if j == 0 else muted,
                    U,
                    None,
                )
            )

        right = WIDTH - PAD
        parts.append(
            pixel_text(right - text_width(ratio, 4), top + 14, ratio, ink, 4, shadow)
        )
        parts.append(
            pixel_text(
                right - text_width(verdict, U), top + 50, verdict, muted, U, None
            )
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    for name, theme in THEMES.items():
        path = ASSETS / f"benchmark-{name}.svg"
        path.write_text(render(name, theme), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
