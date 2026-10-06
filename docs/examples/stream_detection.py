"""Detect the language of every line of a file too large to load at once.

`detect_stream` takes any iterable of strings, reads it a chunk at a time and yields one
label per text in input order, so memory stays flat however long the file is.

Run: uv run python docs/examples/stream_detection.py
"""

import tempfile
from collections import Counter
from collections.abc import Iterator
from itertools import islice, tee
from pathlib import Path

from noentenc.language_detection import LanguageDetector

# A stand-in for a large export: 60,000 comments, one per line.
COMMENTS = [
    "Great product, arrived on time.",
    "El envío tardó tres semanas.",
    "Produit conforme à la description.",
    "Schnelle Lieferung, gerne wieder!",
    "ok",
    "Spedizione veloce, consigliato!",
]
path = Path(tempfile.mkdtemp()) / "comments.txt"
path.write_text("\n".join(COMMENTS * 10_000) + "\n", encoding="utf-8")


def lines(path: Path) -> Iterator[str]:
    """The file's lines without their line breaks, read as they are needed."""
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield line.rstrip("\n")


# Abstain on texts with fewer than 4 letters, such as "ok".
detector = LanguageDetector(min_letters=4)

# 1. Count the languages in the whole file. Only one chunk of lines (1,024 by default)
#    and its labels are in memory at a time.
counts = Counter(detector.detect_stream(lines(path)))
print(counts.most_common())
# [('eng', 10000), ('spa', 10000), ('fra', 10000), ('deu', 10000), ('und', 10000),
#  ('ita', 10000)]

# 2. Write each line with its label as you go. `tee` hands the same lines to the detector
#    and to the loop; it holds the lines the detector has read ahead, at most one chunk.
texts, originals = tee(lines(path))
labelled = path.with_suffix(".tsv")
with labelled.open("w", encoding="utf-8") as out:
    for text, detection in zip(
        originals, detector.detect_stream(texts, detailed=True), strict=True
    ):
        out.write(f"{detection.language}\t{detection.status}\t{text}\n")
print(labelled.read_text(encoding="utf-8").splitlines()[:5])
# ['eng\tdetected\tGreat product, arrived on time.',
#  'spa\tdetected\tEl envío tardó tres semanas.',
#  'fra\tdetected\tProduit conforme à la description.',
#  'deu\tdetected\tSchnelle Lieferung, gerne wieder!',
#  'und\tinsufficient_text\tok']

# 3. Stop as soon as you have what you need: the rest of the file is never read.
texts, originals = tee(lines(path))
not_english = (
    text
    for text, language in zip(originals, detector.detect_stream(texts), strict=True)
    if language not in {"eng", "und"}
)
print(list(islice(not_english, 3)))
# ['El envío tardó tres semanas.', 'Produit conforme à la description.',
#  'Schnelle Lieferung, gerne wieder!']
