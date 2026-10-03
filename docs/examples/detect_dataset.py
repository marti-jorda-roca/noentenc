"""Tag a DataFrame column with its language, then keep only confident English rows.

Run: uv run python docs/examples/detect_dataset.py
"""

import polars as pl

from noentenc.language_detection import LanguageDetector

MIN_SCORE = 0.8

reviews = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "text": [
            "Great product, arrived on time.",
            "El envío tardó tres semanas.",
            "Produit conforme à la description.",
            None,
            "ok",
        ],
    }
)

detector = LanguageDetector()  # fastText lid.176: 0.9 MB, ~360k texts/s

# One label per row. Null and empty texts get "und".
tagged = detector.detect_dataset(reviews, "text", "lang")
print(tagged)

# With scores, each cell is a list of {"language", "score"} records, best first.
scored = detector.detect_dataset(reviews, "text", "langs", with_score=True, top_k=1)
english = scored.filter(
    (pl.col("langs").list.first().struct.field("language") == "eng")
    & (pl.col("langs").list.first().struct.field("score") >= MIN_SCORE)
)
print(english.select("id", "text"))

# pandas works the same way:
#   detector.detect_dataset(df, "text", "lang")  # df is a pandas.DataFrame
