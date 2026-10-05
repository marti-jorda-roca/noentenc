"""Translate a bulk job without one bad row stopping it, and without needless downloads.

Run: uv run python docs/examples/translate_bulk.py
"""

import polars as pl

from noentenc import Language
from noentenc.translation import TranslationStatus, Translator

EN, ES = Language.ENGLISH, Language.SPANISH
translator = Translator()

# 1. Bad arguments fail at once, before any model is chosen or downloaded.
for batch_size in (0, -1, 2.5):
    try:
        translator.translate_batch(["Hello."], ES, EN, batch_size=batch_size)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError) as error:
        print(error)
        # batch_size must be >= 1, got 0
        # batch_size must be >= 1, got -1
        # batch_size must be an int, got float
try:
    translator.translate_batch("Hello.", ES, EN)  # ty: ignore[invalid-argument-type]
except TypeError as error:
    print(error)  # texts must be a list of strings, got str; pass a single text ...

# 2. Input that needs no translation comes back as given, without loading a model:
#    an empty batch, blank texts, and text already in the target language.
print(translator.translate_batch([], ES, EN))  # []
print(translator.translate_batch(["", "  "], ES, EN))  # ['', '  ']
print(translator.translate_batch(["Hello."], EN, EN))  # ['Hello.']

# 3. By default the first text that fails raises. A sentence longer than the model
#    reads (about 500 tokens) is one way to fail.
run_on = " ".join(["the weather is nice"] * 150)
texts = ["Where is the station?", run_on, "I love this city."]

# errors="record" keeps going: the failing text comes back as given, the others
# are still translated, and the detailed result says which rows failed and why.
results = translator.translate_batch(texts, ES, EN, detailed=True, errors="record")
for i, result in enumerate(results):
    print(i, result.status, result.text[:30])
    # 0 translated ¿Dónde está la estación?
    # 1 failed the weather is nice the weathe
    # 2 translated Me encanta esta ciudad.
print(results[1].status is TranslationStatus.FAILED)  # True
# Only failed results carry an error.
failed = {i: r.error for i, r in enumerate(results) if r.error is not None}
print(list(failed))  # [1]
print(failed[1][:60])  # InputTooLongError: OpusMTModel reads at most 511 tokens per

# 4. DataFrames work the same way. Null rows stay null, failed rows get a null
#    translation, and error_column says why. Row order is kept.
df = pl.DataFrame({"text": ["Good morning.", None, run_on, "Thank you!"]})
df = translator.translate_dataset(
    df,
    "text",
    "text_es",
    ES,
    EN,
    show_progress=False,
    errors="record",
    error_column="error",
)
print(df.select("text_es", pl.col("error").str.slice(0, 17)))
# shape: (4, 2)
# ┌──────────────┬───────────────────┐
# │ text_es      ┆ error             │
# │ ---          ┆ ---               │
# │ str          ┆ str               │
# ╞══════════════╪═══════════════════╡
# │ Buenos días. ┆ null              │
# │ null         ┆ null              │
# │ null         ┆ InputTooLongError │
# │ ¡Gracias!    ┆ null              │
# └──────────────┴───────────────────┘
