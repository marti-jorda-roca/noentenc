"""Pick a translation model, precision and thread count instead of the defaults.

Run: uv run python docs/examples/choose_translation_model.py
"""

import pandas as pd

from noentenc import Language, UnsupportedLanguageError
from noentenc.translation import (
    OPUS_MT_PAIRS,
    M2M100Model,
    OpusMTModel,
    Precision,
    SMaLL100Model,
    Translator,
)

EN, ES, CA = Language.ENGLISH, Language.SPANISH, Language.CATALAN

# 1. One direction, as fast as possible: a dedicated ~75M-param Opus-MT model.
print((EN, ES) in OPUS_MT_PAIRS)  # True
opus = Translator(OpusMTModel.from_pair(EN, ES, precision=Precision.Q4))
print(opus.translate("The meeting was moved to Friday.", ES))

# 2. Many directions with one model: SMaLL-100 (MIT, 100 languages, source optional).
#    Cap the threads when several workers share a machine.
small100 = Translator(SMaLL100Model(num_threads=2))
print(small100.translate("Bon dia a tothom!", EN))

# 3. Also MIT: M2M100 418M. About as accurate as SMaLL-100, better into Chinese and
#    Japanese but worse on low-resource languages, about 4x slower, and it needs the
#    source language.
m2m100 = Translator(M2M100Model())
print(m2m100.translate("Bon dia a tothom!", EN, source_language=CA))

# Every model declares the languages it handles, and an unsupported pair raises.
try:
    opus.translate("Hello", Language.FRENCH)
except UnsupportedLanguageError as error:
    print(error)  # OpusMTModel cannot translate into <Language.FRENCH: 'fr'>

# 4. Whole DataFrame columns (pandas or polars). Null cells stay null.
df = pd.DataFrame({"title": ["Summer sale", None, "Free shipping over 50 €"]})
print(opus.translate_dataset(df, "title", "title_es", ES))

# 5. Air-gapped servers: download ahead of time (OpusMTModel.download(...) or
#    noentenc.prepare(...)), then refuse to touch the network.
offline = OpusMTModel.from_pair(EN, ES, only_local_files=True)
# NLLBModel() (200 languages) is also available, but it is CC-BY-NC-4.0: non-commercial only.
