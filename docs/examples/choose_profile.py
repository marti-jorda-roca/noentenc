"""Trade latency for quality with a profile instead of picking models by hand.

Run: uv run python docs/examples/choose_profile.py
The `balance` and `quality` profiles download more than a gigabyte on first use.
"""

import warnings

from noentenc import Language, Profile
from noentenc.language_detection import LanguageDetector
from noentenc.translation import Translator

EN, ES, SW = Language.ENGLISH, Language.SPANISH, Language.SWAHILI
texts = ["Habari ya asubuhi, karibu nyumbani.", "Bon dia! Com estàs?"]

# `speed` is the default: LanguageDetector() and Translator() use it.
fast_detector = LanguageDetector()  # fastText lid176, 0.9 MB
print(fast_detector.detect_batch(texts))  # ['sun', 'cat']: lid176 mistakes Swahili

# Profiles are plain strings or `Profile` members; a typo raises ValueError.
balanced_detector = LanguageDetector("balance")  # fastText OpenLID-v3, 1.2 GB
print(balanced_detector.detect_batch(texts))  # ['swh', 'cat']
best_detector = LanguageDetector(Profile.QUALITY)  # fastText GlotLID, 1.7 GB
print(best_detector.detect_batch(texts, with_score=True, top_k=2))
# [{'swc': 0.88, 'swh': 0.11}, {'cat': 1.0, 'por': 0.0}]
# GlotLID tells apart varieties such as Congo Swahili (swc) and Swahili (swh).
# LanguageDetector(FastTextModel("glotlid", collapse_macrolanguages=True)) folds both
# into Swahili (swa).

# Every translation profile uses a dedicated Opus-MT model when the pair has one:
# there it scores as well as NLLB-200 on average and is about 8x faster. English -> Spanish is such a pair, so all three
# profiles translate it with the same model.
print(Translator("quality").translate("Where is the station?", ES, EN))
# '¿Dónde está la estación?'

# They differ on the other pairs. Swahili -> English has no Opus-MT model:
# `speed` uses SMaLL-100, `balance` NLLB-200 at int8 and `quality` NLLB-200 at fp32.
fast = Translator()
print(fast.translate(texts[0], EN, SW))  # 'The morning news, near home.'

# NLLB-200 is CC-BY-NC-4.0 (non-commercial) and warns when it loads.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    balanced = Translator(Profile.BALANCE)
    print(balanced.translate(texts[0], EN, SW))  # 'Good morning, close to home.'

# Without a source language only SMaLL-100 can translate, so every profile uses it.
print(balanced.translate("Bon dia a tothom!", EN))  # 'Good day to everyone!'

# A model passed explicitly always wins; use one when you need reproducible output,
# since the models behind a profile may change between releases.
