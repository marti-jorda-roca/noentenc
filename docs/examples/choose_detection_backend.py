"""Swap the language-detection backend and tune its labels.

Run: uv run python docs/examples/choose_detection_backend.py
Needs the extras used below: uv add 'noentenc[lingua]'
"""

from noentenc.language_detection import (
    FastTextModel,
    LanguageDetector,
    LinguaModel,
    OnnxClassifierModel,
)

texts = ["Bon dia, com estàs?", "Bom dia, como você está?", "我们明天见"]

# Default: fastText lid.176. Ranked scores for the top candidates:
detector = LanguageDetector()
print(detector.detect(texts[0], with_score=True, top_k=3))

# Every backend returns the same ISO 639-3 labels, so swapping one is a one-line change.
# A small transformer trained on OpenLID (201 languages, 25 MB int8):
bert = LanguageDetector(OnnxClassifierModel("bert-openlid"))
print(bert.detect_batch(texts))

# You already know the input is one of a few languages: restrict the candidates.
# Lingua is strongest on short texts; `languages` takes ISO 639-3 codes.
lingua = LanguageDetector(LinguaModel(languages=["cat", "spa", "por", "eng"]))
print(lingua.detect_batch(texts[:2]))

# Some models say "cmn" (Mandarin), others "zho" (Chinese). Collapse individual languages
# into their macrolanguage so results from different models line up.
collapsed = LanguageDetector(FastTextModel(collapse_macrolanguages=True))
print(collapsed.detect(texts[2]))

# Or get each model's own codes back.
native = LanguageDetector(FastTextModel(normalize_labels=False))
print(native.detect(texts[2]))  # 'zh'

# Long-tail languages (2102 labels, 1.7 GB download): LanguageDetector(FastTextModel("glotlid"))
# Your own fastText model: LanguageDetector(FastTextModel("path/to/model.bin"))
