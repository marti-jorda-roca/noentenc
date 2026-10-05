"""Route support messages by language, and abstain instead of guessing on unclear input.

Run: uv run python docs/examples/conservative_detection.py
"""

from noentenc.language_detection import DetectionStatus, LanguageDetector

messages = [
    "My password reset link doesn't work.",
    "No puedo iniciar sesión en mi cuenta.",
    "lol",
    "Hans Müller",
    "https://example.com/reset?id=42",
    "👍",
    "",
]

# By default every text with letters gets the model's best label, however unsure it is.
print(LanguageDetector().detect_batch(messages))
# ['eng', 'spa', 'eng', 'deu', 'zxx', 'zxx', 'und']

# Digits, emoji, punctuation and bare links are "zxx" (no linguistic content) and never reach
# the model. To abstain on the rest when the model is unsure, set thresholds. These are the
# settings docs/benchmarks.md tested for lid176; scores differ between backends.
detector = LanguageDetector(min_letters=4, min_score=0.5)
for message, detection in zip(
    messages, detector.detect_batch(messages, detailed=True), strict=True
):
    print(f"{message!r:40} {detection.language} {detection.status}")
# "My password reset link doesn't work."   eng detected
# 'No puedo iniciar sesión en mi cuenta.'  spa detected
# 'lol'                                    und insufficient_text
# 'Hans Müller'                            und low_confidence
# 'https://example.com/reset?id=42'        zxx no_linguistic_content
# '👍'                                      zxx no_linguistic_content
# ''                                       und empty

# The best guess is still there when the detector abstains.
unsure = detector.detect("Hans Müller", detailed=True)
print(unsure.top_language, round(unsure.score or 0, 2))
# deu 0.35

# Route only confident detections; send the rest to a person.
routed = [
    detection.language
    for detection in detector.detect_batch(messages, detailed=True)
    if detection.status is DetectionStatus.DETECTED
]
print(routed)
# ['eng', 'spa']

# When the inbox only ever has a few languages, restrict the answer to them. "ca" and "cat"
# both work. The winner keeps its own score; scores aren't renormalised over the candidates.
support_languages = LanguageDetector(candidates=["eng", "spa", "cat"])
print(support_languages.detect("Bon dia, no puc entrar al compte"))
# cat
