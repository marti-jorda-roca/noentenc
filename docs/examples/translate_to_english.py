"""Translate a mixed-language inbox into English: detect each text's language, then
translate each language group in one batch.

Run: uv run python docs/examples/translate_to_english.py
"""

from collections import defaultdict

from noentenc import Language
from noentenc.language_detection import LanguageDetector, to_iso639_3
from noentenc.translation import Translator

messages = [
    "Hola, ¿cuándo llega mi pedido?",
    "Bonjour, je voudrais annuler mon abonnement.",
    "Thanks, the issue is fixed now.",
    "Der Link im Newsletter funktioniert nicht.",
    "¿Puedo cambiar la dirección de entrega?",
]

# The detector returns ISO 639-3 codes ("spa"); `Language` values are ISO 639-1 where one
# exists ("es"). Map one onto the other once.
BY_ISO639_3 = {to_iso639_3(language): language for language in Language}

detector = LanguageDetector()
translator = Translator()  # Opus-MT for direct pairs, SMaLL-100 for everything else

groups: dict[Language, list[int]] = defaultdict(list)
for i, code in enumerate(detector.detect_batch(messages)):
    language = BY_ISO639_3.get(code)
    if language is not None and language != Language.ENGLISH:
        groups[language].append(i)

english = list(messages)
for source, indices in groups.items():
    translations = translator.translate_batch(
        [messages[i] for i in indices], Language.ENGLISH, source
    )
    for i, translation in zip(indices, translations, strict=True):
        english[i] = translation

for original, translated in zip(messages, english, strict=True):
    print(f"{original!r:50} -> {translated!r}")
