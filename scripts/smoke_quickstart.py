"""Run the README quickstart against an installed noentenc, as CI does with the built wheel.

Both run the README's example as written, on the default models. `detection` expects a
core-only install: it checks that the translation dependencies are neither installed nor
imported, and downloads lid176 (0.9 MB). `translation` expects `noentenc[translation]` and
downloads Opus-MT es→en and de→en at q4 (554 MB with lid176).

    uv run --no-project --isolated --with dist/noentenc-*.whl \
        python scripts/smoke_quickstart.py detection
    uv run --no-project --isolated --with "noentenc[translation] @ $(ls dist/*.whl)" \
        python scripts/smoke_quickstart.py translation
"""

import argparse
import importlib.util
import sys

TRANSLATION_DEPENDENCIES = ("onnxruntime", "tokenizers", "huggingface_hub")


def detection() -> None:
    installed = [m for m in TRANSLATION_DEPENDENCIES if importlib.util.find_spec(m)]
    if installed:
        raise SystemExit(f"expected a core-only install, but found {installed}")

    from noentenc.language_detection import LanguageDetector

    detector = LanguageDetector()
    label = detector.detect("Bon dia! Com estàs?")
    labels = detector.detect_batch(["Hello there", "Hola, ¿qué tal?", ""])
    streamed = list(
        detector.detect_stream(iter(["Hello there", "Hola, ¿qué tal?", ""]))
    )
    print(label, labels, streamed)
    if label != "cat" or labels != ["eng", "spa", "und"] or streamed != labels:
        raise SystemExit(f"unexpected detection: {label!r} {labels!r} {streamed!r}")
    imported = [m for m in TRANSLATION_DEPENDENCIES if m in sys.modules]
    if imported:
        raise SystemExit(f"detection imported {imported}")


def translation() -> None:
    from noentenc.translation import TranslationStatus, Translator

    # The README quickstart, as written.
    inbox = [
        "Hola, ¿cuándo llega mi pedido?",
        "Der Link funktioniert nicht.",
        "Thanks, it works now.",
    ]
    translator = Translator()
    texts = translator.translate_batch(inbox, "en", "auto")
    print(texts)
    if "order" not in texts[0].lower() or "link" not in texts[1].lower():
        raise SystemExit(f"unexpected translation: {texts!r}")

    results = translator.translate_batch(inbox, "en", "auto", detailed=True)
    routes = [(r.status, r.detected_language, r.model) for r in results]
    print(routes)
    if [r.text for r in results] != texts or routes != [
        (TranslationStatus.TRANSLATED, "spa", "OpusMTModel(Xenova/opus-mt-es-en)"),
        (TranslationStatus.TRANSLATED, "deu", "OpusMTModel(Xenova/opus-mt-de-en)"),
        (TranslationStatus.UNCHANGED, "eng", None),
    ]:
        raise SystemExit(f"unexpected routing: {routes!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", choices=["detection", "translation"])
    {"detection": detection, "translation": translation}[parser.parse_args().install]()


if __name__ == "__main__":
    main()
