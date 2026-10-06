"""Run the quickstart against an installed noentenc, as CI does with the built wheel.

`detection` expects a core-only install: it checks that the translation dependencies are
neither installed nor imported. `translation` expects `noentenc[translation]` and downloads
the Opus-MT en→es model (107 MB at int8).

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
    from noentenc import Language
    from noentenc.translation import OpusMTModel, Precision, Translator

    translator = Translator(
        OpusMTModel.from_pair(
            Language.ENGLISH, Language.SPANISH, precision=Precision.INT8
        )
    )
    text = translator.translate(
        "The weather is nice today.", Language.SPANISH, Language.ENGLISH
    )
    print(text)
    if "tiempo" not in text.lower():
        raise SystemExit(f"unexpected translation: {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("install", choices=["detection", "translation"])
    {"detection": detection, "translation": translation}[parser.parse_args().install]()


if __name__ == "__main__":
    main()
