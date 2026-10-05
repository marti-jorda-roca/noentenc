"""How often URLs, emails, code, placeholders, tags and numbers survive translation.

Translates support-style messages full of literals with each model, with and without
`preserve`, and reports the share of texts whose literals all came back byte-for-byte, and
how often preservation needed the piece-by-piece fallback.

    uv run python scripts/evaluate_literals.py
    uv run python scripts/evaluate_literals.py --models opus-mt small100 nllb
"""

import argparse
import json
import warnings
from collections.abc import Callable
from pathlib import Path

from noentenc import Language as L
from noentenc.translation import (
    M2M100Model,
    NLLBModel,
    OpusMTModel,
    SMaLL100Model,
    Translator,
)
from noentenc.translation._literals import find_literals

# Messages per source language (ISO 639-1), in scripts/data/literal_messages.json.
MESSAGES = {
    L(source): texts
    for source, texts in json.loads(
        (Path(__file__).parent / "data" / "literal_messages.json").read_text(
            encoding="utf-8"
        )
    ).items()
}

TARGETS = {
    L.ENGLISH: (L.SPANISH, L.GERMAN),
    L.SPANISH: (L.ENGLISH,),
    L.GERMAN: (L.ENGLISH,),
}


def _translators(names: list[str]) -> dict[str, Callable[[L, L], Translator | None]]:
    multilingual = {"small100": SMaLL100Model, "nllb": NLLBModel, "m2m100": M2M100Model}
    loaded: dict[str, Translator] = {}

    def opus(source: L, target: L) -> Translator:
        key = f"{source}-{target}"
        if key not in loaded:
            loaded[key] = Translator(OpusMTModel.from_pair(source, target))
        return loaded[key]

    def shared(name: str) -> Callable[[L, L], Translator]:
        def get(_source: L, _target: L) -> Translator:
            if name not in loaded:
                loaded[name] = Translator(multilingual[name]())
            return loaded[name]

        return get

    found: dict[str, Callable[[L, L], Translator | None]] = {}
    for name in names:
        found[name] = opus if name == "opus-mt" else shared(name)
    return found


def _kept(original: str, translated: str) -> bool:
    return all(
        translated.count(original[a:b]) >= original.count(original[a:b])
        for a, b in find_literals(original)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["opus-mt", "small100"],
        choices=["opus-mt", "small100", "nllb", "m2m100"],
    )
    names = parser.parse_args().models
    warnings.simplefilter("ignore")
    print(
        "| Model | Texts | Literals kept, `preserve=False` | Literals kept | Placeholders | Pieces |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for name, translator_for in _translators(names).items():
        counts = {"texts": 0, "raw": 0, "kept": 0, "placeholders": 0, "segments": 0}
        for source, texts in MESSAGES.items():
            for target in TARGETS[source]:
                translator = translator_for(source, target)
                if translator is None:
                    continue
                raw = translator.translate_batch(texts, target, source, preserve=False)
                kept = translator.translate_batch(texts, target, source, detailed=True)
                for text, plain, result in zip(texts, raw, kept, strict=True):
                    counts["texts"] += 1
                    counts["raw"] += _kept(text, plain)
                    counts["kept"] += _kept(text, result.text)
                    counts[str(result.preservation)] = (
                        counts.get(str(result.preservation), 0) + 1
                    )
        total = counts["texts"]
        print(
            f"| {name} | {total} | {100 * counts['raw'] / total:.0f}% "
            f"| {100 * counts['kept'] / total:.0f}% "
            f"| {100 * counts['placeholders'] / total:.0f}% "
            f"| {100 * counts['segments'] / total:.0f}% |",
            flush=True,
        )


if __name__ == "__main__":
    main()
