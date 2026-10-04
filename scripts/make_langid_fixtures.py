"""Build langid parity fixtures with the reference ``langid`` package.

``langid`` isn't a dependency of this project, so run this in a throwaway environment:

    uv run --no-project --with langid --with numpy python scripts/make_langid_fixtures.py

It writes ``tests/unit/language_detection/fixtures/langid/expected.json``: langid's top-5
normalised probabilities for every line of ``fixtures/sentences.txt``, once with all 97 languages
(``langid``) and once restricted to ``RESTRICTED`` (``langid-restricted``).
"""

import json
from pathlib import Path

from langid.langid import (  # ty: ignore[unresolved-import] - only available in the fixture environment
    LanguageIdentifier,
    model,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/unit/language_detection/fixtures"
OUT = FIXTURES / "langid"
TOP_K = 5
RESTRICTED = ["ca", "es", "pt", "en"]


def _predictions(identifier: LanguageIdentifier, sentences: list[str]) -> list[dict]:
    out = []
    for sentence in sentences:
        ranked = identifier.rank(sentence)[:TOP_K]
        out.append(
            {"labels": [code for code, _ in ranked], "probs": [p for _, p in ranked]}
        )
    return out


def main() -> None:
    sentences = (FIXTURES / "sentences.txt").read_text(encoding="utf-8").splitlines()
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    expected = {"langid": _predictions(identifier, sentences)}
    identifier.set_languages(RESTRICTED)
    expected["langid-restricted"] = _predictions(identifier, sentences)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "expected.json").write_text(
        json.dumps(expected, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
