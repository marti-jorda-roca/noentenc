"""Wrong-label and abstention rates of each detection backend under conservative settings.

The cases in `scripts/data/detection_cases.tsv` are hand-labelled support messages, typos,
names, closely related languages, mixed-language messages, long texts, nonlinguistic input
and ambiguous chat tokens. `expected` is an ISO 639-3 code (alternatives separated by `|`),
`und` for texts that carry no identifiable language (names, "lol") and `zxx` for
nonlinguistic input. Labels are compared with macrolanguages collapsed on both sides.

For each backend and setting it reports, over the cases with a language, how many got the
right label, a wrong label or an abstention (`und`/`zxx`); and how many of the names and
ambiguous tokens were abstained on, and nonlinguistic inputs labelled `zxx`.

    uv run python scripts/evaluate_detection.py
    uv run python scripts/evaluate_detection.py --backends lid176 lingua --by-category
"""

import argparse
import csv
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from noentenc.language_detection import LanguageDetector
from noentenc.language_detection.labels import to_iso639_3
from noentenc.language_detection.models.base import BaseModel

CASES = Path(__file__).parent / "data" / "detection_cases.tsv"
ABSTAIN = {"und", "zxx"}
# Categories whose texts have a language, and those a conservative detector should skip.
LANGUAGE_CATEGORIES = ("support", "typo", "related", "mixed", "long")
SKIP_CATEGORIES = ("name", "ambiguous")


def _backends() -> dict[str, Callable[[], BaseModel]]:
    from noentenc.language_detection import (
        Cld3Model,
        FastTextModel,
        HeliportModel,
        LangidModel,
        LinguaModel,
        OnnxClassifierModel,
    )

    return {
        "lid176": FastTextModel,
        "openlid-v3": lambda: FastTextModel("openlid-v3"),
        "glotlid": lambda: FastTextModel("glotlid"),
        "langid": LangidModel,
        "bert-openlid": OnnxClassifierModel,
        "lingua": LinguaModel,
        "cld3": Cld3Model,
        "heliport": HeliportModel,
    }


# Settings tried per backend. Score thresholds depend on what each backend's scores are:
# heliport's confidences are mostly between 0 and 2.5, the others' between 0 and 1.
_PROBABILITY_SETTINGS: list[dict[str, float | int]] = [
    {},
    {"min_letters": 4},
    {"min_score": 0.5},
    {"min_score": 0.7},
    {"min_score": 0.9},
    {"min_margin": 0.1},
    {"min_margin": 0.3},
    {"min_letters": 4, "min_score": 0.5},
    {"min_letters": 4, "min_score": 0.7},
    {"min_letters": 4, "min_margin": 0.3},
]
SETTINGS: dict[str, list[dict[str, float | int]]] = {
    "heliport": [
        {},
        {"min_letters": 4},
        {"min_score": 0.3},
        {"min_score": 0.5},
        {"min_score": 1.0},
        {"min_letters": 4, "min_score": 0.3},
        {"min_letters": 4, "min_score": 0.5},
    ],
}


@dataclass(frozen=True)
class Case:
    category: str
    expected: frozenset[str]
    text: str


def load_cases() -> list[Case]:
    with CASES.open(encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [
            Case(
                row["category"],
                frozenset(_collapse(code) for code in row["expected"].split("|")),
                row["text"],
            )
            for row in rows
        ]


def _collapse(code: str) -> str:
    return code if code in ABSTAIN else to_iso639_3(code, collapse_macrolanguages=True)


def outcome(case: Case, label: str) -> str:
    """ "right", "wrong" or "abstained"."""
    label = _collapse(label)
    if case.category == "nonlinguistic":
        return "right" if label == "zxx" else "wrong"
    if case.category in SKIP_CATEGORIES:
        return "right" if label in ABSTAIN else "wrong"
    if label in ABSTAIN:
        return "abstained"
    return "right" if label in case.expected else "wrong"


def describe(setting: dict[str, float | int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in setting.items()) or "default"


def _pct(count: int, total: int) -> str:
    return f"{100 * count / total:.0f}%" if total else ""


def evaluate(
    model: BaseModel, setting: dict[str, float | int], cases: list[Case]
) -> dict[str, Counter[str]]:
    detector = LanguageDetector(model, **setting)  # ty: ignore[invalid-argument-type]
    labels = detector.detect_batch([case.text for case in cases])
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case, label in zip(cases, labels, strict=True):
        counts[case.category][outcome(case, label)] += 1
    return counts


def report(name: str, results: list[tuple[str, dict[str, Counter[str]]]]) -> None:
    print(f"\n### {name}\n")
    print(
        "| Setting | Right | Wrong | Abstained | Names/ambiguous abstained | Nonlinguistic `zxx` |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for setting, counts in results:
        language = sum((counts[c] for c in LANGUAGE_CATEGORIES), Counter())
        skip = sum((counts[c] for c in SKIP_CATEGORIES), Counter())
        total, skip_total = language.total(), skip.total()
        nonlinguistic = counts["nonlinguistic"]
        print(
            f"| {setting} | {_pct(language['right'], total)} "
            f"| {_pct(language['wrong'], total)} "
            f"| {_pct(language['abstained'], total)} "
            f"| {_pct(skip['right'], skip_total)} "
            f"| {_pct(nonlinguistic['right'], nonlinguistic.total())} |"
        )


def report_categories(name: str, setting: str, counts: dict[str, Counter[str]]) -> None:
    print(f"\n{name}, {setting}, by category (right / wrong / abstained):")
    for category, counter in counts.items():
        total = counter.total()
        print(
            f"  {category:14} {_pct(counter['right'], total):>5} "
            f"{_pct(counter['wrong'], total):>5} {_pct(counter['abstained'], total):>5}"
            f"  ({total} texts)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["lid176", "langid", "bert-openlid", "lingua", "cld3", "heliport"],
        choices=list(_backends()),
    )
    parser.add_argument("--by-category", action="store_true")
    args = parser.parse_args()
    cases = load_cases()
    print(f"{len(cases)} cases: {dict(Counter(case.category for case in cases))}")
    for name in args.backends:
        model = _backends()[name]()
        results = [
            (describe(setting), evaluate(model, setting, cases))
            for setting in SETTINGS.get(name, _PROBABILITY_SETTINGS)
        ]
        report(name, results)
        if args.by_category:
            for setting, counts in results:
                report_categories(name, setting, counts)


if __name__ == "__main__":
    main()
