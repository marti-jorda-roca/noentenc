"""Score detection and translation models on FLORES-200 devtest, as behind each Profile.

Usage:
    curl -O https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
    tar -xzf flores200_dataset.tar.gz
    uv run python scripts/benchmark_accuracy.py lid --flores flores200_dataset
    uv run --with sacrebleu python scripts/benchmark_accuracy.py translation \\
        --flores flores200_dataset

`lid` reports macro-averaged accuracy, with macrolanguages collapsed on both sides
(`arb`, `ary` → `ara`): over every FLORES language, over the ones each model knows, over
the ones every model knows, and the same on texts cut to 40 characters.

`translation` reports corpus chrF++ for each pair and model (`--models`), on the first
`--sentences` sentences. Weights are downloaded on first use. `opus-mt` is Opus-MT at its
default precision; `opus-mt-q4`, `opus-mt-int8` and `opus-mt-fp32` pick one. `--pairs`
replaces the default pairs, e.g. `--pairs en-es ja-en`.
"""

import argparse
import warnings
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from noentenc.language_detection.labels import to_iso639_3
from noentenc.language_detection.models import (
    BaseModel,
    FastTextModel,
    LangidModel,
    OnnxClassifierModel,
)
from noentenc.languages import Language as L
from noentenc.translation import (
    OPUS_MT_PAIRS,
    M2M100Model,
    NLLBModel,
    OpusMTModel,
    Precision,
    SMaLL100Model,
)
from noentenc.translation.models._seq2seq import Seq2SeqModel
from noentenc.translation.models.nllb import NLLB_CODES

DETECTORS: dict[str, Callable[[], BaseModel]] = {
    "lid176": lambda: FastTextModel("lid176"),
    "lid176-bin": lambda: FastTextModel("lid176-bin"),
    "openlid-v3": lambda: FastTextModel("openlid-v3"),
    "glotlid": lambda: FastTextModel("glotlid"),
    "bert-openlid": lambda: OnnxClassifierModel("bert-openlid"),
    "langid": LangidModel,
}
SHORT_TEXT = 40

# Opus-MT ("opus-mt", or with a precision) is added for the pairs it has a model for.
OPUS_MT: dict[str, Precision | None] = {
    "opus-mt": None,
    "opus-mt-q4": Precision.Q4,
    "opus-mt-int8": Precision.INT8,
    "opus-mt-fp32": Precision.FP32,
}
TRANSLATORS: dict[str, Callable[[], Seq2SeqModel]] = {
    "small100": SMaLL100Model,
    "m2m100-int8": lambda: M2M100Model(precision=Precision.INT8),
    "m2m100-q4": lambda: M2M100Model(precision=Precision.Q4),
    "nllb-int8": lambda: NLLBModel(precision=Precision.INT8),
    "nllb-q4": lambda: NLLBModel(precision=Precision.Q4),
    "nllb-fp32": lambda: NLLBModel(precision=Precision.FP32),
}
DEFAULT_TRANSLATORS = ["opus-mt", "small100", "nllb-int8", "nllb-fp32"]
PAIRS = [
    (L.ENGLISH, L.SPANISH),
    (L.SPANISH, L.ENGLISH),
    (L.ENGLISH, L.GERMAN),
    (L.GERMAN, L.ENGLISH),
    (L.ENGLISH, L.CHINESE),
    (L.CHINESE, L.ENGLISH),
    (L.ENGLISH, L.FINNISH),
    (L.FRENCH, L.GERMAN),
    (L.ENGLISH, L.JAPANESE),
    (L.ENGLISH, L.SWAHILI),
    (L.SWAHILI, L.ENGLISH),
    (L.ENGLISH, L.TAMIL),
    (L.GERMAN, L.ITALIAN),
]


def _collapse(labels: list[str]) -> list[str]:
    return [to_iso639_3(label, collapse_macrolanguages=True) for label in labels]


def _macro_accuracy(
    gold: list[str], predicted: list[str], languages: set[str]
) -> float:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for expected, label in zip(gold, predicted, strict=True):
        if expected in languages:
            counts[expected][0] += expected == label
            counts[expected][1] += 1
    return 100 * sum(right / total for right, total in counts.values()) / len(counts)


def lid(flores: Path, per_language: int) -> None:
    texts: list[str] = []
    gold: list[str] = []
    for path in sorted((flores / "devtest").glob("*.devtest")):
        lines = path.read_text(encoding="utf-8").splitlines()[:per_language]
        texts += lines
        gold += [to_iso639_3(path.stem, collapse_macrolanguages=True)] * len(lines)
    short = [text[:SHORT_TEXT] for text in texts]
    languages = set(gold)

    known: dict[str, set[str]] = {}
    predictions: dict[str, tuple[list[str], list[str]]] = {}
    for name, factory in DETECTORS.items():
        model = factory()
        known[name] = {
            to_iso639_3(label, collapse_macrolanguages=True) for label in model.labels
        } & languages
        full, cut = model.predict_batch(texts), model.predict_batch(short)
        predictions[name] = (_collapse(full), _collapse(cut))
    common = set.intersection(*known.values())

    print(f"{len(texts)} texts, {len(languages)} languages, {len(common)} known by all")
    print(
        f"{'model':<14}{'all':>7}{'known':>7}{'common':>8}{'all 40':>8}{'common 40':>11}"
    )
    for name, (full, cut) in predictions.items():
        print(
            f"{name:<14}{_macro_accuracy(gold, full, languages):>7.2f}"
            f"{_macro_accuracy(gold, full, known[name]):>7.2f}"
            f"{_macro_accuracy(gold, full, common):>8.2f}"
            f"{_macro_accuracy(gold, cut, languages):>8.2f}"
            f"{_macro_accuracy(gold, cut, common):>11.2f}"
        )


def translation(
    flores: Path,
    sentences: int,
    names: list[str],
    pairs: list[tuple[L, L]] = PAIRS,
) -> None:
    import sacrebleu  # noqa: PLC0415 - only this benchmark needs it

    warnings.filterwarnings("ignore", "NLLB-200 is licensed")

    def lines(language: L) -> list[str]:
        path = flores / "devtest" / f"{NLLB_CODES[language]}.devtest"
        return path.read_text(encoding="utf-8").splitlines()[:sentences]

    shared = {name: TRANSLATORS[name]() for name in names if name not in OPUS_MT}
    print(f"{'pair':<8}{'model':<14}{'chrF++':>7}")
    for source, target in pairs:
        models: dict[str, Seq2SeqModel] = {}
        if (source, target) in OPUS_MT_PAIRS:
            models = {
                name: OpusMTModel.from_pair(source, target, precision=OPUS_MT[name])
                for name in names
                if name in OPUS_MT
            }
        for name, model in (models | shared).items():
            if not model.schema.supports(source, target):
                continue
            hypotheses = model.predict_batch(lines(source), target, source)
            score = sacrebleu.corpus_chrf(hypotheses, [lines(target)], word_order=2)
            print(f"{source}-{target:<5}{name:<14}{score.score:>7.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=["lid", "translation"])
    parser.add_argument(
        "--flores", type=Path, required=True, help="extracted flores200_dataset"
    )
    parser.add_argument(
        "--per-language", type=int, default=300, help="lid: texts per language"
    )
    parser.add_argument(
        "--sentences", type=int, default=200, help="translation: sentences per pair"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*OPUS_MT, *TRANSLATORS],
        default=DEFAULT_TRANSLATORS,
        help="translation: models to score",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        type=_pair,
        default=PAIRS,
        help="translation: source-target pairs, e.g. en-es (default: the benchmark's)",
    )
    args = parser.parse_args()
    if args.task == "lid":
        lid(args.flores, args.per_language)
    else:
        translation(args.flores, args.sentences, args.models, args.pairs)


def _pair(value: str) -> tuple[L, L]:
    source, target = value.split("-")
    return L(source), L(target)


if __name__ == "__main__":
    main()
