"""Build fastText parity fixtures with the reference ``fasttext`` implementation.

``fasttext`` has no Python 3.14 wheels, so run this in a throwaway 3.12 environment:

    uv run --no-project --python 3.12 --with fasttext-wheel --with "numpy<2" \\
        python scripts/make_fasttext_fixtures.py [--lid176 PATH/TO/lid.176.ftz] [--model NAME=PATH ...]

It writes, under ``tests/unit/language_detection/fixtures/fasttext/``:

* tiny models trained on a toy corpus, one per code path of our numpy engine (softmax, hs, ova,
  word n-grams, quantized with/without qnorm and pruning). fastText refuses to quantize an output
  matrix under 256 rows, so the `qout` path is covered by synthetic models in the unit tests instead;
* ``expected.json``: fastText's top-5 predictions for every line of ``fixtures/sentences.txt``,
  for each tiny model and for any real model passed with ``--lid176`` / ``--model``.
"""

import argparse
import json
import random
import tempfile
from pathlib import Path

import fasttext  # ty: ignore[unresolved-import] - only available in the fixture environment

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/unit/language_detection/fixtures"
OUT = FIXTURES / "fasttext"
TOP_K = 5

CORPUS = {
    "en": "the cat sat on the mat and the dog ran in the park while it was raining outside today",
    "es": "el gato se sentó en la alfombra y el perro corrió en el parque mientras llovía afuera hoy",
    "ca": "el gat seia a la catifa i el gos corria pel parc mentre plovia a fora avui",
    "de": "die katze saß auf der matte und der hund lief im park während es draußen regnete heute",
    "ru": "кошка сидела на коврике а собака бегала в парке пока на улице шёл дождь сегодня",
    "ja": "猫はマットの上に座って犬は公園を走った今日は外で雨が降っていた",
}

TINY_MODELS: dict[str, dict] = {
    "tiny-softmax.bin": {"loss": "softmax"},
    "tiny-hs.bin": {"loss": "hs"},
    "tiny-ova.bin": {"loss": "ova"},
    "tiny-wordngrams.bin": {"loss": "softmax", "wordNgrams": 2},
}
QUANTIZED_MODELS: dict[str, dict] = {
    "tiny-softmax-qnorm-pruned.ftz": {"qnorm": True, "qout": False, "cutoff": 300},
    "tiny-softmax-plain.ftz": {"qnorm": False, "qout": False, "cutoff": 0},
}


def _write_corpus(path: Path) -> None:
    rng = random.Random(0)
    lines = []
    for label, text in CORPUS.items():
        words = text.split() if label != "ja" else list(text)
        for _ in range(60):
            sample = rng.sample(words, k=min(len(words), rng.randint(3, 8)))
            lines.append(f"__label__{label} {' '.join(sample)}")
    rng.shuffle(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _train(corpus: Path, **overrides: object) -> fasttext.FastText._FastText:
    params = {
        "dim": 8,
        "minn": 2,
        "maxn": 4,
        "bucket": 5000,
        "epoch": 25,
        "minCount": 1,
        "lr": 0.5,
        "thread": 1,
        "seed": 0,
    }
    params.update(overrides)
    return fasttext.train_supervised(input=str(corpus), verbose=0, **params)


def _predictions(
    model: fasttext.FastText._FastText, sentences: list[str]
) -> list[dict]:
    rows = []
    for sentence in sentences:
        labels, probs = model.predict(sentence.replace("\n", " "), k=TOP_K)
        rows.append(
            {
                "labels": [lab.removeprefix("__label__") for lab in labels],
                "probs": [float(p) for p in probs],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lid176", type=Path, help="path to lid.176.ftz")
    parser.add_argument(
        "--model", action="append", default=[], help="NAME=PATH of another real model"
    )
    args = parser.parse_args()

    sentences = (FIXTURES / "sentences.txt").read_text(encoding="utf-8").splitlines()
    OUT.mkdir(parents=True, exist_ok=True)
    expected: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "train.txt"
        _write_corpus(corpus)
        for name, overrides in TINY_MODELS.items():
            model = _train(corpus, **overrides)
            model.save_model(str(OUT / name))
            expected[name] = _predictions(model, sentences)
        for name, quant in QUANTIZED_MODELS.items():
            model = _train(corpus, loss="softmax")
            model.quantize(input=str(corpus), dsub=2, retrain=False, **quant)
            model.save_model(str(OUT / name))
            expected[name] = _predictions(model, sentences)

    real = [f"lid176={args.lid176}"] if args.lid176 else []
    for spec in real + args.model:
        name, path = spec.split("=", 1)
        expected[name] = _predictions(fasttext.load_model(path), sentences)

    (OUT / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(expected)} model fixtures to {OUT}")


if __name__ == "__main__":
    main()
