"""Benchmark every installed language-detection backend.

Usage:
    uv run python scripts/benchmark_lid.py                     # backends whose weights are local
    uv run python scripts/benchmark_lid.py --download          # also fetch missing weights
    uv run python scripts/benchmark_lid.py --models lid176 lingua

Reports load time, median single-text latency, and batched throughput on the multilingual test
sentences repeated to ``--texts`` items. Repetition keeps fastText's per-word cache hot, which
matches real corpora (words repeat); ``FastTextModel(cache_size=0)`` shows the uncached cost.
``--corpus FILE`` measures on your own texts instead, one per line and none repeated, e.g. the
WiLI-2018 test paragraphs used in docs/benchmarks.md.
"""

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from noentenc.language_detection.models import (
    Cld3Model,
    FastTextModel,
    HeliportModel,
    LangidModel,
    LinguaModel,
    OnnxClassifierModel,
)
from noentenc.language_detection.models.base import BaseModel
from noentenc.language_detection.models.fasttext import PRESETS as FASTTEXT
from noentenc.language_detection.models.onnx_classifier import PRESETS as ONNX

ROOT = Path(__file__).resolve().parents[1]
SENTENCES = ROOT / "tests/unit/language_detection/fixtures/sentences.txt"
SINGLE_RUNS = 200


def _factories(only_local: bool) -> dict[str, Callable[[], BaseModel]]:
    factories: dict[str, Callable[[], BaseModel]] = {
        name: (lambda name=name: FastTextModel(name, only_local_files=only_local))
        for name in FASTTEXT
    }
    factories |= {
        name: (lambda name=name: OnnxClassifierModel(name, only_local_files=only_local))
        for name in ONNX
    }
    factories |= {
        "lingua": LinguaModel,
        "lingua-low-accuracy": lambda: LinguaModel(low_accuracy=True),
        "cld3": Cld3Model,
        "heliport": HeliportModel,
        "langid": LangidModel,
    }
    return factories


def _throughput(model: BaseModel, texts: list[str], batch_size: int) -> float:
    start = time.perf_counter()
    model.predict_batch(texts, batch_size=batch_size)
    return len(texts) / (time.perf_counter() - start)


def _single_latency_us(model: BaseModel, sentences: list[str]) -> float:
    timings = []
    for i in range(SINGLE_RUNS):
        text = sentences[i % len(sentences)]
        start = time.perf_counter()
        model.predict(text)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings) * 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="subset of backends to run")
    parser.add_argument(
        "--texts", type=int, default=20_000, help="number of texts for throughput"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--download", action="store_true", help="download missing weights"
    )
    parser.add_argument(
        "--corpus", type=Path, help="text file, one text per line, used without repeats"
    )
    args = parser.parse_args()

    fixtures = SENTENCES.read_text(encoding="utf-8").splitlines()
    if args.corpus:
        lines = args.corpus.read_text(encoding="utf-8").splitlines()
        texts = lines[: args.texts]
        # Time single texts on lines the throughput run hasn't seen (and cached).
        sentences = lines[args.texts : args.texts + SINGLE_RUNS] or texts
        warm_up = fixtures
    else:
        sentences = fixtures
        texts = (sentences * (args.texts // len(sentences) + 1))[: args.texts]
        warm_up = texts[:256]
    factories = _factories(only_local=not args.download)
    names = args.models or list(factories)

    header = f"{'model':<22}{'load s':>9}{'single µs':>12}{'batch texts/s':>15}"
    print(header)
    print("-" * len(header))
    for name in names:
        try:
            start = time.perf_counter()
            model = factories[name]()
            load = time.perf_counter() - start
        except (ImportError, FileNotFoundError) as exc:
            print(f"{name:<22}skipped: {str(exc).splitlines()[0][:80]}")
            continue
        n = (
            len(texts)
            if "xlm" not in name and "bert" not in name
            else min(len(texts), 2_000)
        )
        _throughput(model, warm_up, args.batch_size)
        batch = _throughput(model, texts[:n], args.batch_size)
        single = _single_latency_us(model, sentences)
        print(f"{name:<22}{load:>9.2f}{single:>12.1f}{batch:>15,.0f}")


if __name__ == "__main__":
    main()
