"""Benchmark the translation models at each precision.

Usage:
    uv run python scripts/benchmark_translation.py                       # Opus-MT en->es, all precisions
    uv run python scripts/benchmark_translation.py --models small100 m2m100
    uv run python scripts/benchmark_translation.py --models nllb --precisions int8 q4

Reports load time, median single-sentence latency and batched throughput (sentences/s) on a
fixed set of English sentences translated into Spanish. Weights are downloaded on first use.
"""

import argparse
import statistics
import time
import warnings
from collections.abc import Callable

from noentenc.languages import Language
from noentenc.translation import (
    M2M100Model,
    NLLBModel,
    OpusMTModel,
    Precision,
    SMaLL100Model,
)
from noentenc.translation.models._seq2seq import Seq2SeqModel

SENTENCES = [
    "Hello world, how are you?",
    "The weather is nice today.",
    "I would like to order a coffee and two croissants, please.",
    "The European Central Bank kept interest rates unchanged on Thursday, citing persistent "
    "uncertainty about inflation and slowing growth across the euro area.",
    "Please restart the server before running the migration.",
    "She has been learning to play the piano for three years.",
    "Can you tell me where the nearest train station is?",
    "Our quarterly revenue grew by twelve percent compared to last year.",
]
SINGLE_RUNS = 10
BATCH_SIZE = 32
BATCH_TEXTS = 128

MODELS: dict[str, Callable[[Precision], Seq2SeqModel]] = {
    "opus-mt": lambda p: OpusMTModel.from_pair(
        Language.ENGLISH, Language.SPANISH, precision=p
    ),
    "small100": lambda p: SMaLL100Model(precision=p),
    "m2m100": lambda p: M2M100Model(precision=p),
    "nllb": lambda p: NLLBModel(precision=p),
}


def bench(name: str, precision: Precision) -> None:
    started = time.perf_counter()
    model = MODELS[name](precision)
    load = time.perf_counter() - started

    model.predict(SENTENCES[0], Language.SPANISH, Language.ENGLISH)  # warm-up
    latencies = []
    for i in range(SINGLE_RUNS):
        started = time.perf_counter()
        model.predict(SENTENCES[i % len(SENTENCES)], Language.SPANISH, Language.ENGLISH)
        latencies.append(time.perf_counter() - started)

    texts = (SENTENCES * (BATCH_TEXTS // len(SENTENCES) + 1))[:BATCH_TEXTS]
    started = time.perf_counter()
    outputs = model.predict_batch(texts, Language.SPANISH, Language.ENGLISH, BATCH_SIZE)
    throughput = len(texts) / (time.perf_counter() - started)

    print(
        f"{name:<9} {precision:<5} load {load:5.1f}s  "
        f"single {statistics.median(latencies) * 1000:6.0f} ms  "
        f"batch{BATCH_SIZE} {throughput:6.1f} sent/s  | {outputs[3][:70]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=["opus-mt"]
    )
    parser.add_argument(
        "--precisions", nargs="+", type=Precision, default=list(Precision)
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="NLLB-200 is licensed")
    for name in args.models:
        for precision in args.precisions:
            try:
                bench(name, precision)
            except ValueError as error:  # e.g. SMaLL-100 only ships int8 weights
                print(f"{name:<9} {precision:<5} skipped: {error}")


if __name__ == "__main__":
    main()
