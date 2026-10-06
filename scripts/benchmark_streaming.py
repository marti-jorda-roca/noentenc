"""Memory and speed of the streaming methods against the batch methods.

- `memory`: peak memory added by detecting or translating N texts, for growing N,
  with `detect_batch`/`translate_batch` (a list) and `detect_stream`/`translate_stream`
  (a generator). Each run is a fresh process and the model is loaded before measuring, so
  only memory that grows with the input counts. Translation inference is replaced by an
  echo of the input tokens, which keeps the runs short and leaves out onnxruntime's
  buffers (they depend on `batch_size`, not on N); tokenizing, sorting, decoding and the
  results are real.
- `speed`: sentences per second translating a corpus with `translate_batch` and with
  `translate_stream` at several chunk sizes, in alternating rounds (the median is shown).
  Streaming sorts sentences by length within each chunk only, so small chunks pad more.

Uses Opus-MT en->es and fastText lid176 from the noentenc cache, downloading them if
missing. Needs psutil:

    uv run --with psutil python scripts/benchmark_streaming.py memory
    uv run python scripts/benchmark_streaming.py speed --corpus english.txt

`--corpus` is a text file with one English sentence per line.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable

MB = 1 << 20
DETECTION_SIZES = (100_000, 1_000_000)
TRANSLATION_SIZES = (20_000, 200_000)
CHUNK_SIZES = (64, 256, 1024)

_PRELUDE = """
import ctypes, json, os, sys, threading, warnings
import psutil
warnings.simplefilter("ignore")
from noentenc import Language as L

_info = (ctypes.c_uint64 * 32)()

def used():
    # On macOS, the physical footprint: resident memory leaves out compressed pages.
    if sys.platform == "darwin":
        # proc_pid_rusage(pid, RUSAGE_INFO_V2): ri_phys_footprint is the 10th uint64.
        ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 2, _info)
        return _info[9]
    return psutil.Process().memory_info().rss

def texts(n):
    for i in range(n):
        yield f"Sentence {i} says the weather is nice today, so we are going to the beach."

def measure(run):
    # Sampled, because the process's own peak (ru_maxrss) includes loading the model.
    before = used()
    highest = before
    done = threading.Event()

    def sample():
        nonlocal highest
        while not done.wait(0.002):
            highest = max(highest, used())

    sampler = threading.Thread(target=sample)
    sampler.start()
    run()
    done.set()
    sampler.join()
    highest = max(highest, used())
    print(json.dumps({"peak": highest - before}), flush=True)
    # Skip interpreter teardown: onnxruntime sometimes aborts in it after a large run.
    os._exit(0)
"""

_DETECTION = """
from noentenc.language_detection import LanguageDetector
detector = LanguageDetector()
detector.detect_batch(list(texts(1000)))
if {stream}:
    measure(lambda: sum(1 for _ in detector.detect_stream(texts({n}))))
else:
    measure(lambda: detector.detect_batch(list(texts({n}))))
"""

_TRANSLATION = """
from noentenc.translation import OpusMTModel, Translator

class Echo:
    def generate(self, input_ids, attention_mask, config):
        return [row[mask == 1].tolist() for row, mask in zip(input_ids, attention_mask)]

model = OpusMTModel.from_pair(L.ENGLISH, L.SPANISH)
model._engine = Echo()
translator = Translator(model)
translator.translate_batch(list(texts(1000)), L.SPANISH, L.ENGLISH)
if {stream}:
    stream = translator.translate_stream(texts({n}), L.SPANISH, L.ENGLISH)
    measure(lambda: sum(1 for _ in stream))
else:
    measure(lambda: translator.translate_batch(list(texts({n})), L.SPANISH, L.ENGLISH))
"""


def _peak(code: str) -> float:
    result = subprocess.run(
        [sys.executable, "-c", _PRELUDE + code],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])["peak"]


def memory(repeat: int) -> None:
    print(f"Median of {repeat} runs. Peak RAM added on top of the loaded model.")
    print("| Task | Texts | Batch (list) | Stream (generator) |")
    print("|---|---:|---:|---:|")
    for task, code, sizes in (
        ("Detection", _DETECTION, DETECTION_SIZES),
        ("Translation", _TRANSLATION, TRANSLATION_SIZES),
    ):
        for n in sizes:
            batch, stream = (
                statistics.median(
                    _peak(code.format(n=n, stream=stream)) for _ in range(repeat)
                )
                for stream in (False, True)
            )
            print(f"| {task} | {n:,} | {batch / MB:.0f} MB | {stream / MB:.0f} MB |")


def speed(corpus: str, rounds: int) -> None:
    from noentenc import Language
    from noentenc.translation import OpusMTModel, Translator

    with open(corpus, encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]
    translator = Translator(OpusMTModel.from_pair(Language.ENGLISH, Language.SPANISH))
    es, en = Language.SPANISH, Language.ENGLISH
    translator.translate_batch(sentences[:64], es, en)  # warm up

    runs: dict[str, list[float]] = {}

    def timed(name: str, run: Callable[[], object]) -> None:
        start = time.perf_counter()
        run()
        runs.setdefault(name, []).append(len(sentences) / (time.perf_counter() - start))

    # Alternating rounds, so thermal throttling hits every configuration alike.
    for _ in range(rounds):
        timed("translate_batch", lambda: translator.translate_batch(sentences, es, en))
        for size in CHUNK_SIZES:
            timed(
                f"translate_stream, chunk_size={size}",
                lambda size=size: list(
                    translator.translate_stream(sentences, es, en, chunk_size=size)
                ),
            )
    print(f"{len(sentences)} sentences, median of {rounds} rounds.")
    print("| Method | Sentences/s |")
    print("|---|---:|")
    for name, rates in runs.items():
        print(f"| `{name}` | {statistics.median(rates):.1f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["memory", "speed"])
    parser.add_argument("--repeat", type=int, default=3, help="memory: runs per row")
    parser.add_argument("--rounds", type=int, default=3, help="speed: rounds")
    parser.add_argument("--corpus", help="speed: one English sentence per line")
    args = parser.parse_args()
    if args.what == "memory":
        memory(args.repeat)
    elif not args.corpus:
        parser.error("speed needs --corpus")
    else:
        speed(args.corpus, args.rounds)


if __name__ == "__main__":
    main()
