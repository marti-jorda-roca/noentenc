"""RAM taken by each translation model, and peak RAM of a multilingual `Translator` workload.

Every measurement runs in a fresh process, so models don't share memory between rows.

- `models`: resident memory added by loading one model and translating a sentence with it.
- `workload`: a `Translator()` cycling twice through 8 Opus-MT pairs and one SMaLL-100
  pair, for several `max_loaded_models` limits: peak resident memory, final resident memory
  and wall time (which includes reloading evicted models).

Uses the Hugging Face cache, downloading the models it lacks. Needs psutil:

    uv run --with psutil python scripts/measure_translation_memory.py models
    uv run --with psutil python scripts/measure_translation_memory.py workload
"""

import argparse
import json
import statistics
import subprocess
import sys

MB = 1 << 20
SENTENCE = "The weather is nice today, so we are going to the beach."

MODELS = {
    "Opus-MT en→es q4": "OpusMTModel.from_pair(L.ENGLISH, L.SPANISH, precision='q4')",
    "Opus-MT en→es int8": "OpusMTModel.from_pair(L.ENGLISH, L.SPANISH, precision='int8')",
    "Opus-MT en→es fp32": "OpusMTModel.from_pair(L.ENGLISH, L.SPANISH, precision='fp32')",
    "SMaLL-100 int8": "SMaLL100Model()",
    "M2M100 418M int8": "M2M100Model()",
    "NLLB-200 600M int8": "NLLBModel(precision='int8')",
    "NLLB-200 600M fp32": "NLLBModel(precision='fp32')",
}

# (source, target) pairs; Catalan→English has no Opus-MT model, so it uses SMaLL-100.
PAIRS = [
    ("de", "en"),
    ("es", "en"),
    ("zh", "en"),
    ("en", "es"),
    ("en", "de"),
    ("en", "fi"),
    ("en", "zh"),
    ("fr", "de"),
    ("ca", "en"),
]

_PRELUDE = """
import json, resource, sys, time, warnings
import psutil
warnings.simplefilter("ignore")
from noentenc import Language as L
from noentenc.translation import M2M100Model, NLLBModel, OpusMTModel, SMaLL100Model, Translator
# Imported up front, so the libraries themselves aren't counted as model memory.
import onnxruntime, tokenizers, huggingface_hub

def peak():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024

process = psutil.Process()
"""

_MODEL = """
before = process.memory_info().rss
model = {build}
model.predict({sentence!r}, L.SPANISH, L.ENGLISH)
# Peak is the process's highest resident memory minus what it used before the model.
print(json.dumps({{"rss": process.memory_info().rss - before, "peak": peak() - before}}))
"""

_WORKLOAD = """
limit = {limit}
translator = Translator(max_loaded_models=limit)
start = time.perf_counter()
for _ in range(2):
    for source, target in {pairs!r}:
        translator.translate({sentence!r}, L(target), L(source))
print(json.dumps({{
    "peak": peak(),
    "rss": process.memory_info().rss,
    "seconds": time.perf_counter() - start,
    "loaded": len(translator.loaded_models),
}}))
"""


def _run(code: str) -> dict[str, float]:
    result = subprocess.run(
        [sys.executable, "-c", _PRELUDE + code],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _median(code: str, repeat: int) -> dict[str, float]:
    runs = [_run(code) for _ in range(repeat)]
    return {key: statistics.median(run[key] for run in runs) for key in runs[0]}


def models(repeat: int) -> None:
    print(f"Median of {repeat} runs.")
    print("| Model | RAM after load and one sentence | Peak |")
    print("|---|---:|---:|")
    for name, build in MODELS.items():
        stats = _median(_MODEL.format(build=build, sentence=SENTENCE), repeat)
        print(f"| {name} | {stats['rss'] / MB:.0f} MB | {stats['peak'] / MB:.0f} MB |")


def workload(repeat: int) -> None:
    print(
        f"{len(PAIRS)} pairs, cycled twice. One sentence per call, so the time is mostly loading."
    )
    print("| `max_loaded_models` | Peak RAM | RAM at the end | Models kept | Time |")
    print("|---:|---:|---:|---:|---:|")
    for limit in (1, 2, 4, None):
        stats = _median(
            _WORKLOAD.format(limit=limit, pairs=PAIRS, sentence=SENTENCE), repeat
        )
        print(
            f"| {limit} | {stats['peak'] / MB:.0f} MB | {stats['rss'] / MB:.0f} MB "
            f"| {stats['loaded']:.0f} | {stats['seconds']:.1f} s |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["models", "workload"])
    parser.add_argument("--repeat", type=int, default=3, help="runs per row (median)")
    args = parser.parse_args()
    {"models": models, "workload": workload}[args.what](args.repeat)


if __name__ == "__main__":
    main()
