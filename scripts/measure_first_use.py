"""What the README quickstart costs a new user, from install to warm calls.

For each install (`core` runs the detection quickstart, `translation` the mixed-language
translation quickstart), it:

1. creates a fresh virtual environment with uv and installs the wheel into it with uv's
   package cache off, so every package downloads: install time and installed size;
2. runs the quickstart in a fresh process against an empty weights cache: the time from
   the first import to the first result, which includes downloading the weights, and the
   size of what was downloaded;
3. runs it again in a fresh process with the cache filled: the time from the first import
   to the first result, the median latency of further calls in that process, and the
   process's peak resident memory.

Install and first-call times depend on the network, so the environment is printed first.
Needs uv on PATH, and macOS or Linux (peak memory comes from `resource`):

    uv build --wheel
    uv run python scripts/measure_first_use.py dist/noentenc-*.whl
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MB = 1 << 20

# Keep both quickstarts in step with the README.
_DETECTION = """
import time
start = time.perf_counter()
from noentenc.language_detection import LanguageDetector

detector = LanguageDetector()
detector.detect("Bon dia! Com estàs?")
first = time.perf_counter() - start

# Sentences it hasn't seen, so fastText's word cache doesn't flatter the warm latency.
TEXTS = [
    "The train to Barcelona leaves at nine.",
    "Morgen fahren wir mit dem Zug nach Berlin.",
    "Je voudrais un café, s'il vous plaît.",
    "¿Dónde está la estación de autobuses?",
    "Il libro è sul tavolo della cucina.",
    "Vou ao mercado comprar frutas amanhã.",
    "Ik woon al tien jaar in Amsterdam.",
    "Мы встретимся завтра у библиотеки.",
    "今日はとても暑いですね。",
    "Jag tycker om att läsa böcker på kvällen.",
    "Huomenna menemme junalla Helsinkiin.",
    "Dziękuję za pomoc z tym projektem.",
]
warm = []
for text in TEXTS:
    tick = time.perf_counter()
    detector.detect(text)
    warm.append(time.perf_counter() - tick)
"""

_TRANSLATION = """
import time
start = time.perf_counter()
from noentenc.translation import Translator

INBOX = [
    "Hola, ¿cuándo llega mi pedido?",
    "Der Link funktioniert nicht.",
    "Thanks, it works now.",
]
translator = Translator()
translator.translate_batch(INBOX, "en", "auto")
first = time.perf_counter() - start

warm = []
for _ in range(5):
    tick = time.perf_counter()
    translator.translate_batch(INBOX, "en", "auto")
    warm.append(time.perf_counter() - tick)
"""

_REPORT = """
import json, resource, statistics, sys
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({
    "first": first,
    "warm": statistics.median(warm),
    "peak": peak if sys.platform == "darwin" else peak * 1024,
}))
"""

QUICKSTARTS = {
    "core": ("{wheel}", _DETECTION),
    "translation": ("noentenc[translation] @ {wheel}", _TRANSLATION),
}


def _size(path: Path) -> int:
    # The Hugging Face cache links each snapshot file to a blob; count the blobs once.
    return sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()
    )


def _environment() -> None:
    uv = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
    print(f"- {platform.platform()}, {platform.machine()}, {os.cpu_count()} CPUs")
    print(f"- Python {platform.python_version()}, {uv.stdout.strip()}")


def _quickstart(python: Path, code: str, cache: Path) -> dict[str, float]:
    env = {**os.environ, "NOENTENC_CACHE": str(cache), "HF_HOME": str(cache / "hf")}
    env["TQDM_DISABLE"] = "1"
    result = subprocess.run(
        [str(python), "-I", "-c", code + _REPORT],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode:
        raise SystemExit(
            f"the quickstart failed ({result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def measure(install: str, wheel: Path, workdir: Path) -> None:
    requirement, code = QUICKSTARTS[install]
    venv = workdir / install
    subprocess.run(["uv", "venv", "--quiet", str(venv)], check=True)
    python = venv / "bin" / "python"
    tick = time.perf_counter()
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--no-cache",
            "--python",
            str(python),
            requirement.format(wheel=wheel.resolve()),
        ],
        check=True,
    )
    install_seconds = time.perf_counter() - tick
    (site_packages,) = venv.glob("lib/python*/site-packages")

    cache = workdir / f"{install}-weights"
    cold = _quickstart(python, code, cache)
    downloaded = _size(cache)
    cached = _quickstart(python, code, cache)
    warm = cached["warm"]
    warm_text = f"{warm * 1e6:.0f} µs" if warm < 1e-3 else f"{warm * 1e3:.0f} ms"
    print(
        f"| {install} | {install_seconds:.1f} s | {_size(site_packages) / MB:.0f} MB "
        f"| {cold['first']:.1f} s | {downloaded / MB:.1f} MB "
        f"| {cached['first']:.2f} s | {warm_text} | {cached['peak'] / MB:.0f} MB |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="the noentenc wheel to install")
    parser.add_argument(
        "--installs", nargs="+", choices=list(QUICKSTARTS), default=list(QUICKSTARTS)
    )
    args = parser.parse_args()
    if sys.platform == "win32":
        parser.error("needs macOS or Linux")
    _environment()
    print()
    print(
        "| Install | Install time | Installed | First call | Downloaded "
        "| Cached start-up | Warm call | Peak RAM |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    with tempfile.TemporaryDirectory() as workdir:
        for install in args.installs:
            measure(install, args.wheel, Path(workdir))


if __name__ == "__main__":
    main()
