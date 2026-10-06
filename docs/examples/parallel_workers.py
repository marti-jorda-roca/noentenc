"""Translate the shards of a large job in parallel processes, each with its own cores.

By default onnxruntime gives every model all of the machine's physical cores, so two
processes translating at once fight over the same cores. `num_threads` caps each
process's translator to its share.

Run: uv run python docs/examples/parallel_workers.py
Downloads Opus-MT en→es on first run. Each worker loads its own copy (about 1.1 GB of RAM).
"""

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

from noentenc import Language
from noentenc.translation import Translator

WORKERS = 2
THREADS = max(1, (os.cpu_count() or WORKERS) // WORKERS)

SENTENCES = [
    "Where is the station?",
    "I love this city.",
    "The meeting was moved to Friday.",
    "Please restart the server before running the migration.",
    "Our quarterly revenue grew by twelve percent.",
    "Can you send me the invoice again?",
]


@cache
def translator() -> Translator:
    """This worker's translator, built the first time the worker needs it."""
    return Translator(num_threads=THREADS)


def translate_file(paths: tuple[Path, Path]) -> int:
    """Stream one shard into its output file, line by line. Returns the line count."""
    source, target = paths
    count = 0
    with (
        source.open(encoding="utf-8") as lines,
        target.open("w", encoding="utf-8") as out,
    ):
        texts = (line.rstrip("\n") for line in lines)
        for translation in translator().translate_stream(
            texts, Language.SPANISH, Language.ENGLISH
        ):
            out.write(translation + "\n")
            count += 1
    return count


def main() -> None:
    # Split the input into one shard per worker. With real data, the shards are often
    # files that already exist, such as the parts of an export.
    workdir = Path(tempfile.mkdtemp())
    shards = []
    for i in range(WORKERS):
        source = workdir / f"part-{i}.txt"
        source.write_text("\n".join(SENTENCES[i::WORKERS]) + "\n", encoding="utf-8")
        shards.append((source, workdir / f"part-{i}.es.txt"))

    print(f"{WORKERS} workers with {THREADS} threads each")
    with ProcessPoolExecutor(WORKERS) as pool:
        print(list(pool.map(translate_file, shards)))  # [3, 3]

    for _, target in shards:
        print(target.read_text(encoding="utf-8").splitlines())
    # ['¿Dónde está la estación?', 'La reunión se trasladó al viernes.',
    #  'Nuestros ingresos trimestrales crecieron un doce por ciento.']
    # ['Me encanta esta ciudad.',
    #  'Por favor, reinicie el servidor antes de ejecutar la migración.',
    #  '¿Puedes enviarme la factura de nuevo?']


# Process pools start workers by importing this file, so the work only starts here.
if __name__ == "__main__":
    main()
