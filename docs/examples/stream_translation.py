"""Translate a JSON Lines file of any size into English, writing each result as it comes.

`translate_stream` reads its input a chunk at a time, translates each chunk like
`translate_batch` and yields the results in input order. Memory holds one chunk, not the
whole file, and the output file grows while the job runs.

Run: uv run python docs/examples/stream_translation.py
Downloads Opus-MT es→en, de→en and fr→en on first run.
"""

import json
import tempfile
from collections.abc import Iterator
from itertools import tee
from pathlib import Path

from tqdm import tqdm

from noentenc import Language
from noentenc.translation import Translator

# A stand-in for a large export of support tickets, one JSON object per line.
TICKETS = [
    {"id": 1, "text": "Hola, ¿cuándo llega mi pedido?"},
    {"id": 2, "text": "Der Link im Newsletter funktioniert nicht."},
    {"id": 3, "text": "Thanks, the issue is fixed now."},
    {"id": 4, "text": "Je n'arrive pas à me connecter à mon compte."},
    {"id": 5, "text": None},
    {"id": 6, "text": "¿Puedo cambiar la dirección de entrega?"},
]
workdir = Path(tempfile.mkdtemp())
source = workdir / "tickets.jsonl"
with source.open("w", encoding="utf-8") as file:
    for ticket in TICKETS:
        file.write(json.dumps(ticket, ensure_ascii=False) + "\n")


def read(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


translator = Translator()

# The stream needs strings, so a null text goes in as "" (blank text comes back as
# given, without a model). `tee` keeps each record next to its translation, holding at
# most the chunk the translator has read ahead.
records, for_texts = tee(read(source))
texts = (record["text"] or "" for record in for_texts)

results = translator.translate_stream(
    texts,
    Language.ENGLISH,
    "auto",  # detect each ticket's language
    chunk_size=256,  # tickets read and translated together
    detailed=True,
    errors="record",  # a ticket that fails comes back as given; the job goes on
)

target = workdir / "tickets_en.jsonl"
with target.open("w", encoding="utf-8") as out:
    # Results arrive a chunk at a time; tqdm counts them as they are written.
    for record, result in tqdm(
        zip(records, results, strict=True), total=len(TICKETS), desc="Translating"
    ):
        row = {
            "id": record["id"],
            "text_en": None if record["text"] is None else result.text,
            "status": str(result.status),
            "source": result.source_language,
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

for line in target.read_text(encoding="utf-8").splitlines():
    print(line)
# {"id": 1, "text_en": "Hey, when does my order arrive?", "status": "translated", "source": "es"}
# {"id": 2, "text_en": "The link in the newsletter does not work.", "status": "translated", "source": "de"}
# {"id": 3, "text_en": "Thanks, the issue is fixed now.", "status": "unchanged", "source": "en"}
# {"id": 4, "text_en": "I can't log in to my account.", "status": "translated", "source": "fr"}
# {"id": 5, "text_en": null, "status": "unchanged", "source": null}
# {"id": 6, "text_en": "Can I change the delivery address?", "status": "translated", "source": "es"}

# Each chunk is its own call, so with "auto" the texts are grouped by language within
# a chunk. In a stream that mixes more languages than `max_loaded_models` (2 by
# default), a larger chunk_size or a higher limit saves reloading models.
