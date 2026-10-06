<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.svg">
    <img src="docs/assets/logo-light.svg" alt="noentenc" width="520">
  </picture>
  <p>Language detection and machine translation for Python, on CPU, without torch.</p>
  <p>
    <a href="https://github.com/marti-jorda-roca/noentenc/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/marti-jorda-roca/noentenc/ci.yml?style=flat-square&amp;branch=main&amp;cacheSeconds=300" /></a>
    <a href="https://pypi.org/project/noentenc/"><img alt="PyPI" src="https://img.shields.io/pypi/v/noentenc?style=flat-square&amp;cacheSeconds=300" /></a>
    <a href="https://pypi.org/project/noentenc/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/noentenc?style=flat-square&amp;cacheSeconds=300" /></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/marti-jorda-roca/noentenc?style=flat-square" /></a>
  </p>
</div>

```python
from noentenc.language_detection import LanguageDetector
from noentenc.translation import Translator

LanguageDetector().detect("Bon dia! Com estàs?")
# 'cat'

# Detect each text's language and translate it into English.
Translator().translate_batch(
    [
        "Hola, ¿cuándo llega mi pedido?",
        "Der Link funktioniert nicht.",
        "Thanks, it works now.",
    ],
    "en",
    "auto",
)
# ['Hey, when does my order arrive?', "The link doesn't work.", 'Thanks, it works now.']
```

- **Fast on the defaults.** `LanguageDetector()` labels a sentence it hasn't seen in about 0.12 ms, and about 22k paragraphs a second in batches. Once a sentence's words are cached it takes about 15 µs. `Translator()` translates a sentence in about 60 ms with a dedicated Opus-MT model. These numbers are from an Apple M3; see [benchmarks](docs/benchmarks.md).
- **Small, and cheap to start.** Detection installs 23 MB of packages, needs only numpy and tqdm, and its first call downloads a 0.9 MB model in about a second. With translation, the install is 127 MB, and the example above downloads 554 MB of weights on its first run (about 20 s here). After that a new process gives its first result in 1.2 s. There is no torch, transformers or GPU dependency. See [First use](docs/benchmarks.md#first-use).
- **Faster than the originals.** `langid` runs up to 470× faster than `langid.py`. Against the C++ `fasttext` package, `lid176` labels batches of short sentences 2× faster but is 4× slower one text at a time. The ONNX models answer a single text up to 5.8× faster than transformers + PyTorch. Accuracy is the same. See [the comparison](docs/benchmarks.md#against-the-original-implementations).
- **One API, many models.** 12 detection models and 4 translation model families sit behind the same two classes. Swapping one is a one-line change, and every detector returns the same ISO 639-3 labels. The optional `heliport` backend labels a short sentence in 2.5 µs, but it's GPL-3.0 and has no Windows wheel.
- **Built for datasets.** You can pass a single string, a list or a pandas or polars column, or stream a file of any length with flat memory.

New to noentenc? The [quickstart](docs/quickstart.md) covers install, detection and translation on the defaults first, then choosing models, memory and offline use.

## Install

Requires Python 3.11 or newer, on Linux, macOS or Windows. Install with pip or uv:

| To get | pip | uv |
|---|---|---|
| Detection: fastText and langid, with numpy | `pip install noentenc` | `uv add noentenc` |
| Detection and every translation model | `pip install 'noentenc[translation]'` | `uv add 'noentenc[translation]'` |
| The ONNX detection models (bert-openlid, xlm-roberta-lid) | `pip install 'noentenc[onnx]'` | `uv add 'noentenc[onnx]'` |
| The `noentenc` command, see [Command line](#command-line) | `pip install 'noentenc[cli,translation]'` | `uv add 'noentenc[cli,translation]'` |
| Extra detection backends, see [the table below](#language-detection) | `pip install 'noentenc[lingua,cld3]'` | `uv add 'noentenc[lingua,cld3]'` |
| Everything | `pip install 'noentenc[all]'` | `uv add 'noentenc[all]'` |

Python 3.10 isn't supported: onnxruntime 1.30, pandas 3 and the current numpy releases have no Python 3.10 wheels. `noentenc[heliport]` installs nothing on Windows, where heliport has no wheel.

Nothing is bundled in the wheel. Each model downloads the first time it's used, to `~/.cache/noentenc` (`NOENTENC_CACHE` or `cache_dir` move it), and later processes load it from there. The default detector is 0.9 MB, each Opus-MT pair 287 MB and SMaLL-100 595 MB. `noentenc.plan(...)` tells you what a profile would download, and its licences, before it does. To run offline, download the weights ahead of time with `noentenc.prepare(...)` and pass `only_local_files=True`; see [Run offline](docs/quickstart.md#run-offline).

## Detect a language

```python
from noentenc.language_detection import LanguageDetector

detector = LanguageDetector()  # fastText lid.176: 176 languages, 0.9 MB

detector.detect("Bon dia! Com estàs?")
# 'cat'
detector.detect("Bon dia! Com estàs?", with_score=True, top_k=2)
# {'cat': 0.88, 'por': 0.11}
detector.detect_batch(["Hello there", "Hola, ¿qué tal?", "你好", "👍", ""])
# ['eng', 'spa', 'zho', 'zxx', 'und']

# Add a "lang" column to a pandas or polars DataFrame.
detector.detect_dataset(df, "text", "lang")

# Any iterable, such as a file's lines, read a chunk at a time with flat memory.
for lang in detector.detect_stream(line.rstrip("\n") for line in open("comments.txt")):
    ...
```

Text without letters outside URLs and email addresses, such as digits, emoji, punctuation or a bare link, returns `zxx` (no linguistic content) without running the model. Every other text gets the model's best guess, even `lol` or a person's name. To get `und` instead when the model is unsure, set thresholds, and use `detailed=True` to see why:

```python
detector = LanguageDetector(min_letters=4, min_score=0.5)
detector.detect("lol", detailed=True)
# Detection(language='und', status=<DetectionStatus.INSUFFICIENT_TEXT: 'insufficient_text'>, ...)

LanguageDetector(candidates=["eng", "spa", "cat"])  # only ever answer one of these
```

Scores aren't calibrated and differ between backends, so a threshold that suits one backend doesn't suit another. [Benchmarks](docs/benchmarks.md#conservative-detection) lists tested settings and their wrong-label and abstention rates per backend.

## Translate

```python
from noentenc import Language
from noentenc.translation import Translator

translator = Translator()

# A mixed-language inbox: detect each text's language and translate in one call.
translator.translate_batch(
    [
        "Hola, ¿cuándo llega mi pedido?",
        "Der Link funktioniert nicht.",
        "Thanks, it works now.",
    ],
    Language.ENGLISH,
    "auto",
)
# ['Hey, when does my order arrive?', "The link doesn't work.", 'Thanks, it works now.']

translator.translate_dataset(df, "review", "review_en", "en", "auto")

# When you know the source language, pass it instead of "auto".
translator.translate_batch(
    ["Where is the station?", "I love this city."], Language.SPANISH, Language.ENGLISH
)
# ['¿Dónde está la estación?', 'Me encanta esta ciudad.']
```

`source_language="auto"` detects each text's language, groups the texts by language and translates each group with the model for that pair, then returns them in input order. Texts already in the target language and texts without linguistic content come back unchanged. When the detector isn't sure (`lol`, a name), the text is kept as given with status `unknown_source`. `unknown_source="fallback"` translates such texts without a source language instead, and `unknown_source="raise"` fails the call with `SourceLanguageError`. `detailed=True` reports each text's detected language, score, status and model.

`Translator()` picks the lightest model for each pair. It uses a dedicated Opus-MT model when one exists for the direction (66 directions, about 75M parameters each). For any other pair it uses SMaLL-100, which covers 100 languages, and so does a call with no source language at all. A model that can't handle a pair raises `UnsupportedLanguageError`. Languages are `Language` members or codes and names: `"es"`, `"spa"`, `"es-ES"` and `"spanish"` all mean Spanish. `noentenc.to_language()` does that conversion on its own, which also turns detector labels like `"spa"` or `"cmn"` into `Language` members.

Text of any length works. It is split into sentences, they are translated in one batch, and the translations are joined back with the original spaces, line breaks and blank lines. A single sentence longer than the model can read (about 500 tokens) raises `InputTooLongError` instead of being cut silently. Pass `truncate=True` to translate only its start, and `detailed=True` to get a `Translation` that says whether input was dropped (`input_truncated`) or the output hit its length limit (`output_limit_reached`).

`Translator` keeps the models it loads for later calls, at most two at a time by default. A loaded model needs a lot more RAM than its download: about 1.1 GB for an Opus-MT pair at q4 and 1.15 GB for SMaLL-100. Pass `max_loaded_models=` to change the limit (`None` for no limit), and call `translator.unload()` to free them all. See [memory](docs/benchmarks.md#memory).

Links, emails and placeholders survive. URLs, email addresses, mentions, hashtags, numbers, inline code, template placeholders (`{name}`, `{{x}}`, `${x}`, `%s`), HTML tags and Markdown link targets come back byte-for-byte while the prose around them is translated:

```python
translator.translate(
    "Visit https://example.com/reset?id=42 or email support@acme.io", "es", "en"
)
# 'Visite https://example.com/reset?id=42 o envíe un correo electrónico a support@acme.io'
# Without it (preserve=False), Opus-MT writes https://ejemplo.com/reset?id=42: another site.
```

See [Keep links and placeholders](docs/quickstart.md#keep-links-and-placeholders) for what is covered and its limits.

For input too large to hold in memory, `translate_stream` takes any iterable of strings, such as a file's lines or a database cursor. It translates a chunk at a time and yields the results in input order; see [Stream large inputs](docs/quickstart.md#stream-large-inputs). `Translator(num_threads=2)` caps the CPU threads each model uses, for when several workers share a machine.

For bulk jobs, `errors="record"` keeps going past a text that fails. That text comes back as given, with its status and error in the detailed result (or in an `error_column` for `translate_dataset`), and the rest are still translated. Empty batches, blank texts and same-language requests return without loading a model.

## Trade speed for quality

`LanguageDetector` and `Translator` take a profile in place of a model: `"speed"` (the default), `"balance"` or `"quality"`.

```python
from noentenc import Profile
from noentenc.language_detection import LanguageDetector
from noentenc.translation import Translator

LanguageDetector("quality")  # fastText GlotLID
Translator(Profile.BALANCE)  # Opus-MT where it exists, otherwise NLLB-200 at int8
```

| Profile | Detection | Translation, pairs without an Opus-MT model |
|---|---|---|
| `speed` (default) | `lid176`: 0.9 MB | SMaLL-100: 595 MB |
| `balance` | `openlid-v3`: 1.2 GB, GPL-3.0 | NLLB-200 int8: 860 MB, CC-BY-NC-4.0 |
| `quality` | `glotlid`: 1.7 GB | NLLB-200 fp32: 3.5 GB, CC-BY-NC-4.0 |

On FLORES-200, `balance` raises detection accuracy from 50% to 96% over the 176 languages tested, and NLLB-200 adds up to 19 chrF++ on low-resource pairs such as English to Tamil. See [the profiles benchmark](docs/benchmarks.md#profiles) for the numbers behind each choice. The models behind a profile may change between releases, so pass a model explicitly when you need reproducible output.

All three translation profiles use the same Opus-MT model for the 66 pairs it covers, and SMaLL-100 for texts without a source language. They only differ on the other pairs; see [routing](docs/benchmarks.md#routing).

### Check before you download

Ask what a profile would load, what it costs and whether it's allowed, without downloading anything:

```python
import noentenc
from noentenc.translation import Translator

plan = noentenc.plan("balance", translation=[("ja", "ca"), ("en", "es")])
for model in plan.models:
    print(model.name, model.license, model.download_bytes >> 20, "MB", model.cached)
# On a machine that hasn't downloaded anything yet:
# FastTextModel(openlid-v3) GPL-3.0 1175 MB False
# NLLBModel(Xenova/nllb-200-distilled-600M) CC-BY-NC-4.0 869 MB False
# OpusMTModel(Xenova/opus-mt-en-es) Apache-2.0 290 MB False

# Only pick permissively licensed models, and nothing over 700 MB.
translator = Translator(
    "balance",
    allowed_licenses=["MIT", "Apache-2.0", "CC-BY-4.0"],
    max_download_bytes=700 << 20,
)
translator.plan("ca", "ja").models[0].name  # 'SMaLL100Model(casawolice/small100-onnx)'
translator.supports("ace", "en")  # False: only NLLB-200 writes Acehnese
```

Each plan reports the model's licence, download size, RAM (measured, or estimated where marked) and cache state. A pair with no allowed model raises `ModelConstraintError` before anything downloads. `prepare()` takes the same constraints.

## Command line

With the `cli` extra, the `noentenc` command detects and translates text from the shell, one result per input line, and downloads models ahead of time:

```bash
noentenc detect "Bon dia! Com estàs?"
# cat
cat inbox.txt | noentenc translate --to en
noentenc translate --to en --json --file reviews.txt > reviews.en.jsonl
noentenc download --pair es:en --pair :en --cache-dir /models
noentenc list    # every model, its size, licence, profiles and whether it's cached
noentenc info    # where the cache is and what it holds
```

Results go to stdout and progress and errors to stderr, and the exit code says whether a model was missing offline or a text failed. See [docs/cli.md](docs/cli.md).

## Examples

| Example | Shows how to |
|---|---|
| [detect_dataset.py](docs/examples/detect_dataset.py) | Tag a DataFrame column and keep only confident English rows. |
| [conservative_detection.py](docs/examples/conservative_detection.py) | Abstain on names, chat tokens and links instead of guessing, and restrict the candidate languages. |
| [translate_to_english.py](docs/examples/translate_to_english.py) | Translate a mixed-language inbox and DataFrame into English in one call with `source_language="auto"`. |
| [choose_detection_backend.py](docs/examples/choose_detection_backend.py) | Swap backends, restrict candidate languages, collapse macrolanguages. |
| [choose_translation_model.py](docs/examples/choose_translation_model.py) | Pick a model, precision and thread count, and run offline. |
| [prepare_offline.py](docs/examples/prepare_offline.py) | Download a profile's weights into one directory, then detect and translate without network access. |
| [translate_long_text.py](docs/examples/translate_long_text.py) | Translate emails and documents, and choose between an error and `truncate=True` for over-long sentences. |
| [manage_memory.py](docs/examples/manage_memory.py) | Bound how many translation models stay loaded, and free them with `unload()`. |
| [stream_detection.py](docs/examples/stream_detection.py) | Label every line of a large file with flat memory, and stop reading once you have what you need. |
| [stream_translation.py](docs/examples/stream_translation.py) | Translate a JSON Lines file of any size with `translate_stream`, writing each result as it comes. |
| [parallel_workers.py](docs/examples/parallel_workers.py) | Split a job across processes and give each translator its share of the cores with `num_threads`. |
| [plan_and_limit.py](docs/examples/plan_and_limit.py) | See what a profile would download and what it costs, and restrict it by licence and download size. |
| [preserve_literals.py](docs/examples/preserve_literals.py) | Translate messages without breaking their links, emails, code, tags and placeholders. |
| [translate_bulk.py](docs/examples/translate_bulk.py) | Run bulk jobs with `errors="record"` so one bad row doesn't stop them, and skip model loads for empty, blank and same-language input. |
| [choose_profile.py](docs/examples/choose_profile.py) | Trade latency for quality with the `speed`, `balance` and `quality` profiles. |
| [custom_models.py](docs/examples/custom_models.py) | Plug your own detector and translator into the same API. |

## Models

### Language detection

| Backend | `model=` | Install | Languages | Size | Licence (weights) |
|---|---|---|---|---|---|
| `FastTextModel` | `lid176` (default) | core | 176 | 0.9 MB | CC-BY-SA-3.0 |
| | `lid176-bin` | core | 176 | 126 MB | CC-BY-SA-3.0 |
| | `openlid-v2` | core | 200 | 1.2 GB | GPL-3.0 |
| | `openlid-v3` | core | 195 | 1.2 GB | GPL-3.0 |
| | `glotlid` | core | 2102 | 1.7 GB | Apache-2.0 |
| | `nllb-lid218e` | core | 218 | 1.2 GB | CC-BY-NC-4.0 |
| | path to a `.bin`/`.ftz` | core | | | |
| `OnnxClassifierModel` | `bert-openlid` (int8) | `noentenc[onnx]` | 201 | 25 MB | MIT |
| | `xlm-roberta-lid` (int8) | `noentenc[onnx]` | 20 | 279 MB | MIT |
| `LangidModel` | | core | 97 | 1.9 MB | BSD-2-Clause |
| `LinguaModel` | | `noentenc[lingua]` | 75 | ~300 MB wheel | Apache-2.0 |
| `Cld3Model` | | `noentenc[cld3]` | 107 | 1 MB | Apache-2.0 |
| `HeliportModel` | | `noentenc[heliport]` (no Windows) | 220 | ~130 MB wheel | GPL-3.0 |

```python
from noentenc.language_detection import FastTextModel, LanguageDetector, LinguaModel

LanguageDetector(FastTextModel("glotlid"))  # 2102 languages
LanguageDetector(LinguaModel(languages=["cat", "spa", "eng"]))  # only these candidates
```

- Labels are ISO 639-3 codes (`eng`, `cat`, `zho`). Empty and whitespace-only texts return `und`, and texts with no letters outside URLs and email addresses return `zxx`, for every backend and without running it.
  - `normalize_labels=False` returns each model's native codes instead.
  - `collapse_macrolanguages=True` folds individual languages into their macrolanguage (`arb` → `ara`, `cmn` → `zho`), so results from different models line up.
- `FastTextModel` is our own numpy implementation of fastText inference. It needs neither the `fasttext` package nor onnxruntime, and it matches `fasttext`'s output to within 1e-6. Large `.bin` models are memory-mapped, so they open instantly.
- `LangidModel` is our own numpy implementation of [langid.py](https://github.com/saffsd/langid.py). It reads the weights from the langid 1.1.6 source release on PyPI, without installing or importing the `langid` package, and matches its probabilities to within 1e-9.

### Translation

Every translation model needs `noentenc[translation]`.

| Model | Languages | Download (default precision) | Licence (weights) |
|---|---|---|---|
| `OpusMTModel.from_pair(src, tgt)` | 1 direction each, 66 available (`OPUS_MT_PAIRS`) | 287 MB (q4), 107 MB (int8) | CC-BY-4.0 or Apache-2.0, per pair |
| `SMaLL100Model` | any source → 100 targets | 595 MB (int8) | MIT |
| `M2M100Model` | 100 ↔ 100 | 603 MB (int8) | MIT |
| `NLLBModel` | 196 ↔ 196 | 860 MB (int8) | CC-BY-NC-4.0, used by the `balance` and `quality` profiles |

Every translation model takes `precision=` (`fp32`, `int8`, `q4`, where the export has it), `num_threads=` and a Hugging Face repo id or local directory as `model=`.

### Weights and licences

Weights are pinned to a revision and, where we download them ourselves, checked against a sha256. `cache_dir=` or `NOENTENC_CACHE` sets one directory for detection and translation weights. `noentenc.prepare(profile, translation=[(source, target), ...])` downloads a profile's models without loading them. With `only_local_files=True` on `LanguageDetector`, `Translator` or any model, a model that isn't cached raises at once instead of downloading, which is what you want on an air-gapped server.

The weights' licences apply to your use of the weights. They don't affect this package's licence.

## Bring your own model

Load your own fastText, ONNX classifier or seq2seq weights into an existing backend, or wrap any detector or translator by subclassing a base class with a few methods. See [docs/custom-models.md](docs/custom-models.md).

```python
from noentenc.language_detection import FastTextModel, LanguageDetector

LanguageDetector(FastTextModel("models/my-domain-lid.ftz"))
```

## Development

Development uses [uv](https://docs.astral.sh/uv/).

```bash
make install             # uv sync with every extra + dev tools
make lint
make unit-tests
make integration-tests   # tests/integrations; downloads real weights
```

`scripts/benchmark_lid.py` and `scripts/benchmark_translation.py` reproduce the speed numbers, and `scripts/benchmark_vs_reference.py` the comparison with the original implementations. `scripts/measure_first_use.py` measures install, first-call and start-up costs from a built wheel, and `scripts/smoke_quickstart.py` runs the README quickstart against one, as CI does. `scripts/make_fasttext_fixtures.py` regenerates the fastText parity fixtures with the reference `fasttext` package. That package needs Python 3.12, and the script's docstring has the command. `scripts/make_langid_fixtures.py` does the same for langid with the reference `langid` package.
