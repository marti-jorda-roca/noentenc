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
from noentenc import Language
from noentenc.language_detection import LanguageDetector
from noentenc.translation import Translator

LanguageDetector().detect("Bon dia! Com estàs?")
# 'cat'

Translator().translate("The weather is nice today.", Language.SPANISH, Language.ENGLISH)
# 'El tiempo es bueno hoy.'
```

- **Fast.** Detects a language in as little as 2.5 µs and translates a sentence in about 52 ms, on a laptop CPU. See [benchmarks](docs/benchmarks.md).
- **Faster than the originals.** Up to 470× faster than `langid.py`, 2× faster than the C++ `fasttext` package and 5.8× faster than transformers + PyTorch, with the same accuracy. See [the comparison](docs/benchmarks.md#against-the-original-implementations).
- **Small.** The default detection model is 0.9 MB. noentenc has no torch, transformers or GPU dependency, only numpy, onnxruntime, tokenizers, huggingface-hub and tqdm.
- **One API, many models.** 12 detection models and 4 translation model families sit behind the same two classes. Swapping one is a one-line change, and every detector returns the same ISO 639-3 labels.
- **Built for datasets.** You can pass a single string, a list or a pandas or polars column.

New to noentenc? The [quickstart](docs/quickstart.md) covers detection, translation and choosing a model on one page.

## Install

noentenc uses [uv](https://docs.astral.sh/uv/) to manage dependencies.

Requires Python 3.14 or newer. Install with pip or uv:

```bash
pip install noentenc
pip install 'noentenc[all]'
```

```bash
uv add noentenc                  # fastText, ONNX and langid detection + every translation model
uv add 'noentenc[lingua,cld3]'   # extra detection backends, see the table below
uv add 'noentenc[all]'
```

Weights download on first use to `~/.cache/noentenc` (detection) and the Hugging Face cache (translation). Nothing is bundled in the wheel.

## Detect a language

```python
from noentenc.language_detection import LanguageDetector

detector = LanguageDetector()  # fastText lid.176: 176 languages, 0.9 MB

detector.detect("Bon dia! Com estàs?")
# 'cat'
detector.detect("Bon dia! Com estàs?", with_score=True, top_k=2)
# {'cat': 0.88, 'por': 0.11}
detector.detect_batch(["Hello there", "Hola, ¿qué tal?", "你好", ""])
# ['eng', 'spa', 'zho', 'und']

# Add a "lang" column to a pandas or polars DataFrame.
detector.detect_dataset(df, "text", "lang")
```

## Translate

```python
from noentenc import Language
from noentenc.translation import Translator

translator = Translator()

translator.translate_batch(
    ["Where is the station?", "I love this city."], Language.SPANISH, Language.ENGLISH
)
# ['¿Dónde está la estación?', 'Me encanta esta ciudad.']

# The source language is optional.
translator.translate("Bon dia a tothom!", Language.ENGLISH)
# 'Good day to everyone!'

translator.translate_dataset(df, "review", "review_en", Language.ENGLISH)
```

`Translator()` picks the lightest model for each pair. It uses a dedicated Opus-MT model when one exists for the direction (66 directions, about 75M parameters each). For any other pair it uses SMaLL-100, which covers 100 languages. Languages are always `Language` enum members, so a typo fails at the call site, and a model that can't handle a pair raises `UnsupportedLanguageError`.

Text of any length works. It is split into sentences, they are translated in one batch, and the translations are joined back with the original spaces, line breaks and blank lines. A single sentence longer than the model can read (about 500 tokens) raises `InputTooLongError` instead of being cut silently. Pass `truncate=True` to translate only its start, and `detailed=True` to get a `Translation` that says whether input was dropped (`input_truncated`) or the output hit its length limit (`output_limit_reached`).

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

## Examples

| Example | Shows how to |
|---|---|
| [detect_dataset.py](docs/examples/detect_dataset.py) | Tag a DataFrame column and keep only confident English rows. |
| [translate_to_english.py](docs/examples/translate_to_english.py) | Detect each message's language, then batch-translate everything into English. |
| [choose_detection_backend.py](docs/examples/choose_detection_backend.py) | Swap backends, restrict candidate languages, collapse macrolanguages. |
| [choose_translation_model.py](docs/examples/choose_translation_model.py) | Pick a model, precision and thread count, and run offline. |
| [translate_long_text.py](docs/examples/translate_long_text.py) | Translate emails and documents, and choose between an error and `truncate=True` for over-long sentences. |
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
| `OnnxClassifierModel` | `bert-openlid` (int8) | core | 201 | 25 MB | MIT |
| | `xlm-roberta-lid` (int8) | core | 20 | 279 MB | MIT |
| `LangidModel` | | core | 97 | 1.9 MB | BSD-2-Clause |
| `LinguaModel` | | `noentenc[lingua]` | 75 | ~300 MB wheel | Apache-2.0 |
| `Cld3Model` | | `noentenc[cld3]` | 107 | 1 MB | Apache-2.0 |
| `HeliportModel` | | `noentenc[heliport]` (no Windows) | 220 | ~130 MB wheel | GPL-3.0 |

```python
from noentenc.language_detection import FastTextModel, LanguageDetector, LinguaModel

LanguageDetector(FastTextModel("glotlid"))  # 2102 languages
LanguageDetector(LinguaModel(languages=["cat", "spa", "eng"]))  # only these candidates
```

- Labels are ISO 639-3 codes (`eng`, `cat`, `zho`). Empty and whitespace-only texts return `und`.
  - `normalize_labels=False` returns each model's native codes instead.
  - `collapse_macrolanguages=True` folds individual languages into their macrolanguage (`arb` → `ara`, `cmn` → `zho`), so results from different models line up.
- `FastTextModel` is our own numpy implementation of fastText inference. It needs neither the `fasttext` package nor onnxruntime, and it matches `fasttext`'s output to within 1e-6. Large `.bin` models are memory-mapped, so they open instantly.
- `LangidModel` is our own numpy implementation of [langid.py](https://github.com/saffsd/langid.py). It reads the weights from the langid 1.1.6 source release on PyPI, without installing or importing the `langid` package, and matches its probabilities to within 1e-9.

### Translation

| Model | Languages | Download (default precision) | Licence (weights) |
|---|---|---|---|
| `OpusMTModel.from_pair(src, tgt)` | 1 direction each, 66 available (`OPUS_MT_PAIRS`) | 287 MB (q4), 107 MB (int8) | CC-BY-4.0 or Apache-2.0, per pair |
| `SMaLL100Model` | any source → 100 targets | 595 MB (int8) | MIT |
| `M2M100Model` | 100 ↔ 100 | 603 MB (int8) | MIT |
| `NLLBModel` | 196 ↔ 196 | 860 MB (int8) | CC-BY-NC-4.0, used by the `balance` and `quality` profiles |

Every translation model takes `precision=` (`fp32`, `int8`, `q4`, where the export has it), `num_threads=` and a Hugging Face repo id or local directory as `model=`.

### Weights and licences

Weights are pinned to a revision and, where we download them ourselves, checked against a sha256. Set `NOENTENC_CACHE` to change the detection cache location. With `only_local_files=True`, a model that isn't cached raises instead of downloading, which is what you want on an air-gapped server.

The weights' licences apply to your use of the weights. They don't affect this package's licence.

## Bring your own model

Load your own fastText, ONNX classifier or seq2seq weights into an existing backend, or wrap any detector or translator by subclassing a base class with a few methods. See [docs/custom-models.md](docs/custom-models.md).

```python
from noentenc.language_detection import FastTextModel, LanguageDetector

LanguageDetector(FastTextModel("models/my-domain-lid.ftz"))
```

## Development

```bash
make install             # uv sync with every extra + dev tools
make lint
make unit-tests
make integration-tests   # tests/integrations; downloads real weights
```

`scripts/benchmark_lid.py` and `scripts/benchmark_translation.py` reproduce the speed numbers, and `scripts/benchmark_vs_reference.py` the comparison with the original implementations. `scripts/make_fasttext_fixtures.py` regenerates the fastText parity fixtures with the reference `fasttext` package. That package needs Python 3.12, and the script's docstring has the command. `scripts/make_langid_fixtures.py` does the same for langid with the reference `langid` package.
