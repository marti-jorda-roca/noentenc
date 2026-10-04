<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.svg">
    <img src="docs/assets/logo-light.svg" alt="noentenc" width="520">
  </picture>
  <p>Language detection and machine translation for Python, on CPU, without torch.</p>
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

- **Fast.** The default detector labels about 350,000 texts per second on a laptop CPU, and a direct Opus-MT translation takes about 60 ms per sentence. See [benchmarks](docs/benchmarks.md).

  | Task | Model | Single input | Batched |
  |---|---|---:|---:|
  | Detection | heliport | 4 µs | 872k texts/s |
  | Detection | **lid176 (default)** | 17 µs | 356k texts/s |
  | Detection | cld3 | 16 µs | 66k texts/s |
  | Detection | lingua | 0.44 ms | 9k texts/s |
  | Detection | bert-openlid | 0.35 ms | 3.5k texts/s |
  | Translation (en→es) | **Opus-MT (default for direct pairs)** | 63 ms | 100 sent/s |
  | Translation (en→es) | **SMaLL-100 (default otherwise)** | 52 ms | 75 sent/s |
  | Translation (en→es) | M2M100 418M | 231 ms | 15 sent/s |
  | Translation (en→es) | NLLB-200 600M (int8) | 770 ms | 10 sent/s |

  Apple M3, batches of 256 texts for detection and 32 sentences for translation.

- **Small.** The default detection model is 0.9 MB. noentenc has no torch, transformers or GPU dependency, only numpy, onnxruntime, tokenizers, huggingface-hub and tqdm.
- **One API, many models.** 12 detection models and 4 translation model families sit behind the same two classes. Swapping one is a one-line change, and every detector returns the same ISO 639-3 labels.
- **Built for datasets.** You can pass a single string, a list or a pandas or polars column.

## Install

noentenc uses [uv](https://docs.astral.sh/uv/) to manage dependencies.

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

## Examples

| Example | Shows how to |
|---|---|
| [detect_dataset.py](docs/examples/detect_dataset.py) | Tag a DataFrame column and keep only confident English rows. |
| [translate_to_english.py](docs/examples/translate_to_english.py) | Detect each message's language, then batch-translate everything into English. |
| [choose_detection_backend.py](docs/examples/choose_detection_backend.py) | Swap backends, restrict candidate languages, collapse macrolanguages. |
| [choose_translation_model.py](docs/examples/choose_translation_model.py) | Pick a model, precision and thread count, and run offline. |
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
| `M2M100Model` | 100 ↔ 100 | 1.2 GB (q4), 603 MB (int8) | MIT |
| `NLLBModel` | 196 ↔ 196 | 860 MB (int8) | CC-BY-NC-4.0, never picked by default |

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

`scripts/benchmark_lid.py` and `scripts/benchmark_translation.py` reproduce the speed numbers. `scripts/make_fasttext_fixtures.py` regenerates the fastText parity fixtures with the reference `fasttext` package. That package needs Python 3.12, and the script's docstring has the command. `scripts/make_langid_fixtures.py` does the same for langid with the reference `langid` package.
