# noentenc documentation

noentenc detects languages and translates text in Python, on CPU, without torch. This folder holds the guides, the benchmarks and runnable examples. The [project README](../README.md) is the short tour; start there if you haven't seen the library before.

## Start here

1. Install the package with translation support, with pip or uv. It needs Python 3.11 or newer.

   ```bash
   pip install 'noentenc[translation]'  # with pip
   uv add 'noentenc[translation]'       # or with uv
   ```

2. Detect a language, and translate a mixed-language batch into English:

   ```python
   from noentenc.language_detection import LanguageDetector
   from noentenc.translation import Translator

   LanguageDetector().detect("Bon dia! Com estàs?")
   # 'cat'

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

   The first run downloads the weights it needs: 0.9 MB for detection and 554 MB for two Opus-MT models, Spanish and German into English. [What the first call costs](quickstart.md#what-the-first-call-costs) has the timings.

3. Read the [quickstart](quickstart.md). It covers detection, translation, DataFrames and streaming on the defaults, then profiles, models, memory and offline use.

4. Run an [example](#examples) close to your task and change it to fit.

## Guides

| Page | Read it to |
|---|---|
| [quickstart.md](quickstart.md) | Learn the API: detect, translate, abstain when unsure, handle long text, keep links and placeholders, pick a profile or a model, work with DataFrames, stream large inputs and run offline. |
| [benchmarks.md](benchmarks.md) | See what the first install and call cost, compare models on speed, memory and accuracy, choose detection thresholds, and see why each profile uses the models it does. |
| [custom-models.md](custom-models.md) | Load your own weights, wrap another detector or translator, add an ONNX encoder-decoder, or contribute a model to noentenc. |

The project README has the [model tables](../README.md#models) with each model's size and licence, and the [extras](../README.md#install) each backend needs.

## Find a topic

| I want to | Go to |
|---|---|
| Install only what I need | [Install](../README.md#install) |
| Know what the first call downloads and how long it takes | [What the first call costs](quickstart.md#what-the-first-call-costs), [First use](benchmarks.md#first-use) |
| Return `und` instead of guessing on short or unclear text | [Abstain when unsure](quickstart.md#abstain-when-unsure), [Conservative detection](benchmarks.md#conservative-detection) |
| Translate texts in several languages in one call | [Translate](quickstart.md#translate) |
| Translate emails and documents | [Long text](quickstart.md#long-text) |
| Keep URLs, emails, code and placeholders intact | [Keep links and placeholders](quickstart.md#keep-links-and-placeholders) |
| Keep a bulk job going past bad rows | [Keep going past bad rows](quickstart.md#keep-going-past-bad-rows) |
| Trade speed for quality | [Trade speed for quality](quickstart.md#trade-speed-for-quality), [Profiles benchmark](benchmarks.md#profiles) |
| Check downloads, RAM and licences before loading anything | [Plan, and set limits](quickstart.md#plan-and-set-limits) |
| Pick a specific detection or translation model | [Choose a detection model](quickstart.md#choose-a-detection-model), [Choose a translation model](quickstart.md#choose-a-translation-model) |
| Bound memory in a long-running service | [Memory](quickstart.md#memory), [Memory benchmark](benchmarks.md#memory) |
| Tag or translate a pandas or polars column | [Work with DataFrames](quickstart.md#work-with-dataframes) |
| Process a file or stream too large to hold in memory | [Stream large inputs](quickstart.md#stream-large-inputs), [Streaming benchmark](benchmarks.md#streaming) |
| Share a machine's cores between several workers | [Threads](quickstart.md#threads) |
| Run without network access | [Run offline](quickstart.md#run-offline) |
| Use my own model | [Custom models](custom-models.md) |

## Examples

Each script in [`examples/`](examples) solves one task and runs as is from the repository root:

```bash
uv run python docs/examples/detect_dataset.py
```

The docstring at the top of each script says which extras it needs and what it downloads on first run.

**Detection**

| Script | What it shows |
|---|---|
| [detect_dataset.py](examples/detect_dataset.py) | Tag a DataFrame column and keep only confident English rows. |
| [conservative_detection.py](examples/conservative_detection.py) | Abstain on names, chat tokens and links instead of guessing, and restrict the candidate languages. |
| [choose_detection_backend.py](examples/choose_detection_backend.py) | Swap backends, restrict candidate languages, collapse macrolanguages. |
| [stream_detection.py](examples/stream_detection.py) | Label every line of a large file with flat memory, and stop reading once you have what you need. |

**Translation**

| Script | What it shows |
|---|---|
| [translate_to_english.py](examples/translate_to_english.py) | Translate a mixed-language inbox and DataFrame into English in one call with `source_language="auto"`. |
| [translate_long_text.py](examples/translate_long_text.py) | Translate emails and documents, and choose between an error and `truncate=True` for over-long sentences. |
| [preserve_literals.py](examples/preserve_literals.py) | Translate messages without breaking their links, emails, code, tags and placeholders. |
| [translate_bulk.py](examples/translate_bulk.py) | Run bulk jobs with `errors="record"` so one bad row doesn't stop them. |
| [choose_translation_model.py](examples/choose_translation_model.py) | Pick a model, precision and thread count. |
| [manage_memory.py](examples/manage_memory.py) | Bound how many translation models stay loaded, and free them with `unload()`. |
| [stream_translation.py](examples/stream_translation.py) | Translate a JSON Lines file of any size with `translate_stream`, writing each result as it comes. |
| [parallel_workers.py](examples/parallel_workers.py) | Split a job across processes and give each translator its share of the cores with `num_threads`. |

**Models, profiles and deployment**

| Script | What it shows |
|---|---|
| [choose_profile.py](examples/choose_profile.py) | Trade latency for quality with the `speed`, `balance` and `quality` profiles. |
| [plan_and_limit.py](examples/plan_and_limit.py) | See what a profile would download and what it costs, and restrict it by licence and download size. |
| [prepare_offline.py](examples/prepare_offline.py) | Download a profile's weights into one directory, then detect and translate without network access. |
| [custom_models.py](examples/custom_models.py) | Plug your own detector and translator into the same API. |

## Contributing to the docs

Docs live next to the code and change in the same pull request as the feature they describe. When you change behaviour, update the quickstart section and the example that cover it, and run the example to check its output comments still match. To reproduce the benchmark numbers, see the commands at the top of [benchmarks.md](benchmarks.md). [CONTRIBUTING.md](../CONTRIBUTING.md) covers the development setup.
