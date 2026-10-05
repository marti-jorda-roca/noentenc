# Quickstart

This page walks through the basics: detect a language, translate text, and choose a model when the default doesn't fit.

## Install

noentenc needs Python 3.11 or newer.

```bash
uv add 'noentenc[translation]'
```

That installs detection and every translation model. If you only detect languages, `uv add noentenc` is enough: it has the fastText and langid backends and needs only numpy. Weights download the first time you use a model, so expect the first call to be slow. After that they load from the cache.

## Detect a language

```python
from noentenc.language_detection import LanguageDetector

detector = LanguageDetector()

detector.detect("Bon dia! Com estàs?")
# 'cat'
```

`LanguageDetector()` uses fastText lid.176, which knows 176 languages and weighs 0.9 MB. It returns ISO 639-3 codes like `eng`, `spa` and `cat`. Empty and whitespace-only texts return `und` (undetermined). Texts without letters outside URLs and email addresses (digits, emoji, punctuation, a bare link) return `zxx` (no linguistic content). Neither runs the model.

To see how sure the model is, ask for scores. `top_k` sets how many candidates come back.

```python
detector.detect("Bon dia! Com estàs?", with_score=True, top_k=2)
# {'cat': 0.88, 'por': 0.11}
```

For many texts, pass a list to `detect_batch`.

```python
detector.detect_batch(["Hello there", "Hola, ¿qué tal?", "你好", ""])
# ['eng', 'spa', 'zho', 'und']
```

### Abstain when unsure

Any other text gets the model's best guess, however unsure it is: `lol` comes back as `eng` and `Hans Müller` as `deu`. To route only confident detections, set thresholds. A text that misses one gets `und`, and `detailed=True` says why.

```python
detector = LanguageDetector(min_letters=4, min_score=0.5)

detector.detect("lol", detailed=True).status  # DetectionStatus.INSUFFICIENT_TEXT

detection = detector.detect("Hans Müller", detailed=True)
detection.language, detection.status  # ('und', DetectionStatus.LOW_CONFIDENCE)
detection.top_language, detection.score  # ('deu', 0.35): the guess it didn't trust
```

- `min_letters`: fewer letters than this (URLs and emails don't count) is `insufficient_text`. The model doesn't run.
- `min_score`: a top score below this is `low_confidence`.
- `min_margin`: a top score less than this above the runner-up is `ambiguous`.

Scores aren't probabilities you can compare across backends, so pick thresholds per backend. [Benchmarks](benchmarks.md#conservative-detection) has tested settings with their wrong-label and abstention rates. `detect_dataset` writes each row's status to `status_column=` if you pass one.

If you know which languages to expect, `candidates=["eng", "spa", "cat"]` makes the detector answer one of them. It works with the fastText, ONNX, langid and lingua backends; CLD3 and heliport raise, because they don't score every language.

## Translate

```python
from noentenc import Language
from noentenc.translation import Translator

translator = Translator()

translator.translate("The weather is nice today.", Language.SPANISH, Language.ENGLISH)
# 'El tiempo es bueno hoy.'
```

The arguments are the text, the target language and the source language, in that order. Languages are `Language` members, or codes and names that mean one: `"es"`, `"spa"`, `"es-ES"` and `"spanish"` all work. An unknown one raises `UnsupportedLanguageError` before any model loads.

The source language is optional.

```python
translator.translate("Bon dia a tothom!", Language.ENGLISH)
# 'Good day to everyone!'
```

Pass it anyway when you know it. `Translator()` picks the lightest model for each pair, and it can only pick a dedicated Opus-MT model (about 75M parameters, 66 directions) when it knows the source. Without a source it falls back to SMaLL-100, a 595 MB model that covers 100 languages.

`translate_batch` translates a list in one call.

```python
translator.translate_batch(
    ["Where is the station?", "I love this city."], Language.SPANISH, Language.ENGLISH
)
# ['¿Dónde está la estación?', 'Me encanta esta ciudad.']
```

### Long text

Emails and documents can go in whole. The models are trained on single sentences, so `Translator` splits the text into sentences, translates them in one batch and joins the translations back with the original whitespace. Line breaks and blank lines stay where they were.

```python
translator.translate(
    "Hi Ana,\n\nThanks for your help.\n- See you on Monday.\n- Call me later.",
    Language.SPANISH,
    Language.ENGLISH,
)
# 'Hola Ana,\n\nGracias por tu ayuda.\n- Nos vemos el lunes.\n- Llámame más tarde.'
```

A single sentence longer than the model can read (about 500 tokens) raises `InputTooLongError` rather than losing its end. `truncate=True` translates its start instead. To find out whether that happened, or whether a translation stopped at the output length limit, ask for details.

```python
result = translator.translate(
    email_body, Language.SPANISH, Language.ENGLISH, truncate=True, detailed=True
)
result.text
result.input_truncated  # True if part of a sentence was dropped
result.output_limit_reached  # True if the output may be cut short
```

`translate_batch` takes the same `truncate` and `detailed` arguments, and `translate_dataset` takes `truncate`.

Blank texts, and texts whose source language is already the target, come back unchanged without loading a model. Their detailed status is `unchanged`.

### Memory

A `Translator` loads a model the first time a pair needs it and keeps it for later calls. It keeps at most two by default. When a pair needs a third, the least recently used model is dropped first, and it's loaded again if a later call needs it. Each one takes 0.6 to 1.2 GB of RAM for Opus-MT and SMaLL-100, and up to 4 GB for NLLB-200, far more than its download.

```python
translator = Translator(
    max_loaded_models=4
)  # fewer reloads, more memory; None for no limit
translator.loaded_models  # the models it holds right now
translator.unload()  # free them; the next call loads what it needs again
```

### Keep links and placeholders

Machine translation models translate everything they see, including the parts that must not change. Opus-MT turns `https://example.com/reset` into `https://ejemplo.com/reset`, which is a different site. So before a text reaches the model, `Translator` swaps these literals for placeholders the models copy through, and puts them back afterwards:

- URLs (`https://…`, `www.…`) and bare domains with a lowercase ending (`acme.io/help`). A sentence's full stop or an unmatched closing bracket after a URL isn't part of it.
- Email addresses, `@mentions` and `#hashtags`.
- Numbers, including decimals, thousands separators, times and dates (`1,299.99`, `14:30`, `12/05/2026`).
- Inline code in backticks.
- Template placeholders: `{name}`, `{0}`, `{{name}}`, `${name}`, `%s`, `%d` and `%(name)s`.
- HTML and XML tags and character references (`<a href="…">`, `</b>`, `&amp;`).
- Markdown link targets, `](https://…)`. The link text is translated.

```python
translator.translate(
    "Hi {name}, your code is 4821. Reset it at https://acme.io/r.", "es", "en"
)
# 'Hola {name}, tu código es 4821. Reestablecerlo en https://acme.io/r.'
```

If the translation drops, repeats or changes a placeholder, the text is translated again in pieces. The prose between the literals is translated on its own and the literals are kept between the pieces, so they always survive, though the prose can read a little less fluently. The detailed result's `preservation` says which path was taken: `"placeholders"`, `"segments"`, or `None` when the text had no literals. Pass `preserve=False` to send texts to the model untouched.

Limits:

- Only the literals above are protected. Markdown emphasis (`**bold**`), lists and headings, and the text inside HTML tags, are translated like any other text, and a model may move or drop the markup around them.
- Words can move across tags. German Opus-MT turns `Click <a>here</a> to…` into `Klicken Sie hier <a></a>, um…`. The tags are intact, but the link now wraps nothing.
- Protected numbers keep their source format: `1,299.99` stays `1,299.99` in Spanish rather than becoming `1.299,99`.
- This isn't a document translator. Long HTML or Markdown documents work best when you translate their text nodes yourself.

[Benchmarks](benchmarks.md#literal-text) has how often each model keeps literals with and without preservation.

### Keep going past bad rows

By default the first text that fails to translate raises, which suits scripts you watch. For bulk jobs pass `errors="record"`. A text that fails comes back as given with status `failed` and the error message, and the rest of the batch is still translated. Errors that apply to the whole call still raise: an invalid `batch_size`, an unknown language or an unsupported pair.

```python
from noentenc.translation import TranslationStatus

results = translator.translate_batch(
    texts, Language.ENGLISH, Language.SPANISH, detailed=True, errors="record"
)
failed = {
    i: result.error
    for i, result in enumerate(results)
    if result.status is TranslationStatus.FAILED
}

df = translator.translate_dataset(
    df, "text", "text_en", Language.ENGLISH, errors="record", error_column="error"
)
# Failed rows have a null "text_en" and the message in "error".
```

## Detect, then translate

For a mixed-language inbox, pass `source_language="auto"`. Each text's language is detected, and texts in the same language go through the right model together. Results come back in input order.

```python
messages = [
    "Hola, ¿cuándo llega mi pedido?",
    "Der Link im Newsletter funktioniert nicht.",
    "Thanks, the issue is fixed now.",
    "lol",
]
translator.translate_batch(messages, Language.ENGLISH, "auto")
# ['Hey, when does my order arrive?', 'The link in the newsletter does not work.',
#  'Thanks, the issue is fixed now.', 'lol']
```

- Pairs with an Opus-MT model use it, and the other languages use the profile's fallback (SMaLL-100 for `speed`).
- Texts already in the target language, blank texts, and texts without linguistic content (a bare link, emoji) come back unchanged, without a model.
- The detector is the profile's `LanguageDetector`, set to abstain with `min_letters=4, min_score=0.5`. When it can't tell the language, the text comes back as given with status `unknown_source`. If it detects a language no model translates into the target, the status is `unsupported_source`. Pass `unknown_source="fallback"` to translate those texts without a source language, or `unknown_source="raise"` to fail the call with `SourceLanguageError` (from `noentenc.translation`) before anything is translated. The error names the first such text by index. To use other thresholds or another backend, pass `Translator(detector=LanguageDetector(...))`.
- `detailed=True` gives each text's `source_language`, `detected_language`, `detection_score`, `status` and `model`.
- Each model loads once per call and respects `max_loaded_models`, including in `translate_dataset`, which also takes `status_column=` and `source_column=`.

```python
df = translator.translate_dataset(
    df, "message", "english", Language.ENGLISH, "auto", status_column="status"
)
```

`noentenc.prepare(translation=[("auto", "en")])` downloads every model that can be needed for a target, for running offline. [translate_to_english.py](examples/translate_to_english.py) runs a whole inbox and a DataFrame.

To map a detector label to a `Language` yourself, use `to_language`. It accepts ISO 639-1 and 639-3 codes, BCP-47 tags and English names. It folds individual languages into their macrolanguage when only that is supported, so `"cmn"` becomes `Language.CHINESE` and `"nob"` becomes `Language.NORWEGIAN`.

```python
from noentenc import to_language

to_language("deu")  # Language.GERMAN
to_language("und", None)  # None instead of UnsupportedLanguageError
```

## Trade speed for quality

Pass a profile instead of a model and noentenc picks one for you. The three profiles are `"speed"`, `"balance"` and `"quality"`, and `Profile.SPEED`, `Profile.BALANCE` and `Profile.QUALITY` work too. A misspelled profile raises `ValueError`.

```python
from noentenc import Profile
from noentenc.language_detection import LanguageDetector
from noentenc.translation import Translator

LanguageDetector("quality")
Translator(Profile.BALANCE)
```

- **`speed`** is the default, so `LanguageDetector()` and `Translator()` already use it. It answers fastest and downloads least.
- **`balance`** is much more accurate on languages beyond the most common ones, and still fast enough for large datasets.
- **`quality`** gives the best output this package has. It is the slowest and downloads the most.

| Profile | Detection | Translation, pairs without an Opus-MT model |
|---|---|---|
| `speed` | fastText `lid176`: 0.9 MB, 176 languages | SMaLL-100: 595 MB, 100 languages |
| `balance` | fastText `openlid-v3`: 1.2 GB, 195 languages, GPL-3.0 | NLLB-200 600M at int8: 860 MB, 196 languages, CC-BY-NC-4.0 |
| `quality` | fastText `glotlid`: 1.7 GB, 2102 languages | NLLB-200 600M at fp32: 3.5 GB, 196 languages, CC-BY-NC-4.0 |

Every translation profile uses the dedicated Opus-MT model when the pair has one (66 directions). On those pairs it scored as well as NLLB-200 on average and is about 8× faster. Without a source language, Opus-MT and NLLB-200 can't be used, so every profile uses SMaLL-100.

`openlid-v3` and `glotlid` return individual languages: Swahili comes back as `swh`, and Congo Swahili as `swc`, where `lid176` says `swa`. To get one code per macrolanguage, pass the model yourself with `collapse_macrolanguages=True`.

Check the licences before you use `balance` or `quality` commercially. NLLB-200 is non-commercial (CC-BY-NC-4.0) and warns when it loads, and `openlid-v3` is GPL-3.0. [Benchmarks](benchmarks.md#profiles) has the accuracy and speed of each choice, and [choose_profile.py](examples/choose_profile.py) runs all three profiles.

The three profiles often pick the same translation model. Opus-MT pairs use Opus-MT in every profile, and texts without a source language use SMaLL-100 in every profile. So `quality` only changes the output of other pairs. [Routing](benchmarks.md#routing) has the full table.

### Plan, and set limits

To see what a profile would download before it does, ask for a plan. It loads and downloads nothing.

```python
import noentenc

plan = noentenc.plan("balance", translation=[("ja", "ca")])
[(m.name, m.license, m.download_bytes >> 20, m.memory_bytes >> 20) for m in plan.models]
# [('FastTextModel(openlid-v3)', 'GPL-3.0', 1175, 1142),
#  ('NLLBModel(Xenova/nllb-200-distilled-600M)', 'CC-BY-NC-4.0', 869, 4057)]
plan.missing_bytes  # what still has to download
```

`memory_bytes` is resident memory once loaded. `memory_basis` says whether that was measured (see [Memory](benchmarks.md#memory)) or estimated. `Translator(...).plan(target, source)` does the same for one pair, and `LanguageDetector(...).plan()` for a detector. `Translator.supports(target, source)` and `Translator.supported_languages()` say which pairs a profile covers.

To keep a profile within your licence policy or download budget, pass `allowed_licenses` (SPDX identifiers) and `max_download_bytes`. A profile then skips a model that breaks them for its next choice. NLLB-200 gives way to SMaLL-100, and Opus-MT q4 gives way to the smaller Opus-MT int8. If nothing is left, the call raises `ModelConstraintError` before downloading.

```python
from noentenc.translation import Translator

translator = Translator("quality", allowed_licenses=["MIT", "Apache-2.0", "CC-BY-4.0"])
translator.translate(
    "こんにちは", "ca", "ja"
)  # SMaLL-100, not the non-commercial NLLB-200
```

`LanguageDetector`, `noentenc.plan` and `noentenc.prepare` take the same two arguments. With `source_language="auto"`, a text whose pair has no allowed model gets status `unsupported_source`.

The models behind a profile may change between releases. If you need the same output every time, pass a model explicitly as shown below.

## Choose a detection model

Pass a model to `LanguageDetector`. Every backend returns the same ISO 639-3 labels, so you can swap one for another without touching the rest of your code.

```python
from noentenc.language_detection import (
    FastTextModel,
    LanguageDetector,
    LinguaModel,
    OnnxClassifierModel,
)

# Rare and low-resource languages: GlotLID knows 2102 (1.7 GB download).
LanguageDetector(FastTextModel("glotlid"))

# A small transformer trained on OpenLID: 201 languages, 25 MB. Needs `noentenc[onnx]`.
LanguageDetector(OnnxClassifierModel("bert-openlid"))

# You already know the text is one of a few languages. Needs `uv add 'noentenc[lingua]'`.
LanguageDetector(LinguaModel(languages=["cat", "spa", "eng"]))
```

Start with the default. It's 0.9 MB, needs nothing beyond numpy, and labels about 400k short sentences per second. Lingua is much slower, but restricting the candidates helps on very short texts, where n-gram models struggle. The [README](../README.md#language-detection) lists every backend with its size and licence.

## Choose a translation model

Pass a model to `Translator` to use it for every call.

```python
from noentenc import Language
from noentenc.translation import (
    M2M100Model,
    OpusMTModel,
    Precision,
    SMaLL100Model,
    Translator,
)

# One direction, fastest. Check `OPUS_MT_PAIRS` for the 66 available directions.
Translator(OpusMTModel.from_pair(Language.ENGLISH, Language.SPANISH))

# Many directions with one model. The source language is optional.
Translator(SMaLL100Model())

# About as accurate as SMaLL-100, better into Chinese and Japanese but worse on
# low-resource languages, and about 4x slower. It needs the source language.
Translator(M2M100Model())

# Pick a precision explicitly: fp32, int8 or q4, where the export has it.
Translator(
    OpusMTModel.from_pair(Language.ENGLISH, Language.GERMAN, precision=Precision.FP32)
)
```

Every translation model also takes `num_threads=`, which is worth capping when several workers share a machine.

If a model can't handle a pair, the call raises `UnsupportedLanguageError`.

```python
from noentenc import UnsupportedLanguageError

english_to_spanish = Translator(
    OpusMTModel.from_pair(Language.ENGLISH, Language.SPANISH)
)

try:
    english_to_spanish.translate("Hello", Language.FRENCH)
except UnsupportedLanguageError as error:
    print(error)  # OpusMTModel cannot translate into <Language.FRENCH: 'fr'>
```

## Work with DataFrames

Both classes add a column to a pandas or polars DataFrame and return the new frame.

```python
df = detector.detect_dataset(df, "text", "lang")
df = translator.translate_dataset(df, "text", "text_en", Language.ENGLISH)
```

Null cells get `und` from the detector and stay null in the translation. Both show a progress bar, which `show_progress=False` turns off.

## Run offline

Inference never touches the network. Only downloading weights does, and that happens the first time a model is used. To deploy somewhere without network access, download the weights ahead of time with `prepare`, then point the same classes at them with `only_local_files=True`.

```python
import noentenc
from noentenc import Language
from noentenc.language_detection import LanguageDetector
from noentenc.translation import Translator

# On a machine with network access, e.g. while building the image:
noentenc.prepare(
    "speed",
    translation=[(Language.ENGLISH, Language.SPANISH), (None, Language.ENGLISH)],
    cache_dir="/models",
)

# In the deployment, with /models mounted or copied:
detector = LanguageDetector(only_local_files=True, cache_dir="/models")
translator = Translator(only_local_files=True, cache_dir="/models")
```

`prepare` downloads the profile's detection model (`detection=False` skips it) and, for each `(source, target)` pair, the model that profile's `Translator` would pick. A `None` source is translation without a source language. It doesn't load any of them. With `only_local_files=True`, a model that isn't there raises `FileNotFoundError` straight away, and the message says what to prepare. Models you build yourself take the same `only_local_files` and `cache_dir` arguments.

One directory holds both kinds of weights. The cache root is the `cache_dir` you pass, else `NOENTENC_CACHE`, else `~/.cache/noentenc`. Detection weights go directly under it and translation weights under `<root>/hub`, in the Hugging Face cache layout. If you set neither `cache_dir` nor `NOENTENC_CACHE`, translation weights stay in the Hugging Face cache (`HF_HUB_CACHE` or `HF_HOME`), as in earlier releases, so models you already downloaded are reused. If you start setting `NOENTENC_CACHE`, translation models download once more into the new location.

Downloads show a progress bar (`TQDM_DISABLE=1` hides it). Detection downloads give up on a connection that stays silent for 30 seconds (`NOENTENC_DOWNLOAD_TIMEOUT` changes it), retry up to 4 times on timeouts, dropped connections and server errors, and resume where they stopped. Translation downloads use huggingface-hub, which has its own retries and its own `HF_HUB_DOWNLOAD_TIMEOUT`.

If a file gets corrupted, run `prepare` again: it checks the checksummed detection files and downloads any that don't match. `prepare(..., force=True)` downloads everything again, translation models included. Deleting the file and running `prepare` works too.

## Next steps

- [Examples](../README.md#examples) are short scripts you can run, one per task.
- [Models](../README.md#models) lists every detection and translation model with its size and licence.
- [Custom models](custom-models.md) shows how to load your own weights or wrap another library.
- [Benchmarks](benchmarks.md) has speed numbers for each model and helps you pick one.
