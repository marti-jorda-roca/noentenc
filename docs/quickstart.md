# Quickstart

This page walks through the basics: detect a language, translate text, and choose a model when the default doesn't fit.

## Install

noentenc needs Python 3.14 or newer.

```bash
uv add noentenc
```

The core install has the fastText, ONNX and langid detection backends and every translation model. Weights download the first time you use a model, so expect the first call to be slow. After that they load from the cache.

## Detect a language

```python
from noentenc.language_detection import LanguageDetector

detector = LanguageDetector()

detector.detect("Bon dia! Com estàs?")
# 'cat'
```

`LanguageDetector()` uses fastText lid.176, which knows 176 languages and weighs 0.9 MB. It returns ISO 639-3 codes like `eng`, `spa` and `cat`. Empty and whitespace-only texts return `und` (undetermined).

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

## Translate

```python
from noentenc import Language
from noentenc.translation import Translator

translator = Translator()

translator.translate("The weather is nice today.", Language.SPANISH, Language.ENGLISH)
# 'El tiempo es bueno hoy.'
```

The arguments are the text, the target language and the source language, in that order. Languages are always `Language` enum members, so a typo fails where you wrote it instead of deep inside a model.

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

## Detect, then translate

The detector returns ISO 639-3 codes (`deu`), but `Language` values use ISO 639-1 where a two-letter code exists (`de`). Build a lookup once with `to_iso639_3`.

```python
from noentenc import Language
from noentenc.language_detection import LanguageDetector, to_iso639_3
from noentenc.translation import Translator

BY_ISO639_3 = {to_iso639_3(language): language for language in Language}

text = "Der Link im Newsletter funktioniert nicht."
source = BY_ISO639_3.get(LanguageDetector().detect(text))  # Language.GERMAN

if source is not None:
    print(Translator().translate(text, Language.ENGLISH, source))
```

`.get` returns `None` for `und` and for languages no translation model covers, so check before translating. [translate_to_english.py](examples/translate_to_english.py) does this for a whole inbox, grouping texts by language so each group goes through the model in one batch.

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

# A small transformer trained on OpenLID: 201 languages, 25 MB.
LanguageDetector(OnnxClassifierModel("bert-openlid"))

# You already know the text is one of a few languages. Needs `uv add 'noentenc[lingua]'`.
LanguageDetector(LinguaModel(languages=["cat", "spa", "eng"]))
```

Start with the default. It's 0.9 MB, needs nothing beyond numpy, and labels about 400k short sentences per second. Lingua is much slower, but restricting the candidates helps on very short texts, where n-gram models struggle. The [README](../README.md#language-detection) lists every backend with its size and licence.

## Choose a translation model

Pass a model to `Translator` to use it for every call.

```python
from noentenc import Language
from noentenc.translation import M2M100Model, OpusMTModel, Precision, SMaLL100Model, Translator

# One direction, fastest. Check `OPUS_MT_PAIRS` for the 66 available directions.
Translator(OpusMTModel.from_pair(Language.ENGLISH, Language.SPANISH))

# Many directions with one model. The source language is optional.
Translator(SMaLL100Model())

# Better quality than SMaLL-100, but it needs the source language.
Translator(M2M100Model(precision=Precision.INT8))
```

Every translation model also takes `num_threads=`, which is worth capping when several workers share a machine.

If a model can't handle a pair, the call raises `UnsupportedLanguageError`.

```python
from noentenc import UnsupportedLanguageError

english_to_spanish = Translator(OpusMTModel.from_pair(Language.ENGLISH, Language.SPANISH))

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

Every model takes `only_local_files=True`. With it, a model that isn't in the cache raises an error instead of downloading. Run your code once with network access to fill the cache, then deploy with the flag set.

```python
LanguageDetector(FastTextModel(only_local_files=True))
Translator(OpusMTModel.from_pair(Language.ENGLISH, Language.SPANISH, only_local_files=True))
```

Detection weights live in `~/.cache/noentenc`, or wherever `NOENTENC_CACHE` points. Translation weights live in the Hugging Face cache.

## Next steps

- [Examples](../README.md#examples) are short scripts you can run, one per task.
- [Models](../README.md#models) lists every detection and translation model with its size and licence.
- [Custom models](custom-models.md) shows how to load your own weights or wrap another library.
- [Benchmarks](benchmarks.md) has speed numbers for each model and helps you pick one.
