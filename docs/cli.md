# Command line

The `noentenc` command detects languages, translates text, downloads models ahead of time and shows what's cached, without writing Python. It runs the same `LanguageDetector`, `Translator` and `prepare` as the library, on the same profiles and cache.

## Install

The command comes with the `cli` extra. Add `translation` for `noentenc translate`:

```bash
pip install 'noentenc[cli,translation]'  # with pip
uv add 'noentenc[cli,translation]'       # or with uv
uv tool install 'noentenc[cli,translation]'  # or as a standalone tool, on your PATH
```

`noentenc[cli]` alone is enough for `detect`, `list`, `info` and downloading detection models. Without the extra, `noentenc` exits with 1 and prints the install command.

## Detect

```bash
noentenc detect "Bon dia! Com estàs?" "Hello there" "👍" ""
# cat
# eng
# zxx
# und
```

Each text gets one line with its ISO 639-3 code, in input order. `und` means undetermined and `zxx` no linguistic content, as in the library. Texts come from the arguments, from `--file FILE`, or one per line from stdin:

```bash
noentenc detect --file inbox.txt
cat inbox.txt | noentenc detect
```

To answer `und` instead of guessing on short or unclear text, set the same thresholds as `LanguageDetector`: `--min-letters`, `--min-score` and `--min-margin`. `--candidates eng,spa,cat` restricts the answer to those languages. `--json` prints one object per text, with its status and the model's top label and score:

```bash
noentenc detect --json --min-letters 4 --min-score 0.5 "lol" "Bon dia! Com estàs?"
# {"text": "lol", "language": "und", "status": "insufficient_text", "top_language": null, "score": null}
# {"text": "Bon dia! Com estàs?", "language": "cat", "status": "detected", "top_language": "cat", "score": 0.8828022480010986}
```

## Translate

```bash
noentenc translate --to en --file inbox.txt
# Hey, when does my order arrive?
# The link doesn't work.
# Thanks, it works now.
```

By default each text's language is detected (`--from auto`), as with `source_language="auto"`. Texts already in the target language and texts without linguistic content are printed unchanged. Pass `--from es` when you know the source language. Languages are codes or names: `en`, `eng`, `en-US` and `english` all work.

When the detector can't tell a text's language, or no model translates it, the text is printed as given. `--unknown-source fallback` translates such texts without a source language instead, and `--unknown-source raise` stops with exit code 4. `--truncate` translates the start of a sentence that is too long for the model instead of failing the text, and `--no-preserve` stops keeping URLs, emails, code and placeholders as they are.

`--json` adds each text's status, source language, detected language and model:

```bash
noentenc translate --to en --json "Hola, ¿cuándo llega mi pedido?"
# {"text": "Hola, ¿cuándo llega mi pedido?", "translation": "Hey, when does my order arrive?", "input_truncated": false, "output_limit_reached": false, "status": "translated", "error": null, "source_language": "es", "detected_language": "spa", "detection_score": 0.9944202899932861, "model": "OpusMTModel(Xenova/opus-mt-es-en)", "preservation": null}
```

A text that fails to translate doesn't stop the others. Its line holds the original text (`"translation": null` with `--json`), its error goes to stderr, and the command exits with 4 once every text is done.

## Download ahead of time

`noentenc download` runs `noentenc.prepare`: it downloads the profile's detection model and, for each `--pair SOURCE:TARGET`, the translation model the profile picks for it, without loading anything. `detect` and `translate` with the same `--profile` and `--cache-dir` then find them, also with `--offline`:

```bash
# While building the image:
noentenc download --pair es:en --pair de:en --pair :en --cache-dir /models

# In the deployment, without network access:
noentenc translate --to en --offline --cache-dir /models --file inbox.txt
```

`:en` is translation without a source language. `auto:en` downloads every model a detected language could be routed to, which is 7.9 GB for English on `speed`. `--no-detection` skips the detection model, and `--force` downloads everything again.

`--dry-run` prints what would be downloaded, with licences and sizes, and downloads nothing:

```bash
noentenc download --pair es:en --dry-run
# TASK         MODEL                              PRECISION  LICENSE       SIZE    CACHED
# detection    FastTextModel(lid176)              -          CC-BY-SA-3.0  0.9 MB  no
# translation  OpusMTModel(Xenova/opus-mt-es-en)  q4         Apache-2.0    290 MB  no
#
# 291 MB in total, 291 MB not cached yet. Loading every model takes about 1.1 GB of RAM.
```

With `--json` the same report is one JSON object; after a download it also lists the files.

## List models

`noentenc list` shows every model noentenc can use, with its precision, licence, download size, the profiles that load it, whether it's cached and the extra it needs. `--task detection` or `--task translation` narrows it down. It downloads nothing.

```bash
noentenc list --task detection
# TASK       MODEL                                 PRECISION  LICENSE       SIZE    PROFILES  CACHED  EXTRA
# detection  FastTextModel(lid176)                 -          CC-BY-SA-3.0  0.9 MB  speed     yes     -
# detection  FastTextModel(lid176-bin)             -          CC-BY-SA-3.0  125 MB  -         no      -
# ...
# detection  OnnxClassifierModel(bert-openlid)     -          MIT           25 MB   -         no      onnx
# detection  LinguaModel                           -          Apache-2.0    -       -         -       lingua (missing)
```

Translation lists the multilingual models, then one Opus-MT model per pair. A model without profiles is only used when you pass it in Python, as in [Choose a translation model](quickstart.md#choose-a-translation-model). Backends with `-` as size ship their weights in their package. `(missing)` marks an extra that isn't installed.

## Inspect the cache

`noentenc info` shows the version, where models are cached and why (`--cache-dir`, `NOENTENC_CACHE` or the default), the extras installed, and each cached model with its size on disk. It only reads the cache directory.

```bash
noentenc info
# Version:     0.3.0
# Cache root:  /home/me/.cache/noentenc (default)
# Detection:   /home/me/.cache/noentenc
# Translation: /home/me/.cache/noentenc/hub
# Extras:      translation, cli
#
# TASK         MODEL                  SIZE    PATH
# detection    FastTextModel(lid176)  0.9 MB  /home/me/.cache/noentenc/fasttext/lid.176.ftz
# translation  Xenova/opus-mt-es-en   290 MB  /home/me/.cache/noentenc/hub/models--Xenova--opus-mt-es-en
#
# 291 MB in 2 models.
```

## Output

Results go to stdout; download progress bars, warnings and errors go to stderr, so stdout stays parseable. `TQDM_DISABLE=1` hides the progress bars.

- Plain output has one line per text, in input order, so it lines up with the input file.
- `--json` on `detect` and `translate` prints JSON Lines: one object per text, in input order. `list --json` prints one object per model, and `download --json` and `info --json` one object.
- Input and output are UTF-8 whatever the locale. Files and stdin are read one text per line; Windows line endings are fine.

`detect` and `translate` read the input a chunk at a time and print results as each chunk is done, so files of any length work with flat memory.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | Any other error, such as a failed download or a missing extra. |
| 2 | Invalid arguments: an unknown option, profile or language, or a pair no model translates. |
| 3 | A model isn't cached and `--offline` was given. Run `noentenc download` with the same `--profile` and `--cache-dir`. |
| 4 | Some texts failed to translate, or `--unknown-source raise` met a text whose language couldn't be used. |

## Shell pipelines

```bash
# Count the languages in a file.
noentenc detect --file comments.txt | sort | uniq -c | sort -rn

# Put each line next to its language.
paste comments.txt <(noentenc detect --file comments.txt)

# Keep only the Spanish lines.
paste comments.txt <(noentenc detect --file comments.txt) | awk -F'\t' '$2 == "spa" { print $1 }'

# Translate into English and keep every field, for jq or a DataFrame.
noentenc translate --to en --json --file reviews.txt > reviews.en.jsonl
jq -r 'select(.status == "translated") | .translation' reviews.en.jsonl

# Fail a build when a model is missing from the image.
noentenc detect --offline --cache-dir /models "smoke test" > /dev/null || exit 1
```
