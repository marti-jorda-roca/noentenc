# Custom models

`LanguageDetector` and `Translator` accept any object that implements their model base class. There are four ways in, from least to most work:

1. [Load your own weights into an existing backend](#1-your-own-weights-in-an-existing-backend). No code.
2. [Wrap any detector](#2-a-new-language-detection-backend) by implementing three methods.
3. [Wrap any translator](#3-a-new-translation-model) by implementing `predict_batch` and declaring a `schema`.
4. [Add an ONNX encoder-decoder](#4-a-new-onnx-encoder-decoder-translation-model) by describing its files and language tokens. The base class does the decoding.

[`examples/custom_models.py`](examples/custom_models.py) runs options 2 and 3 end to end.

## 1. Your own weights in an existing backend

`FastTextModel`, `OnnxClassifierModel` and every translation model take a preset name or a local path as their first argument.

```python
from noentenc.language_detection import (
    FastTextModel,
    LanguageDetector,
    OnnxClassifierModel,
)
from noentenc.translation import M2M100Model, OpusMTModel, Translator

# A fastText model you trained with `fasttext supervised` (.bin or quantized .ftz).
LanguageDetector(FastTextModel("models/support-tickets.ftz"))

# A Hugging Face *ForSequenceClassification model exported to ONNX. The directory needs
# config.json (with id2label), tokenizer.json and model.onnx or onnx/model_quantized.onnx.
LanguageDetector(OnnxClassifierModel("models/my-lid-onnx"))

# Translation models take a Hugging Face repo id or a local directory with config.json,
# tokenizer.json and the ONNX encoder/decoder files listed in the class's `onnx_files`.
Translator(OpusMTModel("Xenova/opus-mt-en-fr"))
Translator(M2M100Model("models/m2m100-finetuned"))
```

Labels from your fastText or ONNX model go through the same ISO 639-3 normalisation as the presets (`__label__en`, `eng_Latn` and `en` all become `eng`). Pass `normalize_labels=False` if your labels aren't language codes.

## 2. A new language-detection backend

Subclass `noentenc.language_detection.BaseModel` and implement:

| Member | Returns |
|---|---|
| `labels` (property) | Every label the model can return. |
| `_predict_chunk(texts)` | The top label for each text. |
| `_predict_score_chunk(texts, top_k)` | One `{label: score}` dict per text, sorted by descending score, with at most `top_k` entries (`None` means all). |

The base class handles the rest: `predict`, `predict_score`, the batched variants, splitting input into `batch_size` chunks and returning results in input order. Empty and whitespace-only texts get `"und"`, and texts without letters outside URLs and email addresses get `"zxx"`, without reaching your model, so the chunk methods only ever see real text. If your `predict_score` does not score every label (it reports only the top spans or label), set `scores_every_label = False` so `LanguageDetector(candidates=...)` refuses it. `batch_size=None`, the default, splits input into `default_batch_size` texts: 32 unless your class sets another value. Raise it when your model has a fixed cost per call.

```python
from noentenc.language_detection import BaseModel, LanguageDetector
from noentenc.language_detection.labels import normalize_scores, to_iso639_3


class MyApiModel(BaseModel):
    def __init__(self, client: MyClient, normalize_labels: bool = True) -> None:
        super().__init__("my-api", normalize_labels=normalize_labels)
        self.client = client

    @property
    def labels(self) -> list[str]:
        return [to_iso639_3(code) for code in self.client.supported_languages()]

    def _predict_chunk(self, texts: list[str]) -> list[str]:
        return [next(iter(scores)) for scores in self._predict_score_chunk(texts, 1)]

    def _predict_score_chunk(
        self, texts: list[str], top_k: int | None
    ) -> list[dict[str, float]]:
        results = []
        for ranked in self.client.detect(texts):  # [[("en", 0.97), ("nl", 0.02)], ...]
            results.append(
                normalize_scores(
                    dict(ranked[:top_k]),
                    normalize=self.normalize_labels,
                    collapse_macrolanguages=self.collapse_macrolanguages,
                )
            )
        return results


detector = LanguageDetector(MyApiModel(client))
```

Rules every backend follows:

- **Return ISO 639-3 labels by default.** `to_iso639_3` and `normalize_scores` convert ISO 639-1, FLORES (`eng_Latn`), BCP-47 (`zh-Latn`) and fastText labels. They also honour `collapse_macrolanguages`.
- **Higher scores mean more likely.** Say in the docstring whether the scores are probabilities.
- **Set `sort_batches_by_length = True` if the model pads each batch to its longest text**, as transformers do. The base class then groups texts of similar length and still returns results in input order.
- **Import optional dependencies lazily** with `noentenc._optional.require("module", "extra")`, so users without the extra get an install hint instead of an `ImportError` at import time.

## 3. A new translation model

Subclass `noentenc.translation.models.base.BaseModel`, declare a `schema` and implement `predict_batch` (`predict` calls it with one text):

```python
from noentenc import ANY_LANGUAGE, Language, LanguageSchema
from noentenc.translation import Translator
from noentenc.translation.models.base import BaseModel


class MyServiceModel(BaseModel):
    # Reads any language and writes these three.
    schema = LanguageSchema(
        source=ANY_LANGUAGE,
        target=frozenset({Language.ENGLISH, Language.SPANISH, Language.FRENCH}),
    )

    def __init__(self, client: MyClient) -> None:
        super().__init__("my-service")
        self.client = client

    def predict_batch(
        self,
        texts: list[str],
        target_language: Language,
        source_language: Language | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        self.schema.validate(source_language, target_language, type(self).__name__)
        out: list[str] = []
        for start in range(0, len(texts), batch_size):
            out += self.client.translate(
                texts[start : start + batch_size], target=str(target_language)
            )
        return out


translator = Translator(MyServiceModel(client))
```

- **`schema` lists what the model reads (`source`) and writes (`target`).** Use `ANY_LANGUAGE` for an unrestricted side. If `source` is a set of more than one language, `validate` makes the source language required.
- **Call `self.schema.validate(...)` first in `predict_batch`.** It raises `UnsupportedLanguageError` for a language the model can't handle, so nothing gets silently mistranslated. `Translator` also checks it before calling you, but direct calls to your model don't go through `Translator`.
- **Languages are always `Language` enum members, never free strings.** `str(Language.SPANISH)` is `"es"` (ISO 639-1, or 639-3 when there's no 639-1 code). Map it to whatever codes your service expects.
- **Return translations in input order, one per text.** `Translator` never passes you an empty batch, blank texts, a same-language request or (in `translate_dataset`) null cells.
- **Expect placeholders such as `ZXQ0` in your input.** `Translator` swaps URLs, emails, code, template placeholders, tags and numbers for them, and puts the originals back in your output. Copy them through unchanged. If your output loses, repeats or alters one, `Translator` calls you again with the prose between those literals, one piece per text, so the literals still survive. If your service protects such text itself, or you need it to see the original text, call the translator with `preserve=False`.
- **Optionally override `predict_batch_detailed`** to report dropped input or cut-short output in a `Translation`, and to honour `truncate`. The default wraps `predict_batch` and reports nothing missing. `Translator` passes your model whole texts; split them yourself if your model only handles short input.

## 4. A new ONNX encoder-decoder translation model

Opus-MT, M2M100, SMaLL-100 and NLLB all subclass `Seq2SeqModel`, which splits text into sentences, tokenizes, encodes, runs a greedy decode with a KV cache, detokenizes and joins the sentences back together. It reads the model's `config.json` for the special token ids and maximum length. A new encoder-decoder family that has an ONNX export (for example one of the `Xenova/*` or `onnx-community/*` repos) usually needs about 30 lines. The class says where the files are and how language tokens wrap the input.

A sketch for mBART-50, which puts the source language token first and forces the target language token as the first generated token:

```python
from pathlib import Path
from typing import ClassVar

from noentenc import Language, LanguageSchema
from noentenc.translation import Precision
from noentenc.translation.models._seq2seq import Seq2SeqModel

MBART50_CODES: dict[Language, str] = {
    Language.ENGLISH: "en_XX",
    Language.SPANISH: "es_XX",
    Language.CHINESE: "zh_CN",
    # ...
}
_LANGUAGES = frozenset(MBART50_CODES)


class MBart50Model(Seq2SeqModel):
    """mBART-50 many-to-many: any pair of 50 languages."""

    schema = LanguageSchema(source=_LANGUAGES, target=_LANGUAGES)
    default_model: ClassVar[str] = "Xenova/mbart-large-50-many-to-many-mmt"
    # The commit of `default_model` to download (see the repo's "History" tab).
    default_revision: ClassVar[str] = "<commit sha>"
    default_precision: ClassVar[Precision] = Precision.INT8

    def _frame(
        self, ids: list[int], source: Language | None, target: Language
    ) -> list[int]:
        if source is None:  # already rejected by `schema`; narrows the type
            raise ValueError("mBART-50 requires a source language")
        return [
            self._token_id(MBART50_CODES[source]),
            *ids,
            self.config["eos_token_id"],
        ]

    def _forced_first_id(self, target: Language) -> int | None:
        return self._token_id(MBART50_CODES[target])
```

The hooks you can override:

| Hook | Default | Override when |
|---|---|---|
| `_frame(ids, source, target)` | `[*ids, eos]` | The model expects language or prefix tokens around the input. |
| `_forced_first_id(target)` | `None` | The decoder must start with a target-language token (M2M100, NLLB, mBART). |
| `_banned_ids()` | `()` | Some token must never be generated (Marian bans `<pad>`). |
| `_load_tokenizer(files)` | `Tokenizer.from_file("tokenizer.json")` | The tokenizer needs patching (see `OpusMTModel`, `M2M100Model`). |
| `onnx_files` | `XENOVA_ONNX_FILES` (fp32, int8, q4) | The repo names its (encoder, decoder) files differently (see `SMaLL100Model`). |
| `extra_files` | `()` | The model needs more files from the repo, such as `source.spm`. |

The engine expects a merged decoder (`decoder_model_merged*.onnx`) with `use_cache_branch` and `past_key_values.*` inputs. That's what `optimum-cli export onnx --task text2text-generation-with-past` produces. Exports with a separate `decoder_with_past_model.onnx` won't load.

Check that the output is right before trusting the speed: translate a few sentences and compare them with the reference `transformers` pipeline once.

## Contributing a model to noentenc

To ship a new model in the package rather than in your own code:

1. **Put it in its own file**, `models/<name>.py`, in `language_detection/` or `translation/`, and export it from the package `__init__.py`.
2. **Use `Language` for every language parameter** and declare a `schema` (translation). Unsupported languages must raise.
3. **Pin presets.** Downloaded weights use a fixed Hugging Face revision (`default_revision`, `OPUS_MT_REVISIONS`) and, for files we fetch ourselves, a sha256 (see `FastTextPreset` and `OnnxPreset`). Weights are never bundled in the wheel.
4. **Add optional dependencies as an extra** in `pyproject.toml`, and add the extra to `all`.
5. **Write tests:**
   - Unit tests in `tests/unit/` that run without network, using a tiny fixture or a fake session (see `tests/unit/translation/test_models.py`).
   - For a detection backend, drop its label list in `tests/unit/language_detection/fixtures/labels/<name>.txt`. `test_labels.py` checks that every label maps to a valid ISO 639-3 code.
   - An integration test in `tests/integrations/` with real weights (`make integration-tests`).
6. **Add it to the benchmark scripts** in `scripts/`, then add a row to the backend table in the README and to [benchmarks.md](benchmarks.md), including the licence of the weights.
