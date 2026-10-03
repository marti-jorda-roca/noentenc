# Model candidates

## Goal and scope

Candidate models for lightweight text language detection and translation in
`noentenc`, without installing PyTorch or Python Transformers.

Use native fastText and CTranslate2 backends first, with an optional ONNX Runtime
backend for additional models. Prefer existing inference artifacts so users do
not need to convert models.

**ONNX: No** means the linked artifact uses another runtime, not that it has no
dependencies. ONNX is a property of the selected artifact, not an inherent
requirement of the model architecture.

Repository files and model cards were checked during candidate research.
These models have not yet been inference-tested in `noentenc`. A candidate is
not a guarantee of compatibility, performance, or translation quality.

## Candidate catalog

Sizes below are approximate weight-file sizes, not installed package sizes or
inference memory usage. ONNX translation models can require multiple graph
files; download only the files needed by the selected adapter and precision.

| Model | ONNX | Use case | Coverage | Proposed role |
| --- | :---: | --- | --- | --- |
| [fastText `lid.176.ftz`](https://huggingface.co/julien-c/fasttext-language-id/blob/main/lid.176.ftz) | No | Language detection | 176 languages | Default detector candidate; compressed weights under 1 MB. |
| [fastText `lid.176.bin`](https://huggingface.co/julien-c/fasttext-language-id/blob/main/lid.176.bin) | No | Language detection | 176 languages | Optional faster, slightly more accurate alternative; approximately 131 MB of weights. |
| [OPUS-MT English → Spanish — CTranslate2](https://huggingface.co/michaelfeil/ct2fast-opus-mt-en-es) | No | Language translation | English → Spanish | Starter translation candidate; approximately 156 MB of weights. |
| [OPUS-MT Spanish → English — CTranslate2](https://huggingface.co/michaelfeil/ct2fast-opus-mt-es-en) | No | Language translation | Spanish → English | Reverse-direction starter candidate; approximately 156 MB of weights. |
| [OPUS-MT English → French — CTranslate2](https://huggingface.co/michaelfeil/ct2fast-opus-mt-en-fr) | No | Language translation | English → French | Additional pair-specific candidate; approximately 150 MB of weights. |
| [XLM-RoBERTa language detection — ONNX](https://huggingface.co/onnx-community/xlm-roberta-base-language-detection-ONNX) | Yes | Language detection | 20 languages | Optional neural detector; larger and narrower language coverage than the default fastText candidate. |
| [OPUS-MT English → Spanish — ONNX](https://huggingface.co/Xenova/opus-mt-en-es) | Yes | Language translation | English → Spanish | First ONNX translation candidate; quantized artifacts available. |
| [OPUS-MT Spanish → English — ONNX](https://huggingface.co/Xenova/opus-mt-es-en) | Yes | Language translation | Spanish → English | Reverse-direction ONNX candidate using the same adapter family. |
| [M2M100 418M — ONNX](https://huggingface.co/Xenova/m2m100_418M) | Yes | Language translation | 100 languages, many-to-many | Opt-in multilingual candidate; substantially heavier than pair-specific OPUS-MT. |
| [NLLB-200 distilled 600M INT8 — CTranslate2](https://huggingface.co/mijuanlo/nllb-200-distilled-600M-ct2-int8) | No | Language translation | 200 languages | Opt-in noncommercial candidate; approximately 623 MB of weights. |
| [NLLB-200 distilled 600M — ONNX](https://huggingface.co/Xenova/nllb-200-distilled-600M) | Yes | Language translation | 200 languages | Opt-in noncommercial candidate; quantized artifacts available. |
| [Multilingual BERT language detection — ONNX](https://huggingface.co/Oblix/bert-base-multilingual-cased-language-detection_ONNX) | Yes | Language detection | 45 language/variant labels | Smaller neural candidate with approximately 179 MB of quantized weights; blocked on license and label-map verification. |
| [ProtectAI XLM-RoBERTa language detection — ONNX](https://huggingface.co/protectai/xlm-roberta-base-language-detection-onnx) | Yes | Language detection | 20 languages | Alternative export with approximately 279 MB of quantized weights and explicit language labels; repository is archived. |
| [Facebook fastText language identification (`lid218e`)](https://huggingface.co/facebook/fasttext-language-identification) | No | Language detection | 217 languages, per the model card | Not a lightweight default: approximately 1.18 GB of weights and a noncommercial license. |

## Recommended first-release selection

### Native backends

- **Detection:** fastText with `lid.176.ftz` as the default candidate.
- **Translation:** CTranslate2 with SentencePiece and model-specific processing,
  initially supporting pair-specific OPUS-MT artifacts.
- Keep translation loading separate from detection. Detection-only users should
  not need to install or import the translation runtime.
- Download models on demand and cache them; do not bundle every candidate into
  the package or load every model at startup.

The English, Spanish, and French translation models are concrete starter
candidates, not a restriction on the language pairs that can be supported.
Translation should select a model for the requested language pair rather than
having a universal default model.

### Optional ONNX backend

- Use raw `onnxruntime`, plus standalone tokenizer support where needed.
- Start with ONNX OPUS-MT so both translation backends can be tested against the
  same underlying model families.
- Add M2M100 as an opt-in multilingual candidate.
- Keep neural detectors explicitly selectable. Do not automatically replace the
  lightweight fastText detector with a much larger neural model.

This is a proposed dependency structure, not an already implemented API.

## Runtime and tokenizer requirements

| Artifact family | Intended inference dependencies | Adapter responsibilities |
| --- | --- | --- |
| fastText | A compatible fastText runtime | Input normalization, language-label mapping, candidate scores, and uncertain-input handling. |
| CTranslate2 OPUS-MT | `ctranslate2`, `sentencepiece` | Model-specific normalization, source/target tokenization, required special tokens, and decoding. |
| ONNX classifiers | `onnxruntime`, usually `tokenizers`, and array handling | Input tensors, padding, special tokens, output scores, and label mapping. |
| ONNX translation | `onnxruntime`, standalone tokenizer support, and array handling | Encoder/decoder execution, generation loop, cache handling, language tokens, stopping rules, and decoding. |
| CTranslate2 NLLB | `ctranslate2`, standalone tokenizer support | Language/script tags, source special tokens, target prefixes, and decoding. |

Some linked model cards demonstrate `AutoTokenizer`, Hugging Face wrapper
packages, or Transformers.js. Those examples do not define the dependencies
`noentenc` must use. Implement and validate lightweight adapters instead of
copying examples that introduce Python Transformers.

CTranslate2 already implements translation decoding. Raw ONNX Runtime does
not supply the model-specific translation generation loop, which makes
CTranslate2 the simpler initial translation backend.

Download support can use `huggingface_hub` independently of Transformers.
Never automatically execute custom Python code from a model repository.

## Licensing and provenance

Check both the original model and the converted artifact before redistribution.
A permissively licensed runtime does not imply that its model weights have the
same license.

- **fastText `lid.176`:** the
  [original fastText documentation](https://fasttext.cc/docs/en/language-identification.html)
  specifies CC-BY-SA 3.0, while the linked Hugging Face mirror labels the model
  CC-BY-SA 4.0. Resolve the discrepancy and preserve upstream attribution before
  redistributing the weights.
- **CTranslate2 OPUS-MT candidates:** their cards declare Apache-2.0. Verify
  corresponding upstream model terms and retain provenance.
- **Xenova OPUS-MT candidates:** the original
  [English → Spanish](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) and
  [Spanish → English](https://huggingface.co/Helsinki-NLP/opus-mt-es-en) models
  declare Apache-2.0. Confirm converted-artifact terms rather than relying only
  on the original model's metadata.
- **M2M100:** the
  [original model](https://huggingface.co/facebook/m2m100_418M) declares MIT.
  Confirm the ONNX conversion's provenance and distribution terms.
- **NLLB:** both listed conversions inherit the original model's
  CC-BY-NC-4.0 restrictions. Do not use them as an unrestricted commercial
  default.
- **Facebook fastText `lid218e`:** the repository declares CC-BY-NC-4.0.
- **ProtectAI XLM-R:** the model card declares MIT, but the repository explicitly
  states that it is archived and no longer actively maintained.
- **Oblix multilingual BERT:** the fine-tuned and ONNX repositories do not
  clearly declare licensing terms. The multilingual BERT base model's
  Apache-2.0 license alone does not resolve the fine-tuned artifact's terms.
- **Other community conversions:** review their own model cards and the
  corresponding upstream licenses before approving them for distribution.

## Detector-specific caveats

- Short, ambiguous, numeric-only, and mixed-language inputs need explicit
  handling. Allow an unknown result rather than always forcing a language.
- Raw classifier scores should not be described as calibrated probabilities
  without validation.
- The Oblix ONNX classifier uses generic labels `LABEL_0` through `LABEL_44`.
  Verify the exact language ordering from upstream training metadata before
  integration. Its upstream model is
  [jb2k/bert-base-multilingual-cased-language-detection](https://huggingface.co/jb2k/bert-base-multilingual-cased-language-detection).
- Standardize public language codes, but preserve distinctions needed by
  translation models, such as NLLB language/script tags and regional labels.
- Language coverage does not establish accuracy. Compare candidates on the
  same representative inputs rather than comparing unrelated model-card
  benchmark scores.

## Acceptance checklist

Before marking a candidate as supported:

1. Resolve the license and record the original model and conversion provenance.
2. Pin the repository revision and record artifact checksums.
3. Verify that installation and inference work without PyTorch or Transformers.
4. Validate tokenizer normalization, token mappings, padding, and special tokens.
5. Check detector label ordering or translation language-tag handling.
6. Run representative quality tests, including short and ambiguous detection
   inputs and all supported translation directions.
7. Measure cold-start time, warm latency, batch throughput, download size, and
   peak memory on target hardware.
8. Verify native-wheel availability across supported Python versions and
   operating systems.
9. Test offline loading from the cache and clear errors for unsupported
   artifacts, missing dependencies, and unsupported language pairs.
