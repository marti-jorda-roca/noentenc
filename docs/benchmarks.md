# Benchmarks

Speed numbers for every backend, measured on an Apple M3 (8 cores, macOS) with Python 3.14 and the weights noentenc downloads by default. Most sections measure speed only; [Profiles](#profiles) also measures accuracy, which is how the `speed`, `balance` and `quality` profiles were chosen.

To reproduce:

```bash
uv run python scripts/benchmark_lid.py --download
uv run python scripts/benchmark_translation.py --models opus-mt small100 m2m100 nllb
```

## Language detection

Two workloads, both in batches of 256:

- **Short sentences.** The 43 multilingual test sentences in `tests/unit/language_detection/fixtures/sentences.txt`, repeated up to 20,000 texts. The two transformer models run on 2,000 texts. "Single text" is the median latency of one `predict` call over 200 runs. Words repeat, so fastText's per-word cache stays warm.
- **Unique paragraphs.** The first 20,000 paragraphs of the [WiLI-2018](https://huggingface.co/datasets/MartinThoma/wili_2018) test set: 235 languages, a median of 272 characters, no text seen twice. "Single text" is the median over 200 further paragraphs the model hasn't seen. This is closer to labelling a real dataset, and the cost of words the cache hasn't seen yet shows.

| Backend | Load | Single sentence | Sentences/s | Single paragraph | Paragraphs/s | Size | Languages |
|---|---:|---:|---:|---:|---:|---:|---:|
| `heliport` | 0.10 s | 2.5 µs | 1.2M | 28 µs | 135k | ~130 MB wheel | 220 |
| `lid176` (default) | 0.02 s | 15 µs | 419k | 0.15 ms | 22k | 0.9 MB | 176 |
| `lid176-bin` | 0.02 s | 15 µs | 436k | 0.16 ms | 12k | 126 MB | 176 |
| `openlid-v3` | 0.06 s | 47 µs | 210k | 0.31 ms | 2.7k | 1.2 GB | 195 |
| `langid` | 0.80 s | 19 µs | 730k | 36 µs | 65k | 1.9 MB | 97 |
| `cld3` | <0.01 s | 16 µs | 66k | 65 µs | 12k | 1 MB | 107 |
| `lingua` (low accuracy) | <0.01 s | 0.36 ms | 18k | 2.2 ms | 1.8k | ~300 MB wheel | 75 |
| `lingua` | <0.01 s | 0.44 ms | 9.2k | 1.2 ms | 3.6k | ~300 MB wheel | 75 |
| `bert-openlid` | 0.14 s | 0.35 ms | 3.5k | 2.2 ms | 570 | 25 MB | 201 |
| `xlm-roberta-lid` | 0.45 s | 3.9 ms | 395 | 28 ms | 32 | 279 MB | 20 |

To measure on your own texts, pass a file with one text per line: `scripts/benchmark_lid.py --corpus texts.txt`. The multi-threaded backends (onnxruntime, lingua, heliport) vary by about 15% between runs on the benchmark laptop, which throttles under sustained load.

The other fastText presets (`openlid-v2`, `glotlid`, `nllb-lid218e`) share `openlid-v3`'s 256-dimension architecture and run on the same numpy code. Their dense matrices are memory-mapped, so they load in milliseconds despite being 1.2 to 1.7 GB.

What keeps the numpy backends fast:

- **fastText** hashes the character n-grams of every word a batch hasn't seen in one pass of numpy operations, then caches each word's summed embedding (`cache_size` words, 1 << 17 by default). A text costs one cache lookup per word. `FastTextModel(cache_size=0)` measures the uncached cost.
- **langid** computes the state of its byte automaton for every byte of a batch at once, because the state only depends on the last four bytes. It then scores only the states the batch visits. Its throughput peaks around the default `batch_size=32` (89k paragraphs/s).

Both return the same labels and scores as the reference `fasttext` and `langid` packages; the parity tests check that.

`bert-openlid` reads at most 96 tokens per text. Its accuracy on the 117,500 WiLI-2018 test paragraphs is the same at 96 and 128 tokens (59.58%), and 96 tokens label paragraphs about 35% faster in batches and 20% faster one at a time. Cutting to 64 tokens is another 1.6× faster but costs 0.26 points (59.32%). Pass `max_length=` to choose. `xlm-roberta-lid` keeps 128 tokens.

### Which one to use

- **Default to `lid176`.** It's 0.9 MB, has no dependencies beyond numpy, and labels about 20k paragraphs or 400k short sentences per second.
- **For the long tail, use `glotlid`.** It has 2102 labels and runs on the same engine, but downloads 1.7 GB.
- **For short texts in a known set of languages, use `LinguaModel(languages=[...])`.** It's slower, but restricting the candidates helps where n-gram models struggle.
- **`heliport` is the fastest**, but it's GPL-3.0 and has no Windows wheel.

## Conservative detection

`LanguageDetector(min_letters=..., min_score=..., min_margin=...)` returns `und` instead of a guess when a text misses a threshold. These rates come from `scripts/evaluate_detection.py`, run on the 126 hand-labelled cases in `scripts/data/detection_cases.tsv`:

- 92 texts with a language: 40 short support messages in 15 languages, 17 messages with typos, 19 sentences in closely related languages (Catalan/Spanish, Galician/Portuguese, Danish/Norwegian/Swedish, Czech/Slovak, Croatian/Serbian, Indonesian/Malay, Ukrainian/Russian, Afrikaans/Dutch), 8 mixed-language messages where either language counts, and 8 paragraphs.
- 20 texts a conservative router should skip: 10 people's names and 10 chat tokens such as `lol`, `ok` and `xD`.
- 14 nonlinguistic texts: numbers, emoji, punctuation, URLs and email addresses.

**Right**, **Wrong** and **Abstained** are shares of the 92 texts with a language. **Skipped** is the share of the 20 names and chat tokens that got `und` or `zxx`. Labels are compared with macrolanguages collapsed. All 14 nonlinguistic texts get `zxx` under every setting, without running the model; before this rule they all got a language label.

| Backend | Setting | Right | Wrong | Abstained | Skipped |
|---|---|---:|---:|---:|---:|
| `lid176` | default | 97% | 3% | 0% | 0% |
| | `min_letters=4, min_score=0.5` (tested) | 92% | 2% | 5% | 90% |
| | `min_score=0.7` | 84% | 1% | 15% | 100% |
| `langid` | default | 92% | 8% | 0% | 0% |
| | `min_letters=4, min_score=0.7` (tested) | 89% | 4% | 7% | 85% |
| `bert-openlid` | default | 91% | 9% | 0% | 0% |
| | `min_score=0.7` (tested) | 82% | 2% | 16% | 95% |
| `lingua` | default | 92% | 8% | 0% | 0% |
| | `min_margin=0.1` (tested) | 85% | 1% | 14% | 70% |
| | `min_score=0.5` | 66% | 1% | 33% | 90% |
| `cld3` | default | 87% | 13% | 0% | 0% |
| | `min_letters=4, min_score=0.7` (tested) | 80% | 4% | 15% | 65% |
| `heliport` | default | 98% | 2% | 0% | 35% |
| | `min_letters=4, min_score=0.3` (tested) | 93% | 0% | 7% | 85% |

- **`min_letters=4` alone skips most chat tokens** (`lol`, `ok`, `xD`) at no cost to real messages, and it saves the model call. It doesn't catch names.
- **A score threshold catches names**, which score low on most backends. The tested setting cuts wrong labels by a third on `lid176` and by half or more on the others, and skips 85 to 95% of names and chat tokens. CLD3 is the exception: it's confident about too many names.
- **lingua's scores are relative**, so it abstains much more at the same `min_score` than the other backends. A small `min_margin` suits it better.
- **heliport's confidences run from 0 to about 2.5**, not 0 to 1, so its thresholds are lower.
- `openlid-v3` and `glotlid` weren't evaluated here. Their softmax scores behave like `bert-openlid`'s, but evaluate them with `--backends openlid-v3 glotlid` before you pick a threshold.

The cases are few and hand-picked, so treat the rates as a guide to the trade-off, not as accuracy figures. To evaluate on your own messages, add rows to the TSV and rerun the script; `--by-category` breaks the rates down.

## Against the original implementations

How many times faster noentenc is than each model's usual package, on the same texts and laptop (below 1× means the original is faster):

| Model | Compared with | Single text | Batches of sentences | Batches of paragraphs | Start-up | Same output |
|---|---|---:|---:|---:|---:|---|
| `langid` | `langid.py` | **20×** | **470×** | **47×** | 1× | identical labels |
| `lid176` (default) | `fasttext` (C++) | 0.27× | **2×** | 0.86× | 0.8× | identical labels |
| `bert-openlid` | transformers + PyTorch | **3.8×** | **2×** | 0.8× | **8.5×** | 99% of labels, same accuracy |
| `xlm-roberta-lid` | transformers + PyTorch | **5.8×** | **1.7×** | 0.9× | **4.8×** | identical on its 20 languages |
| Opus-MT en→es | transformers + PyTorch | **1.2×** | **1.9×** | | **4×** | BLEU 42.0 vs 42.3 |

noentenc also runs fastText models on Python 3.14, where the `fasttext` package has no wheels.

The same models run with the packages they usually ship with: `fasttext` 0.9.2 (C++, from `fasttext-wheel`), `langid` 1.1.6, and transformers 5.18 on PyTorch 2.14 with the original fp32 weights. Both sides get the same texts, the same batch sizes (256 texts for detection, 32 sentences for translation), the same token limit (96 for bert-openlid, 128 for xlm-roberta-lid) and greedy decoding. Sentences and paragraphs are the two detection workloads above. Start-up is a fresh process that imports the library, loads the cached model offline and returns one prediction. Runs alternated between the two sides, and each side keeps its best of two.

| Model | Implementation | Single sentence | Sentences/s | Single paragraph | Paragraphs/s | Start-up |
|---|---|---:|---:|---:|---:|---:|
| `lid176` | noentenc | 15 µs | 475k | 154 µs | 22k | 0.20 s |
| | `fasttext` | 4.0 µs | 241k | 42 µs | 26k | 0.16 s |
| `langid` | noentenc | 19 µs | 745k | 36 µs | 62k | 0.95 s |
| | `langid.py` | 378 µs | 1.6k | 610 µs | 1.3k | 0.98 s |
| `bert-openlid` | noentenc (int8) | 0.35 ms | 3.6k | 1.9 ms | 607 | 0.25 s |
| | transformers (fp32) | 1.34 ms | 1.8k | 2.6 ms | 746 | 2.1 s |
| `xlm-roberta-lid` | noentenc (int8) | 2.9 ms | 310 | 21 ms | 40 | 0.62 s |
| | transformers (fp32) | 16.7 ms | 185 | 30 ms | 44 | 3.0 s |

| Opus-MT en→es | Single sentence | Sentences/s | Start-up | BLEU | chrF |
|---|---:|---:|---:|---:|---:|
| noentenc (q4, the default) | 53 ms | 122 | 0.69 s | 41.97 | 63.68 |
| transformers (fp32) | 62 ms | 66 | 2.8 s | 42.26 | 63.98 |

BLEU and chrF are sacrebleu scores on the 2,000 sentences of the OPUS-100 en→es test set. noentenc's fp32 export scores 41.99 and 63.76.

The outputs match:

- **`lid176` and `langid`** return the same label as `fasttext` and `langid.py` on all 245,000 texts tried: the WiLI-2018 test paragraphs, the same paragraphs cut to 40 characters, and the papluca test set.
- **`bert-openlid`**, int8 against fp32, agrees on 98.9 to 99.8% of those texts, and its accuracy is within 0.06 points on each set.
- **`xlm-roberta-lid`** returns the same labels on papluca, which covers its 20 languages. On WiLI, mostly languages it doesn't know, the int8 and fp32 models agree on 87 to 93% of texts and are equally accurate.

The original is faster in two places. The C++ `fasttext` answers a single text in 4 µs, where noentenc spends 15 µs in about twenty numpy calls. And on the M3, PyTorch's fp32 matrix products run on Apple's AMX units through Accelerate, which onnxruntime's int8 kernels don't use, so PyTorch handles big batches of long texts 10 to 25% faster. Other CPUs may differ.

noentenc's dependencies take about 140 MB installed on macOS (onnxruntime 77 MB, numpy 25 MB); torch and transformers with theirs take about 770 MB.

To reproduce, run each side with `scripts/benchmark_vs_reference.py` (its docstring has the commands) and a corpus file such as the WiLI-2018 test paragraphs, one per line.

## Translation

The translation benchmark translates 8 English sentences of mixed length into Spanish. "Single" is the median latency over 10 sentences, after a warm-up call. "Batched" translates 128 sentences with `batch_size=32`. Decoding is greedy, with onnxruntime using every core.

| Model | Precision | Load | Single sentence | Batched | Download | Licence |
|---|---|---:|---:|---:|---:|---|
| Opus-MT en→es | fp32 | 0.6 s | 56 ms | 121 sent/s | 425 MB | CC-BY-4.0 or Apache-2.0, per pair |
| | int8 | 0.3 s | 78 ms | 118 sent/s | 107 MB | |
| | q4 (default) | 0.4 s | 63 ms | 100 sent/s | 287 MB | |
| SMaLL-100 | int8 (only) | 0.5 s | 52 ms | 75 sent/s | 595 MB | MIT |
| M2M100 418M | int8 (default) | 1.2 s | 445 ms | 18 sent/s | 603 MB | MIT |
| | q4 | 2.7 s | 231 ms | 15 sent/s | 1.2 GB | |
| NLLB-200 600M | int8 (default) | 1.3 s | 770 ms | 10 sent/s | 860 MB | CC-BY-NC-4.0 |
| | fp32 | 5.5 s | 450 ms | 12 sent/s | 3.5 GB | |

`Translator()` picks the Opus-MT model for the pair when one exists (66 directions, see `OPUS_MT_PAIRS`), and SMaLL-100 for any other pair. Both are in the top rows of the table.

Download sizes are the ONNX encoder and decoder that each precision loads.

The Xenova q4 exports keep some weights in fp32, so for Opus-MT and M2M100 q4 is larger than int8. On batches it's also slower. For M2M100 and NLLB-200, q4 also translates much worse (see [Profiles](#translation-1)), so int8 is their default. Benchmark your own workload before you pick a precision.

### Memory

A loaded model takes much more RAM than its download: onnxruntime keeps the parsed graph, optimised copies of the weights and its working buffers. Resident memory added by loading each model and translating one sentence, in a fresh process with onnxruntime and tokenizers already imported (median of 3 runs):

| Model | Precision | RAM | Download |
|---|---|---:|---:|
| Opus-MT en→es | int8 | 630 MB | 107 MB |
| | q4 (default) | 1.08 GB | 287 MB |
| | fp32 | 1.31 GB | 425 MB |
| SMaLL-100 | int8 | 1.15 GB | 595 MB |
| M2M100 418M | int8 | 2.38 GB | 603 MB |
| NLLB-200 600M | int8 | 4.06 GB | 860 MB |
| | fp32 | 3.81 GB (5.8 GB peak while loading) | 3.5 GB |

`Translator` keeps at most `max_loaded_models` models loaded, two by default, and drops the least recently used one before it loads another. Peak resident memory of a `Translator()` that translates one sentence for each of 9 pairs, twice round (8 Opus-MT pairs at q4, plus Catalan→English on SMaLL-100):

| `max_loaded_models` | Peak RAM | Models loaded at the end | Time |
|---:|---:|---:|---:|
| 1 | 1.6 GB | 1 | 8.2 s |
| 2 (default) | 2.4 GB | 2 | 8.3 s |
| 4 | 3.9 GB | 4 | 9.1 s |
| `None` (no limit) | 5.0 GB | 9 | 6.2 s |

With one sentence per call, the time is almost all loading: a limit reloads the 9 models on the second round, about 0.2 s each. Keep the limit low when memory is tight, and raise it when calls keep alternating between more pairs than it allows. `translator.unload()` drops every loaded model.

These are macOS figures. macOS compresses memory that isn't being used, which lowers resident memory for idle models, so expect higher numbers without a limit on Linux. To reproduce, run `uv run --with psutil python scripts/measure_translation_memory.py models` and `... workload`.

## Profiles

`LanguageDetector(profile)` and `Translator(profile)` pick models from the measurements below. Accuracy comes from the [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200) devtest set, which has the same 1,012 sentences in 204 language variants.

To reproduce:

```bash
curl -O https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
tar -xzf flores200_dataset.tar.gz
uv run python scripts/benchmark_accuracy.py lid --flores flores200_dataset
uv run --with sacrebleu python scripts/benchmark_accuracy.py translation --flores flores200_dataset
```

### Language detection

The first 300 sentences of each FLORES-200 language: 61,200 texts. Labels are compared with macrolanguages collapsed on both sides, so Moroccan Arabic (`ary`) counts as Arabic (`ara`) and the 204 variants become 176 languages. Accuracy is averaged per language, so every language weighs the same.

- **All**: every language. A model scores 0 on a language it doesn't know.
- **Known**: only the languages the model has a label for.
- **Common**: the 79 languages every model below knows.
- **40 chars**: the same texts cut to their first 40 characters, closer to chat messages and titles.

| Model | All | Known | Common | All, 40 chars | Common, 40 chars | Single sentence | Sentences/s | Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lid176` (`speed`) | 50.4% | 83.0% | 92.9% | 44.4% | 83.9% | 0.11 ms | 55k | 0.9 MB |
| `lid176-bin` | 52.4% | 86.3% | 94.9% | 47.3% | 87.8% | | | 126 MB |
| `langid` | 44.9% | 92.9% | 94.8% | 40.8% | 86.4% | | | 1.9 MB |
| `openlid-v3` (`balance`) | 95.8% | 98.1% | 99.6% | 88.7% | 96.0% | 0.20 ms | 11k | 1.2 GB |
| `bert-openlid` | 96.2% | 96.2% | 98.7% | 85.9% | 92.0% | | | 25 MB |
| `glotlid` (`quality`) | **97.7%** | **98.3%** | **99.9%** | **91.6%** | **97.2%** | 0.55 ms | 9.6k | 1.7 GB |

Speed is measured on the 206,448 FLORES-200 devtest sentences, shuffled, so no text repeats. Throughput is the best of two runs: the first run of a large model also reads its memory-mapped weights from disk and was about half as fast.

- **`lid176` misses most of the long tail.** It has no label for 69 of the 176 languages. On the ones it knows it's accurate, and nothing else here comes close to its speed or size.
- **`openlid-v3` is the big step up.** It knows 172 of the 176 languages and is right 96% of the time. It's about 5× slower than `lid176` in batches, but still labels about 11k sentences per second.
- **`glotlid` is the most accurate on every measure**, especially on short texts. It's about 2.7× slower than `openlid-v3` for one text at a time.
- **`bert-openlid` is dominated.** It's less accurate than `openlid-v3` and about 5× slower on paragraphs (see the first table), so no profile uses it.

`openlid-v3` and `glotlid` tell apart varieties that `lid176` lumps together: they return `swh` (Swahili) or `swc` (Congo Swahili) where `lid176` returns `swa`. Pass `collapse_macrolanguages=True` to a model to fold them back.

### Translation

chrF++ ([sacrebleu](https://github.com/mjpost/sacrebleu), `word_order=2`) on the first 200 FLORES-200 devtest sentences of each pair, with greedy decoding. Opus-MT runs at its default q4. Speed is from the [translation benchmark](#translation) above: one sentence at a time, then batches of 32.

| Pair | Opus-MT | SMaLL-100 (`speed`) | NLLB-200 int8 (`balance`) | NLLB-200 fp32 (`quality`) |
|---|---:|---:|---:|---:|
| en→es | 52.3 | 49.1 | 52.6 | **52.9** |
| es→en | 54.4 | 51.6 | 56.7 | **56.8** |
| en→de | **59.6** | 53.1 | 58.2 | 58.2 |
| de→en | 63.2 | 58.5 | **64.7** | **64.7** |
| en→zh | **22.2** | 18.5 | 19.1 | 18.4 |
| zh→en | 49.6 | 46.3 | 52.0 | **52.3** |
| en→fi | **51.9** | 44.9 | 47.5 | 47.8 |
| fr→de | **51.1** | 48.0 | 50.5 | 50.3 |
| en→ja | | 21.7 | **22.6** | **22.6** |
| en→sw | | 52.9 | 58.7 | **59.0** |
| sw→en | | 55.1 | 62.4 | **62.9** |
| en→ta | | 28.3 | 47.2 | **47.4** |
| de→it | | 48.2 | 50.4 | **50.5** |
| Single sentence | 63 ms | 53 ms | 790 ms | 450 ms |
| Sentences/s | 100 | 70 | 11 | 12 |
| Download | 287 MB per pair | 595 MB | 860 MB | 3.5 GB |
| Load | 0.4 s | 0.7 s | 1.4 s | 5.5 s |

Chinese and Japanese scores are low for every model because chrF++ counts word n-grams, and those languages aren't written with spaces. Compare them across models, not with the other rows.

- **Every profile uses Opus-MT where it has a model.** On those 8 pairs it averages 50.5, against 50.2 for NLLB-200 fp32, and it's about 8× faster in batches. NLLB-200 is better into English and Opus-MT is better out of it.
- **On the other pairs, NLLB-200 beats SMaLL-100 on every pair.** The gap is small on high-resource pairs (1 to 2 points on en→ja and de→it) and large on low-resource ones (7 points on sw→en, 19 on en→ta). It's also about 6× slower in batches, which is why `speed` keeps SMaLL-100.
- **fp32 adds a little over int8**: 0.25 points on average over the five pairs without Opus-MT. int8 downloads 4× less, loads 4× faster and needs less memory. On the M3, fp32 is nonetheless faster for single sentences and as fast in batches; other CPUs may differ.
- **q4 hurts NLLB-200 and M2M100, so both default to int8.** At q4, NLLB-200 scored up to 7 points lower (en→ta 40.3 against 47.2, en→fi 42.3 against 47.5) and M2M100 up to 26 points lower (en→sw 18.3 against 44.8, en→de 36.9 against 52.8). Their q4 files are also 2 to 2.6× larger.
- **M2M100 418M isn't in any profile.** At int8 it's about as accurate as SMaLL-100 (43.8 against 44.3 averaged over the 13 pairs): better into Japanese (25.4 against 21.7) and Chinese (20.6 against 18.5), worse on Swahili (44.8 against 52.9 into it) and Tamil (24.9 against 28.3). It's about 4× slower and needs the source language. NLLB-200 beats it on every pair except into Chinese and Japanese.
