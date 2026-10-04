# Benchmarks

Speed numbers for every backend, measured on an Apple M3 (8 cores, macOS) with Python 3.14 and the weights noentenc downloads by default. These numbers measure speed, not accuracy.

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
| M2M100 418M | int8 | 1.2 s | 445 ms | 18 sent/s | 603 MB | MIT |
| | q4 (default) | 2.7 s | 231 ms | 15 sent/s | 1.2 GB | |
| NLLB-200 600M | int8 | 1.3 s | 770 ms | 10 sent/s | 860 MB | CC-BY-NC-4.0 |

`Translator()` picks the Opus-MT model for the pair when one exists (66 directions, see `OPUS_MT_PAIRS`), and SMaLL-100 for any other pair. Both are in the top rows of the table.

Download sizes are the ONNX encoder and decoder that each precision loads.

The Xenova q4 exports keep some weights in fp32, so for Opus-MT and M2M100 q4 is larger than int8. On batches it's also slower. Benchmark your own workload before you pick a precision.
