# Benchmarks

Speed numbers for every backend, measured on an Apple M3 (8 cores, macOS) with Python 3.14 and the weights noentenc downloads by default. These numbers measure speed, not accuracy.

To reproduce:

```bash
uv run python scripts/benchmark_lid.py --download
uv run python scripts/benchmark_translation.py --models opus-mt small100 m2m100 nllb
```

## Language detection

The detection benchmark repeats the 43 multilingual test sentences in `tests/unit/language_detection/fixtures/sentences.txt` up to 20,000 texts, in batches of 256. The two transformer models run on 2,000 texts. "Single text" is the median latency of one `predict` call over 200 runs.

| Backend | Load | Single text | Batched throughput | Size | Languages |
|---|---:|---:|---:|---:|---:|
| `heliport` | 0.15 s | 3.7 µs | 872k texts/s | ~130 MB wheel | 220 |
| `lid176` (default) | 0.01 s | 17 µs | 356k texts/s | 0.9 MB | 176 |
| `cld3` | <0.01 s | 16 µs | 66k texts/s | 1 MB | 107 |
| `langid` | 0.73 s | 38 µs | 55k texts/s | 1.9 MB | 97 |
| `lingua` (low accuracy) | <0.01 s | 0.36 ms | 18k texts/s | ~300 MB wheel | 75 |
| `lingua` | <0.01 s | 0.44 ms | 9.2k texts/s | ~300 MB wheel | 75 |
| `bert-openlid` | 0.14 s | 0.35 ms | 3.5k texts/s | 25 MB | 201 |
| `xlm-roberta-lid` | 0.45 s | 3.9 ms | 395 texts/s | 279 MB | 20 |

The other fastText presets (`lid176-bin`, `openlid-v2`, `openlid-v3`, `glotlid`, `nllb-lid218e`) run on the same numpy code as `lid176`. Their dense matrices are memory-mapped, so they load in milliseconds despite being 0.1 to 1.7 GB. They weren't cached on the benchmark machine. Run the script with `--download` to measure them.

Repeating sentences keeps fastText's per-word cache warm, which matches real corpora where words repeat. Pass `FastTextModel(cache_size=0)` to measure the uncached cost.

### Which one to use

- **Default to `lid176`.** It's 0.9 MB, has no dependencies beyond numpy, and handles about 350k texts per second.
- **For the long tail, use `glotlid`.** It has 2102 labels and runs on the same engine, but downloads 1.7 GB.
- **For short texts in a known set of languages, use `LinguaModel(languages=[...])`.** It's slower, but restricting the candidates helps where n-gram models struggle.
- **`heliport` is the fastest**, but it's GPL-3.0 and has no Windows wheel.

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
