"""Time noentenc against each model's standard implementation, on the same texts.

The standard implementations don't install on Python 3.14, so the script runs once per side and
prints one JSON line per model. Run the sides one after another on an idle machine:

    uv run python scripts/benchmark_vs_reference.py ours --corpus wili.txt
    uv run --no-project --python 3.12 --with fasttext-wheel --with "numpy<2" --with langid \\
        python scripts/benchmark_vs_reference.py reference --models lid176 langid --corpus wili.txt
    uv run --no-project --python 3.12 --with torch --with transformers --with sentencepiece \\
        --with sacremoses python scripts/benchmark_vs_reference.py reference \\
        --models bert-openlid xlm-roberta-lid opus-mt --corpus wili.txt

The references are the ``fasttext`` package (C++), ``langid.py``, and Hugging Face
``transformers`` on PyTorch with the original fp32 weights. Both sides get the same batch sizes,
the same token limit (96 for bert-openlid, 128 for xlm-roberta-lid) and greedy decoding.

Detection is timed on the test sentences in ``tests/unit/language_detection/fixtures`` (short
texts, repeated to fill the batches) and on ``--corpus``, one text per line and none repeated:
docs/benchmarks.md uses the WiLI-2018 test paragraphs. lid176 is read from noentenc's cache, so
run the ``ours`` side first.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SENTENCES = ROOT / "tests/unit/language_detection/fixtures/sentences.txt"
LID176 = Path.home() / ".cache/noentenc/fasttext/lid.176.ftz"
TRANSLATE = [
    "Hello world, how are you?",
    "The weather is nice today.",
    "I would like to order a coffee and two croissants, please.",
    "The European Central Bank kept interest rates unchanged on Thursday, citing persistent "
    "uncertainty about inflation and slowing growth across the euro area.",
    "Please restart the server before running the migration.",
    "She has been learning to play the piano for three years.",
    "Can you tell me where the nearest train station is?",
    "Our quarterly revenue grew by twelve percent compared to last year.",
]
TOKEN_LIMITS = {"bert-openlid": 96, "xlm-roberta-lid": 128}
HF_REPOS = {
    "bert-openlid": "alexneakameni/language_detection",
    "xlm-roberta-lid": "papluca/xlm-roberta-base-language-detection",
    "opus-mt": "Helsinki-NLP/opus-mt-en-es",
}
DETECTION_BATCH = 256
TRANSLATION_BATCH = 32
SINGLE_RUNS = 200
# (short texts, corpus texts, single corpus texts) per model: fewer for the transformers.
WORKLOADS = {
    "bert-openlid": (2_000, 1_000, 200),
    "xlm-roberta-lid": (400, 200, 30),
}
DEFAULT_WORKLOAD = (20_000, 20_000, 200)

One = Callable[[str], Any]
Many = Callable[[list[str]], Any]


def _median_latency(one: One, texts: list[str], runs: int) -> float:
    one(texts[0])
    timings = []
    for i in range(runs):
        text = texts[i % len(texts)]
        start = time.perf_counter()
        one(text)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def _throughput(many: Many, texts: list[str]) -> float:
    many(texts[:64])
    start = time.perf_counter()
    many(texts)
    return len(texts) / (time.perf_counter() - start)


def _detection(one: One, many: Many, corpus: list[str], model: str) -> dict[str, float]:
    sentences = SENTENCES.read_text(encoding="utf-8").splitlines()
    short_n, corpus_n, corpus_runs = WORKLOADS.get(model, DEFAULT_WORKLOAD)
    repeated = (sentences * (short_n // len(sentences) + 1))[:short_n]
    paragraphs = corpus[:corpus_n]
    unseen = corpus[corpus_n : corpus_n + corpus_runs] or paragraphs
    return {
        "short_single_us": _median_latency(one, sentences, SINGLE_RUNS) * 1e6,
        "short_texts_s": _throughput(many, repeated),
        "corpus_single_us": _median_latency(one, unseen, corpus_runs) * 1e6,
        "corpus_texts_s": _throughput(many, paragraphs),
    }


def _translation(one: One, many: Many) -> dict[str, float]:
    single = _median_latency(one, TRANSLATE, 10)
    texts = (TRANSLATE * 16)[:128]
    return {"single_ms": single * 1e3, "sentences_s": _throughput(many, texts)}


def _ours(model: str) -> tuple[One, Many]:
    if model == "opus-mt":
        from noentenc.languages import Language
        from noentenc.translation import OpusMTModel

        mt = OpusMTModel.from_pair(Language.ENGLISH, Language.SPANISH)
        en, es = Language.ENGLISH, Language.SPANISH
        return (
            lambda text: mt.predict(text, es, en),
            lambda texts: mt.predict_batch(texts, es, en, TRANSLATION_BATCH),
        )
    from noentenc.language_detection.models import (
        FastTextModel,
        LangidModel,
        OnnxClassifierModel,
    )

    detector = (
        FastTextModel("lid176")
        if model == "lid176"
        else LangidModel()
        if model == "langid"
        else OnnxClassifierModel(model)
    )
    return detector.predict, lambda texts: detector.predict_batch(
        texts, batch_size=DETECTION_BATCH
    )


def _reference(model: str) -> tuple[One, Many]:
    if model == "lid176":
        import fasttext  # ty: ignore[unresolved-import] - only in the reference environment

        ft = fasttext.load_model(str(LID176))
        return ft.predict, ft.predict
    if model == "langid":
        import langid  # ty: ignore[unresolved-import] - only in the reference environment

        return langid.classify, lambda texts: [langid.classify(t) for t in texts]
    return _transformers(model)


def _transformers(model: str) -> tuple[One, Many]:
    import torch  # ty: ignore[unresolved-import] - only in the reference environment
    import transformers  # ty: ignore[unresolved-import]

    repo = HF_REPOS[model]
    if model == "opus-mt":
        tok = transformers.MarianTokenizer.from_pretrained(repo)
        net = transformers.MarianMTModel.from_pretrained(repo).eval()

        @torch.inference_mode()
        def translate(texts: list[str]) -> list[str]:
            out: list[str] = []
            for i in range(0, len(texts), TRANSLATION_BATCH):
                enc = tok(
                    texts[i : i + TRANSLATION_BATCH], padding=True, return_tensors="pt"
                )
                limit = 2 * enc["input_ids"].shape[1] + 10  # noentenc's limit
                ids = net.generate(
                    **enc, num_beams=1, do_sample=False, max_new_tokens=limit
                )
                out += tok.batch_decode(ids, skip_special_tokens=True)
            return out

        return lambda text: translate([text]), translate

    tok = transformers.AutoTokenizer.from_pretrained(repo)
    net = transformers.AutoModelForSequenceClassification.from_pretrained(repo).eval()

    @torch.inference_mode()
    def classify(texts: list[str]) -> list[int]:
        out: list[int] = []
        for i in range(0, len(texts), DETECTION_BATCH):
            enc = tok(
                texts[i : i + DETECTION_BATCH],
                padding=True,
                truncation=True,
                max_length=TOKEN_LIMITS[model],
                return_tensors="pt",
            )
            out += net(**enc).logits.argmax(-1).tolist()
        return out

    return lambda text: classify([text]), classify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("side", choices=["ours", "reference"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lid176", "langid", "bert-openlid", "xlm-roberta-lid", "opus-mt"],
    )
    parser.add_argument(
        "--corpus", type=Path, required=True, help="one text per line, none repeated"
    )
    args = parser.parse_args()
    corpus = args.corpus.read_text(encoding="utf-8").splitlines()
    load = _ours if args.side == "ours" else _reference
    for model in args.models:
        one, many = load(model)
        if model == "opus-mt":
            result = _translation(one, many)
        else:
            result = _detection(one, many, corpus, model)
        rounded = {key: round(value, 1) for key, value in result.items()}
        print(json.dumps({"side": args.side, "model": model, **rounded}), flush=True)


if __name__ == "__main__":
    main()
