"""The `noentenc` commands: detect, translate, download, list and info.

Results go to stdout, one line per input text and in input order. Download progress,
warnings and errors go to stderr, so stdout stays parseable. Exit codes:

- 0: success.
- 1: any other error, such as a failed download or a missing extra.
- 2: invalid arguments: an unknown option, profile or language, or a pair no model
  translates.
- 3: a model isn't cached and `--offline` was given.
- 4: some texts failed to translate (their lines hold the original text), or
  `--unknown-source raise` met a text whose language can't be used.
"""

from __future__ import annotations

import io
import itertools
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any, NoReturn, TypeVar

import click

from noentenc._cache import cache_root, translation_cache
from noentenc._plan import MB, Plan
from noentenc._prepare import Pair, plan, prepare
from noentenc.cli import _inventory
from noentenc.languages import UnsupportedLanguageError
from noentenc.profiles import Profile
from noentenc.translation import (
    SourceLanguageError,
    Translation,
    TranslationStatus,
    Translator,
)
from noentenc.translation.base import UnknownSourcePolicy

ERROR = 1
USAGE = 2
NOT_CACHED = 3
TEXTS_FAILED = 4

_GB = 1 << 30
_AUTO = "auto"

F = TypeVar("F", bound=Callable[..., Any])


class _Group(click.Group):
    """Turns the library's exceptions into a message on stderr and an exit code."""

    def invoke(self, ctx: click.Context) -> Any:  # noqa: ANN401 - click's signature
        try:
            return super().invoke(ctx)
        except BrokenPipeError:
            # The reader stopped early, e.g. `| head`: stop without a traceback.
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(ERROR)
        except FileNotFoundError as error:
            _fail(
                f"{error}\nRun `noentenc download` with the same --profile and "
                "--cache-dir first.",
                NOT_CACHED,
            )
        except SourceLanguageError as error:
            _fail(str(error), TEXTS_FAILED)
        except (UnsupportedLanguageError, ValueError) as error:
            _fail(str(error), USAGE)
        except (ImportError, OSError) as error:
            _fail(str(error), ERROR)


def _fail(message: str, code: int) -> NoReturn:
    click.echo(f"Error: {message}", err=True)
    sys.exit(code)


@click.group(cls=_Group)
@click.version_option(package_name="noentenc")
def cli() -> None:
    """Detect languages and translate text on CPU.

    Results go to stdout, one line per text in input order; progress and errors go to
    stderr. See docs/cli.md for examples and exit codes.
    """
    _use_utf8()


def _profile_options(function: F) -> F:
    options = (
        click.option(
            "--profile",
            type=click.Choice([str(profile) for profile in Profile]),
            default=str(Profile.SPEED),
            show_default=True,
            help="Latency/quality trade-off of the models to use.",
        ),
        click.option(
            "--cache-dir",
            type=click.Path(file_okay=False, path_type=Path),
            help="Model cache. Default: $NOENTENC_CACHE, else ~/.cache/noentenc.",
        ),
    )
    for option in reversed(options):
        function = option(function)
    return function


def _input_options(function: F) -> F:
    options = (
        click.argument("texts", nargs=-1),
        click.option(
            "--file",
            type=click.File("r", encoding="utf-8"),
            metavar="FILE",
            help="Read one text per line from FILE ('-' for stdin).",
        ),
        click.option(
            "--json",
            "as_json",
            is_flag=True,
            help="Print one JSON object per text, with its status and scores.",
        ),
        click.option(
            "--offline",
            is_flag=True,
            help="Only use cached models: exit with 3 instead of downloading one.",
        ),
    )
    for option in reversed(options):
        function = option(function)
    return function


@cli.command()
@_input_options
@click.option(
    "--min-letters",
    type=click.IntRange(min=0),
    help="Answer 'und' for texts with fewer letters than this.",
)
@click.option(
    "--min-score",
    type=click.FloatRange(min=0),
    help="Answer 'und' when the top score is below this.",
)
@click.option(
    "--min-margin",
    type=click.FloatRange(min=0),
    help="Answer 'und' when the top two scores are closer than this.",
)
@click.option(
    "--candidates",
    metavar="CODES",
    help="Comma-separated languages the answer must be one of, e.g. eng,spa,cat.",
)
@_profile_options
def detect(
    *,
    texts: tuple[str, ...],
    file: IO[str] | None,
    as_json: bool,
    offline: bool,
    min_letters: int | None,
    min_score: float | None,
    min_margin: float | None,
    candidates: str | None,
    profile: str,
    cache_dir: Path | None,
) -> None:
    """Print the language of each text as an ISO 639-3 code.

    Texts come from the arguments, from --file, or one per line from stdin. 'und' means
    undetermined (empty text, or a threshold wasn't met) and 'zxx' no linguistic
    content. --json adds each text, its status and the model's top label and score.

    \b
      noentenc detect "Bon dia!" "Hello there"
      noentenc detect --file comments.txt | sort | uniq -c
    """
    from noentenc.language_detection import LanguageDetector

    lines = _read_texts(texts, file)
    detector = LanguageDetector(
        profile,
        min_letters=min_letters,
        min_score=min_score,
        min_margin=min_margin,
        candidates=_split(candidates),
        only_local_files=offline,
        cache_dir=cache_dir,
    )
    inputs, feed = itertools.tee(lines)
    results = detector.detect_stream(feed, detailed=True)
    for text, detection in zip(inputs, results, strict=True):
        if as_json:
            _write(_json({"text": text, **asdict(detection)}))
        else:
            _write(detection.language)


def _check_terms(
    _ctx: click.Context, _param: click.Parameter, values: Sequence[str]
) -> list[str]:
    if any(not value.strip() for value in values):
        raise click.BadParameter("terms can't be empty or whitespace")
    return list(values)


def _compile_patterns(
    _ctx: click.Context, _param: click.Parameter, values: Sequence[str]
) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for value in values:
        try:
            patterns.append(re.compile(value))
        except re.error as error:
            raise click.BadParameter(
                f"{value!r} isn't a valid regex: {error}"
            ) from None
    return patterns


@cli.command()
@_input_options
@click.option(
    "--to",
    "target",
    required=True,
    metavar="LANGUAGE",
    help="Language to translate into, as a code or a name: en, spa, es-ES, spanish.",
)
@click.option(
    "--from",
    "source",
    default=_AUTO,
    metavar="LANGUAGE",
    show_default=True,
    help="Language of the texts; 'auto' detects each text's.",
)
@click.option(
    "--unknown-source",
    type=click.Choice(["keep", "fallback", "raise"]),
    default="keep",
    show_default=True,
    help="With --from auto, for texts whose language can't be told or translated: "
    "print them as given, translate them without a source language, or exit with 4.",
)
@click.option(
    "--truncate",
    is_flag=True,
    help="Translate the start of a sentence too long for the model instead of "
    "failing the text.",
)
@click.option(
    "--keep",
    "keep_terms",
    multiple=True,
    metavar="TERM",
    callback=_check_terms,
    help="Keep this term as it is, e.g. a brand name, where it appears as a whole "
    "word. Repeat for more terms.",
)
@click.option(
    "--keep-regex",
    "keep_patterns",
    multiple=True,
    metavar="PATTERN",
    callback=_compile_patterns,
    help="Keep whatever this Python regular expression matches, e.g. 'SKU-[0-9]+'. "
    "Repeat for more patterns.",
)
@click.option(
    "--no-preserve",
    is_flag=True,
    help="Send URLs, emails, code, placeholders and --keep terms to the model like "
    "any other text.",
)
@_profile_options
def translate(
    *,
    texts: tuple[str, ...],
    file: IO[str] | None,
    as_json: bool,
    offline: bool,
    target: str,
    source: str,
    unknown_source: UnknownSourcePolicy,
    truncate: bool,
    keep_terms: list[str],
    keep_patterns: list[re.Pattern[str]],
    no_preserve: bool,
    profile: str,
    cache_dir: Path | None,
) -> None:
    """Translate each text, printing one translation per line.

    Texts come from the arguments, from --file, or one per line from stdin. A text that
    fails to translate is printed as given, its error goes to stderr, and the command
    exits with 4 once every text is done. --json adds each text's status, source
    language, detected language and model.

    \b
      noentenc translate --to en "Hola, ¿qué tal?"
      cat reviews.txt | noentenc translate --to en --json > reviews.jsonl
    """
    lines = _read_texts(texts, file)
    translator = Translator(
        profile,
        only_local_files=offline,
        cache_dir=cache_dir,
        keep=[*keep_terms, *keep_patterns],
    )
    inputs, feed = itertools.tee(lines)
    results = translator.translate_stream(
        feed,
        target,
        source,
        truncate=truncate,
        detailed=True,
        errors="record",
        unknown_source=unknown_source,
        preserve=not no_preserve,
    )
    total = failed = 0
    for text, translation in zip(inputs, results, strict=True):
        total += 1
        if translation.status is TranslationStatus.FAILED:
            failed += 1
            click.echo(f"Text {total} failed: {translation.error}", err=True)
        line = _json(_translation_record(text, translation)) if as_json else None
        _write(line or translation.text)
    if failed:
        sys.stdout.flush()
        _fail(f"{failed} of {total} texts failed to translate", TEXTS_FAILED)


def _translation_record(text: str, translation: Translation) -> dict[str, Any]:
    fields = asdict(translation)
    output = fields.pop("text")
    failed = translation.status is TranslationStatus.FAILED
    return {"text": text, "translation": None if failed else output, **fields}


def _parse_pair(
    _ctx: click.Context, _param: click.Parameter, values: Sequence[str]
) -> list[Pair]:
    pairs: list[Pair] = []
    for value in values:
        source, separator, target = value.partition(":")
        if not separator or not target:
            raise click.BadParameter(
                f"{value!r} isn't SOURCE:TARGET, e.g. es:en, auto:en or :en"
            )
        pairs.append((source or None, target))
    return pairs


@cli.command()
@click.option(
    "--pair",
    "pairs",
    multiple=True,
    metavar="SOURCE:TARGET",
    callback=_parse_pair,
    help="Download the model the profile translates this pair with, e.g. es:en. "
    "'auto:en' downloads every model a detected language could need (several GB), "
    "':en' the one for texts without a source language. Repeat for more pairs.",
)
@click.option(
    "--no-detection", is_flag=True, help="Skip the profile's detection model."
)
@click.option("--force", is_flag=True, help="Download again, replacing cached files.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would be downloaded, with sizes and licences; download nothing.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the models as JSON.")
@_profile_options
def download(
    *,
    pairs: list[Pair],
    no_detection: bool,
    force: bool,
    dry_run: bool,
    as_json: bool,
    profile: str,
    cache_dir: Path | None,
) -> None:
    """Download a profile's models ahead of time, for offline use.

    Downloads the profile's detection model and, for each --pair, the translation
    model the profile picks for it, without loading them. `detect` and `translate`
    with the same --profile and --cache-dir then find them, also with --offline.

    \b
      noentenc download --pair es:en --pair :en --cache-dir /models
      noentenc download --profile balance --pair auto:en --dry-run
    """
    selection: dict[str, Any] = {
        "detection": not no_detection,
        "translation": pairs,
        "cache_dir": cache_dir,
    }
    planned = plan(profile, **selection)
    if not planned.models:
        raise click.UsageError(
            "Nothing to download: pass --pair or drop --no-detection."
        )
    files: list[Path] = []
    if not dry_run:
        click.echo(
            f"Downloading {_size(planned.missing_bytes)} into {cache_root(cache_dir)}",
            err=True,
        )
        files = prepare(profile, force=force, **selection)
        planned = plan(profile, **selection)
    if as_json:
        record = _plan_record(planned)
        if not dry_run:
            record["files"] = [str(path) for path in files]
        _write(_json(record))
        return
    _print_plan(planned)


def _plan_record(planned: Plan) -> dict[str, Any]:
    return {
        "models": [asdict(model) for model in planned.models],
        "download_bytes": planned.download_bytes,
        "missing_bytes": planned.missing_bytes,
        "memory_bytes": planned.memory_bytes,
    }


def _print_plan(planned: Plan) -> None:
    _print_table(
        ("TASK", "MODEL", "PRECISION", "LICENSE", "SIZE", "CACHED"),
        [
            (
                model.task,
                model.name,
                model.precision or "-",
                model.license,
                _size(model.download_bytes),
                _yes_no(model.cached),
            )
            for model in planned.models
        ],
    )
    _write(
        f"\n{_size(planned.download_bytes)} in total, "
        f"{_size(planned.missing_bytes)} not cached yet. "
        f"Loading every model takes about {_size(planned.memory_bytes)} of RAM.",
    )


@cli.command("list")
@click.option(
    "--task",
    type=click.Choice(["detection", "translation"]),
    help="Only list models for this task.",
)
@click.option(
    "--json", "as_json", is_flag=True, help="Print one JSON object per model."
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Model cache to check. Default: $NOENTENC_CACHE, else ~/.cache/noentenc.",
)
def list_models(*, task: str | None, as_json: bool, cache_dir: Path | None) -> None:
    """List the models noentenc can use and download.

    Shows each model's precision, licence, download size, the profiles that use it,
    whether it's cached and the extra it needs. Downloads nothing.

    \b
      noentenc list --task detection
      noentenc list --json | jq -r 'select(.cached) | .name'
    """
    models = [m for m in _inventory.catalog(cache_dir) if task in (None, m.task)]
    if as_json:
        for model in models:
            _write(_json(asdict(model)))
        return
    _print_table(
        (
            "TASK",
            "MODEL",
            "PRECISION",
            "LICENSE",
            "SIZE",
            "PROFILES",
            "CACHED",
            "EXTRA",
        ),
        [
            (
                model.task,
                model.name,
                model.precision or "-",
                model.license,
                "-" if model.download_bytes is None else _size(model.download_bytes),
                ",".join(model.profiles) or "-",
                "-" if model.cached is None else _yes_no(model.cached),
                _extra(model.extra, installed=model.installed),
            )
            for model in models
        ],
    )


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Model cache to inspect. Default: $NOENTENC_CACHE, else ~/.cache/noentenc.",
)
def info(*, as_json: bool, cache_dir: Path | None) -> None:
    """Show where models are cached, which are, and the space they take.

    Only reads the cache directory: nothing is downloaded or loaded.
    """
    from importlib.metadata import version

    models = _inventory.cached_models(cache_dir)
    extras = {
        extra: _inventory.extra_installed(extra) for extra in _inventory.EXTRA_MODULES
    }
    report: dict[str, Any] = {
        "version": version("noentenc"),
        "cache_root": str(cache_root(cache_dir)),
        "cache_root_from": _inventory.cache_source(cache_dir),
        "translation_cache": str(translation_cache(cache_dir)),
        "extras": extras,
        "models": [asdict(model) for model in models],
        "total_bytes": sum(model.size_bytes for model in models),
    }
    if as_json:
        _write(_json(report))
        return
    installed = [extra for extra, present in extras.items() if present]
    for label, value in (
        ("Version", report["version"]),
        ("Cache root", f"{report['cache_root']} ({report['cache_root_from']})"),
        ("Detection", report["cache_root"]),
        ("Translation", report["translation_cache"]),
        ("Extras", ", ".join(installed) or "none"),
    ):
        _write(f"{label + ':':<13}{value}")
    _write("")
    if not models:
        _write("No models cached.")
        return
    _print_table(
        ("TASK", "MODEL", "SIZE", "PATH"),
        [
            (model.task, model.name, _size(model.size_bytes), str(model.path))
            for model in models
        ],
    )
    count = f"{len(models)} model{'s' if len(models) > 1 else ''}"
    _write(f"\n{_size(report['total_bytes'])} in {count}.")


def _read_texts(texts: tuple[str, ...], file: IO[str] | None) -> Iterable[str]:
    """The texts to process: the arguments, or each line of `file` or stdin."""
    if texts and file is not None:
        raise click.UsageError("Pass texts as arguments or with --file, not both.")
    if texts:
        return texts
    if file is None:
        file = sys.stdin
        if file.isatty():
            raise click.UsageError(
                "No input: pass texts as arguments, --file FILE, or lines on stdin."
            )
    return (line.removesuffix("\n").removesuffix("\r") for line in file)


def _split(codes: str | None) -> list[str] | None:
    if codes is None:
        return None
    return [code.strip() for code in codes.split(",") if code.strip()]


def _use_utf8() -> None:
    """Read and write UTF-8 whatever the locale, so pipes decode the same everywhere."""
    for stream in (sys.stdin, sys.stdout):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")


def _write(line: str) -> None:
    sys.stdout.write(line + "\n")


def _json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, default=_json_default)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    # numpy scores
    return float(value)  # ty: ignore[invalid-argument-type]


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(row[i]) for row in (headers, *rows)) for i in range(len(headers))]
    for row in (headers, *rows):
        cells = (cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        _write("  ".join(cells).rstrip())


def _size(size: int) -> str:
    if size == 0:
        return "0 MB"
    if size >= _GB:
        return f"{size / _GB:.1f} GB"
    if size >= 10 * MB:
        return f"{size / MB:.0f} MB"
    return f"{size / MB:.1f} MB"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _extra(extra: str | None, *, installed: bool) -> str:
    if extra is None:
        return "-"
    return extra if installed else f"{extra} (missing)"
