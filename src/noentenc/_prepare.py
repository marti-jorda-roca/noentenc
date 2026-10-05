"""Download the weights a profile needs ahead of time, so deployments can run offline."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from noentenc.languages import Language, UnsupportedLanguageError, to_language
from noentenc.profiles import Profile
from noentenc.translation._routing import ModelChoice, choose

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

# A translation pair to prepare: (source, target). A `None` source is translation
# without a source language, as in `Translator.translate(text, target)`; `"auto"` is
# `source_language="auto"`.
Pair = tuple[Language | str | None, Language | str]
_AUTO = "auto"


def prepare(
    profile: Profile | str = Profile.SPEED,
    *,
    detection: bool = True,
    translation: Iterable[Pair] = (),
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Download what `LanguageDetector(profile)` and `Translator(profile)` would load.

    `detection` covers the profile's detection model. `translation` lists the
    `(source, target)` pairs to translate; each downloads the model the profile picks
    for it, without loading it. A source of `"auto"` downloads the detection model and
    every model the profile could pick for a detected language into that target: for
    English at `speed`, 25 Opus-MT pairs and SMaLL-100, about 7.8 GB. Files go to
    `cache_dir` (see `noentenc._cache` for the default), where the same classes then
    find them with `only_local_files=True`:

        prepare("speed", translation=[("en", "es"), (None, "en")], cache_dir="/models")
        LanguageDetector(only_local_files=True, cache_dir="/models")
        Translator(only_local_files=True, cache_dir="/models")

    Cached files are reused; checksummed detection files are checked and downloaded
    again if they don't match. `force` downloads everything again, which also replaces
    a corrupt file. Returns the local path of every file.
    """
    profile = Profile(profile)
    requests = [_request(source, target) for source, target in translation]
    paths: list[Path] = []
    if detection or any(source == _AUTO for source, _ in requests):
        from noentenc.language_detection._download import fetch
        from noentenc.language_detection.base import profile_remote_files

        paths += [
            fetch(remote, cache_dir=cache_dir, force=force, verify=True)
            for remote in profile_remote_files(profile)
        ]
    choices = dict.fromkeys(
        choice
        for source, target in requests
        for choice in _choices(profile, source, target)
    )
    for choice in choices:
        paths += choice.download(cache_dir=cache_dir, force=force).values()
    return list(dict.fromkeys(paths))


def _request(
    source: Language | str | None, target: Language | str
) -> tuple[Language | str | None, Language]:
    """`(source, target)` with both as `Language`s, except a `None` or `"auto"` source."""
    if source is not None and source != _AUTO:
        source = to_language(source)
    return source, to_language(target)


def _choices(
    profile: Profile, source: Language | str | None, target: Language
) -> Iterator[ModelChoice]:
    if source != _AUTO:
        yield choose(profile, source if source is None else to_language(source), target)
        return
    # Every language a detector may report, and the fallback without a source.
    for candidate in (None, *Language):
        if candidate != target:
            with suppress(UnsupportedLanguageError):
                yield choose(profile, candidate, target)
