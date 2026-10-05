"""Download the weights a profile needs ahead of time, so deployments can run offline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from noentenc.languages import Language
from noentenc.profiles import Profile

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# A translation pair to prepare: (source, target). A `None` source is translation
# without a source language, as in `Translator.translate(text, target)`.
Pair = tuple[Language | str | None, Language | str]


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
    for it, without loading it. Files go to `cache_dir` (see `noentenc._cache` for the
    default), where the same classes then find them with `only_local_files=True`:

        prepare("speed", translation=[("en", "es"), (None, "en")], cache_dir="/models")
        LanguageDetector(only_local_files=True, cache_dir="/models")
        Translator(only_local_files=True, cache_dir="/models")

    Cached files are reused; checksummed detection files are checked and downloaded
    again if they don't match. `force` downloads everything again, which also replaces
    a corrupt file. Returns the local path of every file.
    """
    profile = Profile(profile)
    pairs = [_pair(source, target) for source, target in translation]
    paths: list[Path] = []
    if detection:
        from noentenc.language_detection._download import fetch
        from noentenc.language_detection.base import profile_remote_files

        paths += [
            fetch(remote, cache_dir=cache_dir, force=force, verify=True)
            for remote in profile_remote_files(profile)
        ]
    if pairs:
        from noentenc.translation._routing import choose

        choices = list(dict.fromkeys(choose(profile, s, t) for s, t in pairs))
        for choice in choices:
            paths += choice.download(cache_dir=cache_dir, force=force).values()
    return list(dict.fromkeys(paths))


def _pair(
    source: Language | str | None, target: Language | str
) -> tuple[Language | None, Language]:
    return (None if source is None else Language(source)), Language(target)
