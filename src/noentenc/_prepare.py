"""Plan and download the weights a profile needs ahead of time, e.g. to run offline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from noentenc._plan import Constraints, Plan
from noentenc.languages import Language, to_language
from noentenc.profiles import Profile
from noentenc.translation._routing import ModelChoice, auto_choices, choose

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# A translation pair to prepare: (source, target). A `None` source is translation
# without a source language, as in `Translator.translate(text, target)`; `"auto"` is
# `source_language="auto"`.
Pair = tuple[Language | str | None, Language | str]
_AUTO = "auto"


def plan(
    profile: Profile | str = Profile.SPEED,
    *,
    detection: bool = True,
    translation: Iterable[Pair] = (),
    cache_dir: str | Path | None = None,
    allowed_licenses: Iterable[str] | None = None,
    max_download_bytes: int | None = None,
) -> Plan:
    """What `prepare` with the same arguments would download, without downloading it.

    Each model comes with its licence, download size, memory and whether it is cached.
    Raises `ModelConstraintError` when a requested pair or the detector has no model
    that meets `allowed_licenses` and `max_download_bytes`.
    """
    from noentenc.language_detection.base import check_profile_plan

    profile, constraints, detect, choices = _select(
        profile, detection, translation, allowed_licenses, max_download_bytes
    )
    models = [check_profile_plan(profile, constraints, cache_dir)] if detect else []
    models += [choice.plan(cache_dir) for choice in choices]
    return Plan(tuple(models))


def prepare(
    profile: Profile | str = Profile.SPEED,
    *,
    detection: bool = True,
    translation: Iterable[Pair] = (),
    cache_dir: str | Path | None = None,
    force: bool = False,
    allowed_licenses: Iterable[str] | None = None,
    max_download_bytes: int | None = None,
) -> list[Path]:
    """Download what `LanguageDetector(profile)` and `Translator(profile)` would load.

    `detection` covers the profile's detection model. `translation` lists the
    `(source, target)` pairs to translate; each downloads the model the profile picks
    for it, without loading it. A source of `"auto"` downloads the detection model and
    every model the profile could pick for a detected language into that target: for
    English at `speed`, 25 Opus-MT pairs and SMaLL-100, 7.9 GB. Files go to
    `cache_dir` (see `noentenc._cache` for the default), where the same classes then
    find them with `only_local_files=True`:

        prepare("speed", translation=[("en", "es"), (None, "en")], cache_dir="/models")
        LanguageDetector(only_local_files=True, cache_dir="/models")
        Translator(only_local_files=True, cache_dir="/models")

    `allowed_licenses` and `max_download_bytes` pick models as the same arguments to
    `Translator` and `LanguageDetector` do, and raise `ModelConstraintError` before
    anything downloads when a pair has no allowed model. `plan(...)` with the same
    arguments says what would be downloaded.

    Cached files are reused; checksummed detection files are checked and downloaded
    again if they don't match. `force` downloads everything again, which also replaces
    a corrupt file. Returns the local path of every file.
    """
    from noentenc.language_detection._download import fetch
    from noentenc.language_detection.base import (
        check_profile_plan,
        profile_remote_files,
    )

    profile, constraints, detect, choices = _select(
        profile, detection, translation, allowed_licenses, max_download_bytes
    )
    paths: list[Path] = []
    if detect:
        check_profile_plan(profile, constraints, cache_dir)
        paths += [
            fetch(remote, cache_dir=cache_dir, force=force, verify=True)
            for remote in profile_remote_files(profile)
        ]
    for choice in choices:
        paths += choice.download(cache_dir=cache_dir, force=force).values()
    return list(dict.fromkeys(paths))


def _select(
    profile: Profile | str,
    detection: bool,
    translation: Iterable[Pair],
    allowed_licenses: Iterable[str] | None,
    max_download_bytes: int | None,
) -> tuple[Profile, Constraints, bool, list[ModelChoice]]:
    """The profile, constraints, whether detection is needed, and the translation models.

    Every choice is made, and every constraint checked, before anything downloads.
    """
    profile = Profile(profile)
    constraints = Constraints.of(allowed_licenses, max_download_bytes)
    requests = [_request(source, target) for source, target in translation]
    detect = detection or any(auto for _, _, auto in requests)
    choices = dict.fromkeys(
        choice
        for source, target, auto in requests
        for choice in (
            auto_choices(profile, target, constraints)
            if auto
            else [choose(profile, source, target, constraints)]
        )
    )
    return profile, constraints, detect, list(choices)


def _request(
    source: Language | str | None, target: Language | str
) -> tuple[Language | None, Language, bool]:
    """`(source, target, auto)` as `Language`s; the source is None when not given or `"auto"`."""
    auto = source == _AUTO
    if source is None or auto:
        return None, to_language(target), auto
    return to_language(source), to_language(target), False
