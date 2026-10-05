from pathlib import Path

from noentenc._optional import TRANSLATION_EXTRA, require


def resolve_files(
    model: str | Path,
    filenames: list[str],
    only_local_files: bool = False,
    revision: str | None = None,
    *,
    cache_dir: Path,
    force: bool = False,
) -> dict[str, Path]:
    """Map each relative filename to a local path.

    `model` is either a local directory or a Hugging Face repo id. For a repo,
    only the requested files are downloaded at `revision` (a commit; `None` means
    the latest) into the Hugging Face cache `cache_dir`, or read from it when
    `only_local_files` is set. `force` downloads them again.
    """
    local_dir = Path(model)
    if local_dir.is_dir():
        paths = {name: local_dir / name for name in filenames}
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{local_dir} is missing {missing}")
        return paths
    hub = require("huggingface_hub", TRANSLATION_EXTRA)
    paths: dict[str, Path] = {}
    for name in filenames:
        try:
            paths[name] = Path(
                hub.hf_hub_download(
                    str(model),
                    name,
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=only_local_files,
                    force_download=force,
                )
            )
        except FileNotFoundError as error:  # LocalEntryNotFoundError, offline only
            if not only_local_files:
                raise
            raise FileNotFoundError(
                f"{model}/{name} (revision {revision or 'latest'}) is not in {cache_dir} and "
                "only_local_files=True. Download it first with noentenc.prepare(...) "
                "using the same cache directory, or copy a prepared cache there."
            ) from error
    return paths
