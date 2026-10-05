from pathlib import Path

from noentenc._optional import TRANSLATION_EXTRA, require


def resolve_files(
    model: str | Path,
    filenames: list[str],
    only_local_files: bool = False,
    revision: str | None = None,
) -> dict[str, Path]:
    """Map each relative filename to a local path.

    `model` is either a local directory or a Hugging Face repo id. For a repo,
    only the requested files are downloaded at `revision` (a commit; `None` means
    the latest), or read from the local cache when `only_local_files` is set.
    """
    local_dir = Path(model)
    if local_dir.is_dir():
        paths = {name: local_dir / name for name in filenames}
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{local_dir} is missing {missing}")
        return paths
    hub = require("huggingface_hub", TRANSLATION_EXTRA)
    return {
        name: Path(
            hub.hf_hub_download(
                str(model),
                name,
                revision=revision,
                local_files_only=only_local_files,
            )
        )
        for name in filenames
    }
