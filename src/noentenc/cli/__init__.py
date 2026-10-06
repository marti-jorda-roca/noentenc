"""The `noentenc` command. It needs the `cli` extra: `uv add 'noentenc[cli]'`.

docs/cli.md documents the commands, their output and exit codes.
"""

import importlib.util
import sys

CLI_EXTRA = "cli"


def main() -> None:
    # The script is installed with every wheel, but click only comes with the extra.
    if importlib.util.find_spec("click") is None:
        sys.stderr.write(
            "The noentenc command needs click. Install it with: "
            f"uv add 'noentenc[{CLI_EXTRA}]'\n"
        )
        raise SystemExit(1)
    from noentenc.cli._app import cli

    cli()
