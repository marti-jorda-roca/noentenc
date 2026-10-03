"""Lazy imports for optional backends, with an actionable error when the extra is missing."""

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module!r} is required for this model. Install it with: uv add 'noentenc[{extra}]'"
        ) from exc
