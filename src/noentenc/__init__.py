from typing import TYPE_CHECKING, Any

from noentenc._plan import ModelConstraintError, ModelPlan, Plan
from noentenc.languages import (
    ANY_LANGUAGE,
    Language,
    LanguageSchema,
    UnsupportedLanguageError,
    to_language,
)
from noentenc.profiles import Profile

if TYPE_CHECKING:
    from noentenc._prepare import plan, prepare

__all__ = [
    "ANY_LANGUAGE",
    "Language",
    "LanguageSchema",
    "ModelConstraintError",
    "ModelPlan",
    "Plan",
    "Profile",
    "UnsupportedLanguageError",
    "plan",
    "prepare",
    "to_language",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module attributes
    # Imported on first use, so `import noentenc` stays light.
    if name in ("plan", "prepare"):
        from noentenc import _prepare

        return getattr(_prepare, name)
    raise AttributeError(f"module 'noentenc' has no attribute {name!r}")
