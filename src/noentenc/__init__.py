from typing import TYPE_CHECKING, Any

from noentenc.languages import (
    ANY_LANGUAGE,
    Language,
    LanguageSchema,
    UnsupportedLanguageError,
)
from noentenc.profiles import Profile

if TYPE_CHECKING:
    from noentenc._prepare import prepare

__all__ = [
    "ANY_LANGUAGE",
    "Language",
    "LanguageSchema",
    "Profile",
    "UnsupportedLanguageError",
    "prepare",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401 - module attributes
    # Imported on first use, so `import noentenc` stays light.
    if name == "prepare":
        from noentenc._prepare import prepare

        return prepare
    raise AttributeError(f"module 'noentenc' has no attribute {name!r}")
