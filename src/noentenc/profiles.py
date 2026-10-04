from enum import StrEnum


class Profile(StrEnum):
    """The latency/quality trade-off of the models `Translator` and `LanguageDetector` pick.

    - `speed`: the lowest latency and smallest downloads. The default.
    - `balance`: much better output on less common languages, for more latency and
      downloads of about a gigabyte.
    - `quality`: the best output, with the largest downloads (up to 3.5 GB).

    The models behind each profile are listed in docs/benchmarks.md and may change between
    releases; pass a model explicitly for reproducible output.
    """

    SPEED = "speed"
    BALANCE = "balance"
    QUALITY = "quality"
