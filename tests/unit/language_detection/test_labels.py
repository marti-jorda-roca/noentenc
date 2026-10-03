import numpy as np
import pytest

from noentenc.language_detection.labels import (
    LabelMapper,
    normalize_scores,
    to_iso639_3,
    valid_iso639_3_codes,
)
from noentenc.language_detection.models.heliport_model import heliport_code
from tests.unit.language_detection.helpers import FIXTURES

# Labels that are not (current) ISO 639-3 individual/macro codes but are what the model emits:
# GlotLID's `daf` (retired by a split) and `oto` (an ISO 639-5 family code).
KNOWN_NON_ISO = {"daf", "oto"}


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("__label__en", "eng"),
        ("__label__eng_Latn", "eng"),
        ("zho_Hant", "zho"),
        ("zh-Latn", "zho"),
        ("zh-cn", "zho"),
        ("EN", "eng"),
        ("iw", "heb"),
        ("__label__als", "gsw"),
        ("__label__sh", "hbs"),
        ("__label__bh", "bho"),
        ("und_Latn", "und"),
        ("zxx_Zzzz", "zxx"),
        ("cmn", "cmn"),
    ],
)
def test_to_iso639_3(label: str, expected: str) -> None:
    assert to_iso639_3(label) == expected


def test_collapse_macrolanguages() -> None:
    assert to_iso639_3("arb_Arab", collapse_macrolanguages=True) == "ara"
    assert to_iso639_3("cmn", collapse_macrolanguages=True) == "zho"
    assert to_iso639_3("ekk_Latn", collapse_macrolanguages=True) == "est"
    assert to_iso639_3("eng_Latn", collapse_macrolanguages=True) == "eng"


@pytest.mark.parametrize(
    "path", sorted((FIXTURES / "labels").glob("*.txt")), ids=lambda p: p.stem
)
def test_every_backend_label_maps_to_iso639_3(path) -> None:  # noqa: ANN001
    labels = path.read_text(encoding="utf-8").split()
    if path.stem == "heliport":
        labels = [heliport_code(label) for label in labels]
    unmapped = (
        {to_iso639_3(label) for label in labels}
        - valid_iso639_3_codes()
        - KNOWN_NON_ISO
    )
    assert not unmapped


def test_label_mapper_sums_labels_that_collapse() -> None:
    mapper = LabelMapper(["zho_Hans", "eng_Latn", "zho_Hant"])
    assert not mapper.identity
    assert mapper.labels == ["eng", "zho"]
    probs = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    np.testing.assert_allclose(mapper.reduce(probs), [[0.5, 0.5], [0.1, 0.9]])
    assert mapper.to_dicts(mapper.reduce(probs), top_k=1) == [
        {"eng": 0.5},
        {"zho": pytest.approx(0.9)},
    ]


def test_label_mapper_identity_keeps_native_order() -> None:
    mapper = LabelMapper(["__label__en", "__label__es"])
    assert mapper.identity and mapper.labels == ["eng", "spa"]
    probs = np.array([[0.3, 0.7]])
    assert mapper.reduce(probs) is probs
    assert mapper.to_dicts(probs, top_k=None) == [{"spa": 0.7, "eng": 0.3}]


def test_label_mapper_without_normalisation() -> None:
    assert LabelMapper(["__label__eng_Latn"], normalize=False).labels == ["eng_Latn"]


def test_normalize_scores_sums_collisions() -> None:
    assert normalize_scores({"zh-cn": 0.25, "zh-tw": 0.5, "en": 0.25}) == {
        "zho": 0.75,
        "eng": 0.25,
    }
