import pytest

from tests.unit.language_detection.helpers import load_sentences


@pytest.fixture(scope="session")
def sentences() -> list[str]:
    return load_sentences()
