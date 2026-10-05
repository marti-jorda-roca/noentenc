"""The core install (numpy and tqdm) runs detection without the translation dependencies."""

import subprocess
import sys
import textwrap
from pathlib import Path

FIXTURE = (
    Path(__file__).parent
    / "language_detection"
    / "fixtures"
    / "fasttext"
    / "tiny-softmax.bin"
)

# Makes the translation and ONNX dependencies unimportable, as in a core-only install.
_BLOCK_IMPORTS = """
import sys
from importlib.abc import MetaPathFinder

BLOCKED = {"onnxruntime", "tokenizers", "huggingface_hub"}

class Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, Block())
"""


def run_without_optional_dependencies(code: str) -> None:
    script = _BLOCK_IMPORTS + textwrap.dedent(code)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_detection_runs_without_translation_dependencies() -> None:
    run_without_optional_dependencies(
        f"""
        import noentenc
        import noentenc.translation
        from noentenc.language_detection import FastTextModel, LanguageDetector

        detector = LanguageDetector(FastTextModel({str(FIXTURE)!r}))
        assert detector.detect("der hund lief im park") == "deu"
        assert detector.detect_batch(["el perro corrió", ""]) == ["spa", "und"]
        loaded = BLOCKED & {{name.split(".")[0] for name in sys.modules}}
        assert not loaded, loaded
        """
    )


def test_missing_extras_name_the_install_command(tmp_path: Path) -> None:
    run_without_optional_dependencies(
        f"""
        from noentenc import Language
        from noentenc.language_detection import OnnxClassifierModel
        from noentenc.translation import OpusMTModel, Translator

        def error(build):
            try:
                build()
            except ImportError as exc:
                return str(exc)
            raise AssertionError("no ImportError")

        assert "uv add 'noentenc[onnx]'" in error(
            lambda: OnnxClassifierModel({str(tmp_path)!r})
        )
        assert "uv add 'noentenc[translation]'" in error(OpusMTModel)
        assert "uv add 'noentenc[translation]'" in error(
            lambda: Translator().translate("Hello", Language.SPANISH, Language.ENGLISH)
        )
        """
    )
