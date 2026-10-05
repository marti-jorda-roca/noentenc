"""Download a profile's weights into one directory, then use them without network access.

Run: uv run python docs/examples/prepare_offline.py
Downloads lid176 (0.9 MB), Opus-MT en→es (287 MB) and SMaLL-100 (595 MB) on first run.
"""

import tempfile
from pathlib import Path

import noentenc
from noentenc import Language
from noentenc.language_detection import LanguageDetector
from noentenc.translation import Translator

models = Path(tempfile.gettempdir()) / "noentenc-models"

# 1. Where the network is available, e.g. while building a container image. Re-running
#    reuses what is already there, and re-downloads detection files that fail their
#    checksum.
paths = noentenc.prepare(
    "speed",
    translation=[
        (Language.ENGLISH, Language.SPANISH),  # Opus-MT en→es
        (None, Language.ENGLISH),  # any language into English: SMaLL-100
    ],
    cache_dir=models,
)
print(f"{len(paths)} files under {models}")
# 10 files under <your temp directory>/noentenc-models

# 2. In the deployment, with that directory mounted or copied. Nothing downloads now.
detector = LanguageDetector(only_local_files=True, cache_dir=models)
translator = Translator(only_local_files=True, cache_dir=models)

print(detector.detect("Bon dia! Com estàs?"))
# cat
print(
    translator.translate(
        "The weather is nice today.", Language.SPANISH, Language.ENGLISH
    )
)
# El tiempo es bueno hoy.
print(translator.translate("Bon dia a tothom!", Language.ENGLISH))
# Good day to everyone!

# 3. Anything that wasn't prepared fails at once, saying what to prepare.
try:
    translator.translate("Hello", Language.GERMAN, Language.ENGLISH)
except FileNotFoundError as error:
    print(type(error).__name__)
# FileNotFoundError
