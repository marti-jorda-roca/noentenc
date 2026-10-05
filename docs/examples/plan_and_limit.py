"""See what a profile would download before it does, and keep it within your policy.

Run: uv run python docs/examples/plan_and_limit.py
Only the last step downloads anything: SMaLL-100 (595 MB) on first run.
"""

import noentenc
from noentenc import ModelConstraintError
from noentenc.translation import Translator

MB = 1 << 20

# 1. A plan names each model a profile would load, with its licence, download size and
#    RAM. Nothing is loaded or downloaded to answer.
plan = noentenc.plan("balance", translation=[("ja", "ca"), ("en", "es")])
for model in plan.models:
    print(
        f"{model.name:45} {model.license:13} "
        f"{model.download_bytes // MB:5} MB download {model.memory_bytes // MB:5} MB RAM"
    )
# FastTextModel(openlid-v3)                     GPL-3.0        1175 MB download  1142 MB RAM
# NLLBModel(Xenova/nllb-200-distilled-600M)     CC-BY-NC-4.0    869 MB download  4057 MB RAM
# OpusMTModel(Xenova/opus-mt-en-es)             Apache-2.0      290 MB download  1084 MB RAM
print(f"{plan.missing_bytes // MB} MB still to download")  # 0 once everything is cached

# memory_basis says where each RAM figure comes from.
print(plan.models[1].memory_basis)
# measured

# 2. Ask which pairs a profile covers, without loading anything.
speed = Translator()
print(speed.supports("es", "en"), speed.supports("ace", "en"))
# True False
print(Translator("balance").supports("ace", "en"))  # NLLB-200 writes Acehnese
# True

# 3. Keep a profile within a licence policy and a download budget. A model that breaks
#    them is skipped for the profile's next choice: here NLLB-200 (non-commercial) gives
#    way to SMaLL-100 (MIT).
open_licences = ["MIT", "Apache-2.0", "CC-BY-4.0"]
translator = Translator("quality", allowed_licenses=open_licences)
print(translator.plan("ca", "ja").models[0].name)
# SMaLL100Model(casawolice/small100-onnx)

# Under a download budget, Opus-MT's default q4 export gives way to its smaller int8 one.
small = Translator(max_download_bytes=200 * MB).plan("es", "en").models[0]
print(small.name, small.precision, small.download_bytes // MB, "MB")
# OpusMTModel(Xenova/opus-mt-en-es) int8 114 MB

# When nothing fits, the call fails before downloading anything.
try:
    Translator(max_download_bytes=50 * MB).translate("Hello", "es", "en")
except ModelConstraintError as error:
    print(type(error).__name__)
# ModelConstraintError

# 4. Constrained translators translate as usual; Catalan→English uses SMaLL-100 here.
print(translator.translate("Bon dia a tothom!", "en", "ca"))
# Good day to everyone!
