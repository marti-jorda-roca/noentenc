"""Keep a long-running translator's memory bounded.

Run: uv run python docs/examples/manage_memory.py
Downloads Opus-MT en→es, en→de and de→en (287 MB each) on first run.
"""

from noentenc import Language
from noentenc.translation import Translator

EN, ES, DE = Language.ENGLISH, Language.SPANISH, Language.GERMAN

# A translator loads a model the first time a pair needs it and keeps it for later
# calls, at most `max_loaded_models` of them (2 by default). Each Opus-MT model takes
# about 1.1 GB of RAM at its default precision; docs/benchmarks.md#memory has the rest.
translator = Translator(max_loaded_models=2)

translator.translate("Good morning", ES, EN)
translator.translate("Good morning", DE, EN)
print([model.model for model in translator.loaded_models])
# ['Xenova/opus-mt-en-es', 'Xenova/opus-mt-en-de']

# A third pair drops the least recently used model (en→es) before loading its own.
translator.translate("Guten Morgen", EN, DE)
print([model.model for model in translator.loaded_models])
# ['Xenova/opus-mt-en-de', 'Xenova/opus-mt-de-en']

# en→es is loaded again when it's needed; the results don't change.
print(translator.translate("Good morning", ES, EN))
# Buenos días.

# Free every model, e.g. between batch jobs. The translator stays usable.
translator.unload()
print(translator.loaded_models)
# []

# Calls that keep alternating between more pairs than the limit reload models each
# time. Raise the limit, or pass None for no limit, if the memory is there.
roomy = Translator(max_loaded_models=None)
