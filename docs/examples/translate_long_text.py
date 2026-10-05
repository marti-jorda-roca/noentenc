"""Translate long text, and decide what happens to a sentence the model can't read whole.

Run: uv run python docs/examples/translate_long_text.py
"""

from noentenc import Language
from noentenc.translation import InputTooLongError, Translator

EN, ES = Language.ENGLISH, Language.SPANISH
translator = Translator()

# 1. Long text is fine. It is split into sentences, they are translated in one batch,
#    and the translations are joined back with the original line breaks.
email = "Hi Ana,\n\nThanks for your help.\n- See you on Monday.\n- Call me later."
print(translator.translate(email, ES, EN))
# Hola Ana,
#
# Gracias por tu ayuda.
# - Nos vemos el lunes.
# - Llámame más tarde.

# 2. The limit is per sentence: about 500 tokens. Text without punctuation or line
#    breaks, such as a pasted log or a transcript, can be one sentence that long.
run_on = " ".join(["the weather is nice"] * 150)

# truncate=False (the default): the sentence is not translated and nothing is lost
# silently. The error says how long the sentence is and where it starts.
try:
    translator.translate(run_on, ES, EN)
except InputTooLongError as error:
    print(error)
    # OpusMTModel reads at most 511 tokens per sentence, and this one has 600: ...

# truncate=True: only the start of the sentence is translated, the rest is dropped.
# Ask for details to find out whether that happened.
result = translator.translate(run_on, ES, EN, truncate=True, detailed=True)
print(result.input_truncated)  # True: part of the input never reached the model
print(result.output_limit_reached)  # True: the output may also be cut short

# 3. Short text never hits the limit, so the flags stay False.
result = translator.translate(email, ES, EN, detailed=True)
print(result.input_truncated, result.output_limit_reached)  # False False

# translate_batch takes the same arguments, and translate_dataset takes truncate=,
# so one oversized row doesn't stop a whole DataFrame.
short, long = translator.translate_batch(["Hello.", run_on], ES, EN, truncate=True)
print(short, long[:40])  # Hola. el tiempo es agradable el tiempo es agra
