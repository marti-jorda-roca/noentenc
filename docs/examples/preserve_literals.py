"""Translate support messages without breaking their links, emails, code and placeholders.

Brand and product names, and patterns such as SKUs, can be kept too (`keep=`), and
`preserve=` chooses which kinds of literal to keep.

Run: uv run python docs/examples/preserve_literals.py
Downloads Opus-MT en→es (287 MB) on first run.
"""

import re

from noentenc import Language
from noentenc.translation import LiteralKind, Translator

translator = Translator()
messages = [
    "Visit https://example.com/reset?id=42 or email support@acme.io",
    "Hi {name}, your code is 4821 and it expires at 14:30 on 12/05/2026.",
    "Run `pip install noentenc` and ping @marti in #support.",
    "Read [the guide](https://docs.acme.io/start). It takes 5 minutes.",
    'Click <a href="https://acme.io/help">here</a> to open a ticket.',
]

# Literals come back byte-for-byte; the prose around them is translated.
for result in translator.translate_batch(
    messages, Language.SPANISH, Language.ENGLISH, detailed=True
):
    print(result.preservation, "|", result.text)
# placeholders | Visite https://example.com/reset?id=42 o envíe un correo electrónico a support@acme.io
# placeholders | Hola {name}, tu código es 4821 y expira en 14:30 en 12/05/2026.
# placeholders | Ejecutar `pip install noentenc` y la configuración @marti en #support.
# placeholders | Lea [la guía](https://docs.acme.io/start). Se tarda 5 minutos.
# placeholders | Haga clic en <a href="https://acme.io/help">aquí</a> para abrir un ticket.

# Without preservation the model translates whatever it sees, the URL's domain included.
print(
    translator.translate(
        messages[0], Language.SPANISH, Language.ENGLISH, preserve=False
    )
)
# Visite https://ejemplo.com/reset?id=42 o envíe un correo electrónico a support@acme.io

# Your own terms: brand names as whole words, and regular expressions.
branded = Translator(keep=["Apple Watch", re.compile(r"SKU-\d+")])
print(
    branded.translate(
        "Your Apple Watch (SKU-4821) arrives tomorrow.",
        Language.SPANISH,
        Language.ENGLISH,
    )
)
# Su Apple Watch (SKU-4821) llega mañana.

# Code blocks never reach the model, and paths and identifiers are kept too.
print(
    translator.translate(
        "Open ~/app/settings.py and set debug_mode:\n\n```\ndebug_mode = True\n```\n\n"
        "Then restart the server.",
        Language.SPANISH,
        Language.ENGLISH,
    )
)
# Abra ~/app/settings.py y establezca debug_mode:
#
# ```
# debug_mode = True
# ```
#
# A continuación, reinicie el servidor.

# Keep only some kinds. A masked number can't agree with the words around it, so
# leaving numbers to the model reads better, at the cost of their format.
no_numbers = set(LiteralKind) - {LiteralKind.NUMBER}
recipe = "Mix 200 grams of flour with 3 eggs."
print(translator.translate(recipe, Language.SPANISH, Language.ENGLISH))
# Mezclar los gramos 200 de harina con los huevos 3.
print(
    translator.translate(
        recipe, Language.SPANISH, Language.ENGLISH, preserve=no_numbers
    )
)
# Mezclar 200 gramos de harina con 3 huevos.

# What a call would keep, without loading a model.
for span in translator.find_literals(messages[1]):
    print(span.kind, span.text)
# placeholder {name}
# number 4821
# number 14:30
# number 12/05/2026
