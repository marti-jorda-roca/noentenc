"""Translate support messages without breaking their links, emails, code and placeholders.

Run: uv run python docs/examples/preserve_literals.py
Downloads Opus-MT en→es (287 MB) on first run.
"""

from noentenc import Language
from noentenc.translation import Translator

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
