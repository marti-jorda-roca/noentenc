"""Translate a mixed-language inbox into English in one call.

`source_language="auto"` detects each message's language, sends each language to the
model the profile picks for it (Opus-MT where the pair has one, SMaLL-100 otherwise),
and returns the results in input order.

Run: uv run python docs/examples/translate_to_english.py
Downloads Opus-MT es→en, de→en and zh→en, and SMaLL-100, on first run.
"""

import polars as pl

from noentenc import Language
from noentenc.translation import TranslationStatus, Translator

messages = [
    "Hola, ¿cuándo llega mi pedido?",
    "Der Link im Newsletter funktioniert nicht.",
    "Thanks, the issue is fixed now.",
    "Bon dia, la factura d'aquest mes està malament.",
    "Bon dia, no puc entrar al meu compte.",
    "我的订单什么时候到？",
    "lol",
    "https://example.com/order/1234",
    "¿Puedo cambiar la dirección de entrega?",
]

translator = Translator()

# Plain strings, in input order. English messages and links come back unchanged.
for message, english in zip(
    messages,
    translator.translate_batch(messages, Language.ENGLISH, "auto"),
    strict=True,
):
    print(f"{message!r:45} -> {english!r}")
# 'Hola, ¿cuándo llega mi pedido?'              -> 'Hey, when does my order arrive?'
# 'Der Link im Newsletter funktioniert nicht.'  -> 'The link in the newsletter does not work.'
# 'Thanks, the issue is fixed now.'             -> 'Thanks, the issue is fixed now.'
# "Bon dia, la factura d'aquest mes està malament." -> 'The bill this month is bad.'
# 'Bon dia, no puc entrar al meu compte.'       -> 'Bon dia, no puc entrar al meu compte.'
# '我的订单什么时候到？'                        -> 'When will my orders arrive?'
# 'lol'                                         -> 'lol'
# 'https://example.com/order/1234'              -> 'https://example.com/order/1234'
# '¿Puedo cambiar la dirección de entrega?'     -> 'Can I change the delivery address?'

# Detailed results say what was detected, which model ran, and which messages were left
# alone. "lol" is too short to trust the detector. The second Catalan message is kept as
# given too: lid176 thinks it is Portuguese, but only with a score of 0.35, below the
# detector's 0.5 threshold. Pass unknown_source="fallback" to translate such messages
# without a source language, or "raise" to fail instead.
for result in translator.translate_batch(
    messages, Language.ENGLISH, "auto", detailed=True
):
    if result.status is TranslationStatus.TRANSLATED:
        print(result.source_language, result.model)
    else:
        print(result.status, result.detected_language)
# es OpusMTModel(Xenova/opus-mt-es-en)
# de OpusMTModel(Xenova/opus-mt-de-en)
# unchanged eng
# ca SMaLL100Model(casawolice/small100-onnx)
# unknown_source und
# zh OpusMTModel(Xenova/opus-mt-zh-en)
# unknown_source und
# unchanged zxx
# es OpusMTModel(Xenova/opus-mt-es-en)

# The same works on a DataFrame column, with the status and source of each row.
inbox = pl.DataFrame({"message": [*messages, None]})
print(
    translator.translate_dataset(
        inbox,
        "message",
        "english",
        Language.ENGLISH,
        "auto",
        status_column="status",
        source_column="source",
        show_progress=False,
    ).select("source", "status", "english")
)
# source  status          english
# es      translated      Hey, when does my order arrive?
# de      translated      The link in the newsletter does not work.
# en      unchanged       Thanks, the issue is fixed now.
# ca      translated      The bill this month is bad.
# null    unknown_source  Bon dia, no puc entrar al meu compte.
# zh      translated      When will my orders arrive?
# null    unknown_source  lol
# null    unchanged       https://example.com/order/1234
# es      translated      Can I change the delivery address?
# null    null            null
