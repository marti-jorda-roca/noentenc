from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Final, TypeVar, overload

_T = TypeVar("_T")
_PART1_LENGTH = 2


class Language(StrEnum):
    """Languages supported across noentenc models.

    Values are ISO 639-1 codes, or ISO 639-3 when the language has no 639-1 code.
    """

    ACEHNESE = "ace"
    AFRIKAANS = "af"
    AKAN = "ak"
    ALBANIAN = "sq"
    AMHARIC = "am"
    ARABIC = "ar"
    ARMENIAN = "hy"
    ASSAMESE = "as"
    ASTURIAN = "ast"
    AWADHI = "awa"
    AYMARA = "ay"
    AZERBAIJANI = "az"
    BALINESE = "ban"
    BAMBARA = "bm"
    BANJAR = "bjn"
    BASHKIR = "ba"
    BASQUE = "eu"
    BELARUSIAN = "be"
    BEMBA = "bem"
    BENGALI = "bn"
    BHOJPURI = "bho"
    BOSNIAN = "bs"
    BRETON = "br"
    BUGINESE = "bug"
    BULGARIAN = "bg"
    BURMESE = "my"
    CANTONESE = "yue"
    CATALAN = "ca"
    CEBUANO = "ceb"
    CENTRAL_ATLAS_TAMAZIGHT = "tzm"
    CENTRAL_KURDISH = "ckb"
    CHHATTISGARHI = "hne"
    CHICHEWA = "ny"
    CHINESE = "zh"
    CHOKWE = "cjk"
    CRIMEAN_TATAR = "crh"
    CROATIAN = "hr"
    CZECH = "cs"
    DANISH = "da"
    DARI = "prs"
    DINKA = "dik"
    DUTCH = "nl"
    DYULA = "dyu"
    DZONGKHA = "dz"
    EGYPTIAN_ARABIC = "arz"
    ENGLISH = "en"
    ESPERANTO = "eo"
    ESTONIAN = "et"
    EWE = "ee"
    FAROESE = "fo"
    FIJIAN = "fj"
    FINNISH = "fi"
    FON = "fon"
    FRENCH = "fr"
    FRIULIAN = "fur"
    FULA = "ff"
    GALICIAN = "gl"
    GANDA = "lg"
    GEORGIAN = "ka"
    GERMAN = "de"
    GREEK = "el"
    GUARANI = "gn"
    GUJARATI = "gu"
    HAITIAN_CREOLE = "ht"
    HAUSA = "ha"
    HEBREW = "he"
    HINDI = "hi"
    HUNGARIAN = "hu"
    ICELANDIC = "is"
    IGBO = "ig"
    ILOCANO = "ilo"
    INDONESIAN = "id"
    IRISH = "ga"
    ITALIAN = "it"
    JAPANESE = "ja"
    JAVANESE = "jv"
    JINGPHO = "kac"
    KABIYE = "kbp"
    KABUVERDIANU = "kea"
    KABYLE = "kab"
    KAMBA = "kam"
    KANNADA = "kn"
    KANURI = "kr"
    KASHMIRI = "ks"
    KAZAKH = "kk"
    KHMER = "km"
    KIKONGO = "kg"
    KIKUYU = "ki"
    KIMBUNDU = "kmb"
    KINYARWANDA = "rw"
    KIRUNDI = "rn"
    KOREAN = "ko"
    KURDISH = "ku"
    KYRGYZ = "ky"
    LAO = "lo"
    LATGALIAN = "ltg"
    LATVIAN = "lv"
    LIGURIAN = "lij"
    LIMBURGISH = "li"
    LINGALA = "ln"
    LITHUANIAN = "lt"
    LOMBARD = "lmo"
    LUBA_KASAI = "lua"
    LUO = "luo"
    LUXEMBOURGISH = "lb"
    MACEDONIAN = "mk"
    MAGAHI = "mag"
    MAITHILI = "mai"
    MALAGASY = "mg"
    MALAY = "ms"
    MALAYALAM = "ml"
    MALTESE = "mt"
    MAORI = "mi"
    MARATHI = "mr"
    MEITEI = "mni"
    MESOPOTAMIAN_ARABIC = "acm"
    MINANGKABAU = "min"
    MIZO = "lus"
    MONGOLIAN = "mn"
    MOROCCAN_ARABIC = "ary"
    MOSSI = "mos"
    NAJDI_ARABIC = "ars"
    NEPALI = "ne"
    NORTH_LEVANTINE_ARABIC = "apc"
    NORTHERN_SOTHO = "nso"
    NORWEGIAN = "no"
    NORWEGIAN_NYNORSK = "nn"
    NUER = "nus"
    OCCITAN = "oc"
    ODIA = "or"
    OROMO = "om"
    PANGASINAN = "pag"
    PAPIAMENTO = "pap"
    PASHTO = "ps"
    PERSIAN = "fa"
    POLISH = "pl"
    PORTUGUESE = "pt"
    PUNJABI = "pa"
    QUECHUA = "qu"
    ROMANIAN = "ro"
    RUSSIAN = "ru"
    SAMOAN = "sm"
    SANGO = "sg"
    SANSKRIT = "sa"
    SANTALI = "sat"
    SARDINIAN = "sc"
    SCOTTISH_GAELIC = "gd"
    SERBIAN = "sr"
    SHAN = "shn"
    SHONA = "sn"
    SICILIAN = "scn"
    SILESIAN = "szl"
    SINDHI = "sd"
    SINHALA = "si"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    SOMALI = "so"
    SOUTH_AZERBAIJANI = "azb"
    SOUTH_LEVANTINE_ARABIC = "ajp"
    SOUTHERN_SOTHO = "st"
    SPANISH = "es"
    SUNDANESE = "su"
    SWAHILI = "sw"
    SWATI = "ss"
    SWEDISH = "sv"
    TAGALOG = "tl"
    TAIZZI_ADENI_ARABIC = "acq"
    TAJIK = "tg"
    TAMASHEQ = "taq"
    TAMIL = "ta"
    TATAR = "tt"
    TELUGU = "te"
    THAI = "th"
    TIBETAN = "bo"
    TIGRINYA = "ti"
    TOK_PISIN = "tpi"
    TSONGA = "ts"
    TSWANA = "tn"
    TUMBUKA = "tum"
    TUNISIAN_ARABIC = "aeb"
    TURKISH = "tr"
    TURKMEN = "tk"
    TWI = "tw"
    UKRAINIAN = "uk"
    UMBUNDU = "umb"
    URDU = "ur"
    UYGHUR = "ug"
    UZBEK = "uz"
    VENETIAN = "vec"
    VIETNAMESE = "vi"
    WARAY = "war"
    WELSH = "cy"
    WESTERN_FRISIAN = "fy"
    WOLOF = "wo"
    XHOSA = "xh"
    YIDDISH = "yi"
    YORUBA = "yo"
    ZULU = "zu"


class AnyLanguage:
    """Sentinel type for a schema side that accepts any language."""

    _instance: AnyLanguage | None = None

    def __new__(cls) -> AnyLanguage:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ANY_LANGUAGE"


ANY_LANGUAGE: Final = AnyLanguage()


class UnsupportedLanguageError(ValueError):
    pass


@dataclass(frozen=True)
class LanguageSchema:
    """Languages a model can read (`source`) and write (`target`).

    A side set to `ANY_LANGUAGE` is unrestricted. The source language may be
    omitted when the source side is `ANY_LANGUAGE` or a single language.
    """

    source: frozenset[Language] | AnyLanguage
    target: frozenset[Language] | AnyLanguage

    @property
    def requires_source(self) -> bool:
        if isinstance(self.source, AnyLanguage):
            return False
        return len(self.source) > 1

    def supports(self, source: Language | None, target: Language) -> bool:
        return self._accepts_source(source) and _accepts(self.target, target)

    def validate(
        self, source: Language | None, target: Language, model_name: str
    ) -> None:
        if not _accepts(self.target, target):
            raise UnsupportedLanguageError(
                f"{model_name} cannot translate into {target!r}"
            )
        if source is None and self.requires_source:
            raise UnsupportedLanguageError(f"{model_name} requires a source language")
        if not self._accepts_source(source):
            raise UnsupportedLanguageError(
                f"{model_name} cannot translate from {source!r}"
            )

    def _accepts_source(self, source: Language | None) -> bool:
        if source is None:
            return not self.requires_source
        return _accepts(self.source, source)


def _accepts(side: frozenset[Language] | AnyLanguage, language: Language) -> bool:
    return isinstance(side, AnyLanguage) or language in side


class _Raise:
    """Default of `to_language`: raise on unknown codes."""


_RAISE: Final = _Raise()


@overload
def to_language(value: Language | str) -> Language: ...


@overload
def to_language(value: Language | str, default: _T) -> Language | _T: ...


def to_language(value: Language | str, default: object = _RAISE) -> object:
    """The `Language` for a code or name, e.g. `"spa"`, `"es"`, `"es-ES"` or `"spanish"`.

    Accepts ISO 639-1 and 639-3 codes (including deprecated and retired ones such as
    `"iw"`), BCP-47 and FLORES-style tags (`"pt-BR"`, `"zho_Hans"`; only the language
    part counts) and English names (`"Spanish"`, `"norwegian nynorsk"`). An individual
    language that isn't a member maps to its macrolanguage when that is one: `"cmn"`
    (Mandarin) is `CHINESE`, `"arb"` (Standard Arabic) `ARABIC` and `"nob"` (Bokmål)
    `NORWEGIAN`. Members that are individual languages themselves stay distinct:
    `"yue"` is `CANTONESE` and `"ary"` `MOROCCAN_ARABIC`.

    An unknown value, `"und"` (undetermined) and `"zxx"` (no linguistic content) raise
    `UnsupportedLanguageError`, or return `default` when one is given.
    """
    if isinstance(value, Language):
        return value
    if not isinstance(value, str):
        raise TypeError(f"expected a language code or name, got {type(value).__name__}")
    language = _lookup(value.strip())
    if language is not None:
        return language
    if default is not _RAISE:
        return default
    raise UnsupportedLanguageError(
        f"{value!r} is not a language noentenc knows; pass an ISO 639 code such as "
        "'es' or 'spa', or a Language member"
    )


def _lookup(value: str) -> Language | None:
    by_name = Language.__members__.get(re.sub(r"[\s-]+", "_", value).upper())
    if by_name is not None:
        return by_name
    code = re.split(r"[-_]", value.lower(), maxsplit=1)[0]
    by_value = _by_value().get(code)
    if by_value is not None:
        return by_value
    tables = _iso639_tables()
    part3 = tables.part1_to_part3.get(code) if len(code) == _PART1_LENGTH else code
    if part3 is None:
        return None
    part3 = tables.retired.get(part3, part3)
    found = _by_part3().get(part3)
    if found is None and part3 in tables.macro:
        found = _by_part3().get(tables.macro[part3])
    return found


@dataclass(frozen=True)
class _Iso639Tables:
    part1_to_part3: dict[str, str]
    retired: dict[str, str]
    macro: dict[str, str]


@cache
def _iso639_tables() -> _Iso639Tables:
    from noentenc.language_detection._iso639 import (
        INDIVIDUAL_TO_MACRO,
        PART1_TO_PART3,
        RETIRED_TO_CURRENT,
    )
    from noentenc.language_detection.labels import LABEL_OVERRIDES

    # Deprecated ISO 639-1 codes ("iw", "in", "ji"...) as well as the current ones.
    part1 = {
        **PART1_TO_PART3,
        **{c: p3 for c, p3 in LABEL_OVERRIDES.items() if len(c) == _PART1_LENGTH},
    }
    return _Iso639Tables(part1, RETIRED_TO_CURRENT, INDIVIDUAL_TO_MACRO)


@cache
def _by_value() -> dict[str, Language]:
    return {language.value: language for language in Language}


@cache
def _by_part3() -> dict[str, Language]:
    """Each member by its ISO 639-3 code."""
    part1 = _iso639_tables().part1_to_part3
    return {
        part1.get(language.value, language.value): language for language in Language
    }
