from __future__ import annotations

import re


def pick_lang_value(values_by_lang: dict[str, list[str]], lang: str, default_lang: str, fallback: str) -> str:
    vals = values_by_lang.get(lang) or values_by_lang.get(default_lang) or []
    return vals[0] if vals else fallback


def pick_lang_value_by_text(
    values_by_lang: dict[str, list[str]],
    lang: str,
    text: str,
    default_lang: str,
    fallback: str,
    case_insensitive_langs: set[str] | None = None,
) -> str:
    vals = values_by_lang.get(lang) or values_by_lang.get(default_lang) or []
    flags = re.I if case_insensitive_langs and lang in case_insensitive_langs else 0
    for v in vals:
        m = re.search(v, text, flags=flags)
        if m:
            return m.group(0).strip()
    return vals[0] if vals else fallback


def detect_language(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(r"[\u3040-\u30FF]", text):
        return "ja"
    if re.search(r"[\u4E00-\u9FFF]", text):
        if re.search(r"[가-힣]", text):
            return "ko"
        return "zh"
    if re.search(r"[가-힣]", text):
        return "ko"

    lower = text.lower()
    if any(
        w in lower
        for w in [
            "bonjour",
            "envoyer",
            "equipe",
            "équipe",
            "departement",
            "département",
            "resumer",
            "résumer",
            "mettre a jour",
            "mettre à jour",
        ]
    ):
        return "fr"
    if any(
        w in lower
        for w in [
            "bitte",
            "schicken",
            "sende",
            "senden",
            "fasse",
            "nachrichten",
            "aktualisiere",
            "lieferadresse",
            "abteilung",
            "vertrieb",
            "erstatte",
        ]
    ):
        return "de"
    return "en"


def pick_main_verb(
    text: str,
    lang: str,
    verb_patterns_by_lang: dict[str, list[str]],
    default_verb_candidates_by_lang: dict[str, list[str]],
) -> str:
    patterns = verb_patterns_by_lang.get(lang, verb_patterns_by_lang.get("en", []))
    flags = re.I if lang in {"en", "de", "fr"} else 0
    for p in patterns:
        m = re.search(p, text, flags=flags)
        if m:
            return m.group(0).strip()

    for p in verb_patterns_by_lang.get("en", []):
        m = re.search(p, text, flags=re.I)
        if m:
            return m.group(0).strip()

    return pick_lang_value(default_verb_candidates_by_lang, lang, "en", "request")
