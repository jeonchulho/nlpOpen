from __future__ import annotations

import re


def pick_lang_value(values_by_lang: dict[str, list[str]], lang: str, default_lang: str, fallback: str) -> str:
    """`lang` 우선, 없으면 `default_lang`, 그래도 없으면 `fallback`을 반환합니다.

    예시:
        pick_lang_value({"en": ["request"]}, "fr", "en", "target") -> "request"
    """
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
    """언어별 후보 리스트에서 텍스트에 먼저 매칭되는 값을 반환합니다.

    예시:
        pick_lang_value_by_text({"en": ["send", "forward"]}, "en", "please forward it", "en", "request")
        -> "forward"
    """
    vals = values_by_lang.get(lang) or values_by_lang.get(default_lang) or []
    flags = re.I if case_insensitive_langs and lang in case_insensitive_langs else 0
    for v in vals:
        m = re.search(v, text, flags=flags)
        if m:
            return m.group(0).strip()
    return vals[0] if vals else fallback


def detect_language(text: str) -> str:
    """지원 언어를 스크립트/키워드 기반 휴리스틱으로 판별합니다.

    예시:
        detect_language("Résume les messages") -> "fr"
    """
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
    """정규식 우선순위와 언어 fallback을 이용해 핵심 동사를 선택합니다.

    예시:
        pick_main_verb("Please send the report", "en", {"en": ["send"]}, {"en": ["request"]}) -> "send"
    """
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
