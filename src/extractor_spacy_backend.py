from __future__ import annotations

import re
from typing import Any


LATIN_LANGS = {"en", "de", "fr"}


def _normalize_findall_matches(matches: list[Any]) -> list[str]:
    """Normalize re.findall outputs so both str and tuple captures are handled safely."""
    out: list[str] = []
    for item in matches:
        if isinstance(item, tuple):
            for part in item:
                token = str(part).strip()
                if token:
                    out.append(token)
            continue
        token = str(item).strip()
        if token:
            out.append(token)
    return out


def _is_latin_noise_token(value: str) -> bool:
    return bool(
        re.search(
            r"\b(and|or|by|to|from|the|a|an|und|oder|zu|von|der|die|das|et|au|aux|de|du|des|la|le|les)\b",
            value,
            flags=re.I,
        )
    )


def _is_latin_department_like(value: str) -> bool:
    return bool(
        re.search(
            r"\b(team|department|division|abteilung|vertriebsteam|service|equipe|équipe|departement|département)\b",
            value,
            flags=re.I,
        )
    )


def _is_latin_temporal_token(value: str) -> bool:
    return bool(
        re.search(
            r"\b(yesterday|today|tomorrow|gestern|heute|morgen|hier|demain|aujourd'hui|aujourdhui)\b",
            value,
            flags=re.I,
        )
    )


def build_spacy_nlp(model_name: str, spacy_module: Any):
    if spacy_module is None:
        return None
    try:
        return spacy_module.load(model_name)
    except Exception:
        # Fallback keeps the postprocessor alive even when model download is missing.
        return spacy_module.blank("xx")


def extract_single(extractor: Any, text: str):
    result = extractor._extract_spacy_only(text)
    result = extractor._apply_spacy_postprocess_single(result, text)
    return extractor._apply_guardrails_single(result)


def extract_multi(extractor: Any, text: str):
    result = extractor._extract_by_verb_spacy_only(text)
    result = extractor._apply_spacy_postprocess_multi(result, text)
    return extractor._apply_guardrails_multi(result)


def build_spacy_single_payload(extractor: Any, text: str) -> dict[str, Any]:
    lang = extractor._detect_language(text)
    verb = extractor._pick_main_verb(text, lang)
    obj = extractor._pick_object(text, lang)
    conditions = extractor._dedupe_conditions(
        extractor._extract_time_conditions(text, lang)
        + extractor._extract_manner_conditions(text, lang)
        + extractor._extract_additional_conditions(text, lang)
        + extractor._spacy_refine_conditions([], text)
    )
    subject = extractor._pick_subject(text, lang, obj, verb)
    return {
        "language": lang,
        "subject": subject,
        "verb": verb,
        "object": obj,
        "conditions": conditions,
        "confidence": extractor.spacy_only_confidence,
    }


def build_spacy_multi_payload(extractor: Any, text: str) -> dict[str, Any]:
    lang = extractor._detect_language(text)
    actions: list[dict[str, Any]] = []
    scoped = extractor._extract_action_scoped_entities(text, "summarize")
    sender_persons = scoped["persons"]
    receiver_scoped = extractor._extract_action_scoped_entities(text, "send")
    receiver_persons = receiver_scoped["persons"]
    receiver_departments = receiver_scoped["departments"]
    time_conds = extractor._extract_time_conditions(text, lang)
    manner_conds = extractor._extract_manner_conditions(text, lang)
    extra_conds = extractor._extract_additional_conditions(text, lang)

    text_lower = text.lower()
    summarize_hints = list(extractor._get_summarize_hint_keywords(lang))
    # Include action-scope summarize hints to absorb inflections like German "zusammen...".
    summarize_hints.extend(extractor.action_scope_verb_hints.get("summarize", []))
    summarize_hint = any(k and k.lower() in text_lower for k in summarize_hints)
    if summarize_hint:
        first_conditions = [*time_conds, *extra_conds]
        for p in sorted(sender_persons):
            first_conditions.append({"type": "person", "text": p})
        summarize_obj = extractor._pick_summarize_object(text, lang)
        for a in extractor._extract_object_action_modifiers(summarize_obj):
            first_conditions.append({"type": "action", "text": a})
        actions.append(
            {
                "subject": extractor._pick_lang_value(extractor.subject_by_lang, lang, "en", "customer"),
                "verb": extractor._pick_lang_value_by_text(
                    extractor.summarize_verb_by_lang,
                    lang,
                    text,
                    "en",
                    "summarize",
                    case_insensitive_langs={"en", "de", "fr"},
                ),
                "object": summarize_obj,
                "conditions": extractor._dedupe_conditions(first_conditions),
            }
        )

    text_folded = text.casefold()
    send_hint_present = any(
        h and str(h).casefold() in text_folded
        for h in extractor.action_scope_verb_hints.get("send", [])
    )
    has_send_targets = bool(receiver_persons or receiver_departments)
    should_add_send_action = (not summarize_hint) or send_hint_present or has_send_targets

    if should_add_send_action:
        send_verb = extractor._pick_main_verb(text, lang)
        send_hints = sorted(
            [h for h in extractor.action_scope_verb_hints.get("send", []) if h],
            key=lambda s: len(str(s)),
            reverse=True,
        )
        if lang == "fr" and "envoi" not in send_hints:
            send_hints.append("envoi")
        best_send_match: tuple[int, str] | None = None
        for hint in send_hints:
            if not hint:
                continue
            if lang in LATIN_LANGS and re.search(r"[A-Za-z]", hint):
                patt = rf"\b{re.escape(str(hint))}\w*\b"
                matches = list(re.finditer(patt, text, flags=re.I))
            else:
                flags = re.I if lang in LATIN_LANGS else 0
                matches = list(re.finditer(re.escape(str(hint)), text, flags=flags))
            if not matches:
                continue
            candidate = matches[-1]
            token = candidate.group(0).strip()
            pos = candidate.start()
            if best_send_match is None or pos > best_send_match[0]:
                best_send_match = (pos, token)
        if best_send_match is not None:
            send_verb = best_send_match[1]

        second_conditions = [*time_conds, *manner_conds, *extra_conds]
        for p in sorted(receiver_persons):
            second_conditions.append({"type": "person", "text": p})
        for d in sorted(receiver_departments):
            second_conditions.append({"type": "department", "text": d})
        actions.append(
            {
                "subject": extractor._pick_lang_value(extractor.subject_by_lang, lang, "en", "customer"),
                "verb": send_verb,
                "object": extractor._pick_object(text, lang),
                "conditions": extractor._dedupe_conditions(second_conditions),
            }
        )

    return {
        "language": lang,
        "actions": actions,
        "confidence": extractor.spacy_only_multi_confidence,
    }


def extract_action_scoped_entities(extractor: Any, text: str, verb_text: str) -> dict[str, set[str]]:
    persons: set[str] = set()
    departments: set[str] = set()
    lang = extractor._detect_language(text)

    lang_scope = extractor.action_scope_patterns_by_lang.get(lang, extractor.action_scope_patterns_by_lang.get("en", {}))

    def _collect(patterns: list[str], use_icase: bool) -> list[str]:
        out: list[str] = []
        for patt in patterns:
            flags = re.I if use_icase else 0
            out.extend(_normalize_findall_matches(re.findall(patt, text, flags=flags)))
        return out

    latin_lang = lang in LATIN_LANGS
    sender = _collect(lang_scope.get("sender", []), latin_lang)
    receiver_names = _collect(lang_scope.get("receiver_names", []), latin_lang)
    receiver_depts = _collect(lang_scope.get("receiver_departments", []), latin_lang)

    if latin_lang:
        sender = [
            s
            for s in sender
            if s
            and not _is_latin_noise_token(s)
            and not _is_latin_department_like(s)
            and not _is_latin_temporal_token(s)
        ]
        normalized_receiver_names: list[str] = []
        for n in receiver_names:
            cand = re.split(r"\b(?:and|und|et)\b", n, maxsplit=1, flags=re.I)[0].strip()
            cand = re.sub(r"^(?:the|der|die|das|le|la|les|de|du|des)\s+", "", cand, flags=re.I).strip()
            if cand:
                normalized_receiver_names.append(cand)
        receiver_names = normalized_receiver_names
        receiver_names = [
            n
            for n in receiver_names
            if n
            and not _is_latin_noise_token(n)
            and not _is_latin_department_like(n)
            and not _is_latin_temporal_token(n)
            and not re.search(r"\b(email|attachment|report|invoice|chat)\b", n, flags=re.I)
        ]
        receiver_depts = [
            re.sub(r"^(?:the|der|die|das|le|la|les|de|du|des)\s+", "", d, flags=re.I).strip()
            for d in receiver_depts
        ]

    if lang == "ja":
        delimiter = extractor.action_scope_cleanup.get("ja_split_delimiter", "の")
        sender = [s.split(delimiter)[-1].strip() for s in sender]
        receiver_names = [n.split(delimiter)[-1].strip() for n in receiver_names]
        receiver_depts = [d.split(delimiter)[-1].strip() for d in receiver_depts]
        receiver_names = [
            n
            for n in receiver_names
            if n and re.fullmatch(r"[\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFF]{2,8}", n)
        ]
    elif lang == "zh":
        cleaned_depts: list[str] = []
        split_delim = extractor.action_scope_cleanup.get("zh_department_split_delimiter", "和")
        suffix_pattern = extractor.action_scope_cleanup.get("zh_department_suffix_pattern", r"(?:部|团队|小组)$")
        for d in receiver_depts:
            cand = d.strip()
            if split_delim and split_delim in cand:
                cand = cand.split(split_delim)[-1].strip()
            if re.search(suffix_pattern, cand):
                cleaned_depts.append(cand)
        receiver_depts = cleaned_depts

    v = (verb_text or "").strip().lower()
    is_summarize = any(k in v for k in extractor.action_scope_verb_hints.get("summarize", []))
    is_send = any(k in v for k in extractor.action_scope_verb_hints.get("send", []))

    if is_summarize:
        persons.update([p for p in sender if p])
    elif is_send:
        persons.update([p for p in receiver_names if p])
        departments.update([d for d in receiver_depts if d])
    else:
        persons.update([p for p in sender if p])
        persons.update([p for p in receiver_names if p])
        departments.update([d for d in receiver_depts if d])

    return {"persons": persons, "departments": departments}


def extract_object_action_modifiers(extractor: Any, object_text: str) -> list[str]:
    if not object_text.strip():
        return []

    lang = extractor._detect_language(object_text)
    pattern = extractor.object_action_modifier_patterns_by_lang.get(lang)
    if pattern is None and lang in {"de", "fr"}:
        pattern = extractor.object_action_modifier_patterns_by_lang.get("en")
    if pattern is None:
        return []

    flags = re.I if lang in {"en", "de", "fr"} else 0
    out: list[str] = []
    skip_tokens = set(extractor.object_action_modifier_skip_tokens_by_lang.get(lang, []))
    action_tokens = set(extractor.object_action_modifier_action_tokens_by_lang.get(lang, []))
    attribute_suffixes = tuple(extractor.object_action_modifier_attribute_suffixes_by_lang.get(lang, []))
    for m in re.finditer(pattern, object_text, flags=flags):
        token = (m.group(1) if m.lastindex else m.group(0)).strip()
        if token in skip_tokens:
            continue
        if action_tokens and token not in action_tokens:
            continue
        if attribute_suffixes and token.endswith(attribute_suffixes) and token not in action_tokens:
            continue
        out.append(token)
    return out


def extract_spacy_signals(
    extractor: Any,
    text: str,
    korean_person_stopwords: set[str],
    generic_person_stopwords: set[str],
    department_suffixes: str,
) -> dict[str, set[str]]:
    persons: set[str] = set()
    departments: set[str] = set()
    actions: set[str] = set()
    lang = extractor._detect_language(text)

    dept_pattern = re.compile(extractor.spacy_signal_dept_pattern, flags=re.I)
    patt = extractor.spacy_signal_patterns_by_lang.get(lang, extractor.spacy_signal_patterns_by_lang.get("en", {}))
    non_person_tokens = {t.casefold() for t in extractor.non_person_tokens_by_lang.get(lang, [])}
    action_icase = lang in LATIN_LANGS
    person_pattern = re.compile(patt.get("person", r"$^"), flags=0)
    comma_name_pattern = re.compile(patt.get("comma_name", r"$^"), flags=0)
    action_pattern = re.compile(patt.get("action", r"$^"), flags=re.I if action_icase else 0)

    for m in dept_pattern.findall(text):
        dep = m.strip()
        if lang == "en":
            dep = re.sub(r"^the\s+", "", dep, flags=re.I).strip()
        departments.add(dep)
    for m in _normalize_findall_matches(person_pattern.findall(text)):
        cand = m.strip()
        if lang == "ja":
            for prefix in extractor.spacy_signal_ja_person_prefixes:
                if cand.startswith(prefix):
                    cand = cand[len(prefix):]
                    break
            if re.search(r"[をはがにでと].{1,}", cand) or re.search(r"(して|した|ます|です)$", cand):
                continue
        if lang in LATIN_LANGS:
            if re.search(r"^(?:the|a|an|by|to|from|and|der|die|das|zu|von|de|du|des|le|la|les|et)\b", cand, flags=re.I):
                continue
            if _is_latin_noise_token(cand):
                continue
            if _is_latin_department_like(cand):
                continue
            if re.search(r"\b(email|attachment|report|invoice|chat)\b", cand, flags=re.I):
                continue
        if cand in korean_person_stopwords:
            continue
        if cand.lower() in generic_person_stopwords:
            continue
        if cand.casefold() in non_person_tokens:
            continue
        if re.search(extractor.spacy_signal_person_time_suffix_pattern, cand):
            continue
        if re.search(extractor.spacy_signal_latin_time_suffix_pattern, cand.lower()):
            continue
        persons.add(cand)
    for m in _normalize_findall_matches(comma_name_pattern.findall(text)):
        cand = m.strip()
        if lang in LATIN_LANGS:
            if _is_latin_noise_token(cand) or _is_latin_department_like(cand):
                continue
        if cand and cand not in korean_person_stopwords:
            if cand.casefold() in non_person_tokens:
                continue
            persons.add(cand)

    if lang == "ja":
        dep_like = {p for p in persons if re.search(extractor.spacy_signal_department_suffix_pattern_ja, p)}
        persons -= dep_like
        departments |= dep_like
    for m in _normalize_findall_matches(action_pattern.findall(text)):
        actions.add(m.strip())

    if extractor.spacy_nlp is not None:
        doc = extractor.spacy_nlp(text)
        for ent in getattr(doc, "ents", []):
            et = (ent.label_ or "").upper()
            etext = ent.text.strip()
            if not etext:
                continue
            if et in {"PERSON", "PER"}:
                if etext not in korean_person_stopwords and etext.lower() not in generic_person_stopwords:
                    persons.add(etext)
            elif et in {"ORG"} and re.search(department_suffixes + r"$", etext, flags=re.I):
                departments.add(etext)

    return {"persons": persons, "departments": departments, "actions": actions}


def spacy_refine_conditions(
    extractor: Any,
    conditions: list[dict],
    text: str,
    object_text: str = "",
    verb_text: str = "",
    add_global_entities: bool = True,
    korean_person_stopwords: set[str] | None = None,
    generic_person_stopwords: set[str] | None = None,
    department_suffixes: str = "",
) -> list[dict]:
    signals = extract_spacy_signals(
        extractor,
        text,
        korean_person_stopwords or set(),
        generic_person_stopwords or set(),
        department_suffixes,
    )
    scoped = extract_action_scoped_entities(extractor, text, verb_text)
    updated: list[dict] = []

    for c in conditions:
        ctype = str(c.get("type", "other")).strip() or "other"
        ctext = str(c.get("text", "")).strip()
        if not ctext:
            continue

        if ctext in signals["departments"]:
            ctype = "department"
        elif ctext in signals["persons"]:
            ctype = "person"
        elif (
            ctext in signals["actions"]
            and ctype not in extractor.spacy_refine_action_excluded_types
            and ctype in {"other", "action"}
        ):
            ctype = "action"

        updated.append({"type": ctype, "text": ctext})

    existing_texts = {str(c.get("text", "")).strip() for c in updated}
    if add_global_entities:
        for p in sorted(signals["persons"]):
            if p not in existing_texts:
                updated.append({"type": "person", "text": p})
        for d in sorted(signals["departments"]):
            if d not in existing_texts:
                updated.append({"type": "department", "text": d})
    else:
        for p in sorted(scoped["persons"]):
            if p not in existing_texts:
                updated.append({"type": "person", "text": p})
        for d in sorted(scoped["departments"]):
            if d not in existing_texts:
                updated.append({"type": "department", "text": d})

    for a in extract_object_action_modifiers(extractor, object_text):
        if a not in existing_texts:
            updated.append({"type": "action", "text": a})

    return extractor._dedupe_conditions(updated)
