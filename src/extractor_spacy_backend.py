from __future__ import annotations

import re
from typing import Any


LATIN_LANGS = {"en", "de", "fr"}


def _normalize_findall_matches(matches: list[Any]) -> list[str]:
    """`re.findall` 결과를 평탄한 문자열 리스트로 정규화합니다.

    `re.findall`은 패턴에 캡처 그룹이 여러 개 있으면 `tuple` 목록을,
    그룹이 하나면 `str` 목록을 반환합니다. 이 유틸은 두 경우를 모두
    안전하게 문자열 리스트로 통일합니다.

    Args:
        matches: `re.findall` 반환값(문자열 목록 또는 튜플 목록).

    Returns:
        공백 제거가 적용된 문자열 토큰 목록.

    예시:
        _normalize_findall_matches([("Alice", "Sales team"), "Bob"]) -> ["Alice", "Sales team", "Bob"]
    """
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
    """라틴권 언어에서 불용어/연결어 노이즈 토큰인지 판별합니다.

    수신자/발신자 후보를 정제할 때 의미 없는 연결어(`and`, `the`, `de` 등)를
    사람 이름으로 오인하는 문제를 줄이기 위해 사용합니다.

    예시:
        _is_latin_noise_token("and") -> True
    """
    return bool(
        re.search(
            r"\b(and|or|by|to|from|the|a|an|und|oder|zu|von|der|die|das|et|au|aux|de|du|des|la|le|les)\b",
            value,
            flags=re.I,
        )
    )


def _is_latin_department_like(value: str) -> bool:
    """팀/부서명처럼 보이는 토큰인지 판별합니다.

    사람 이름 후보에서 `Sales team` 같은 조직명 토큰을 제외할 때 사용합니다.

    예시:
        _is_latin_department_like("Sales team") -> True
    """
    return bool(
        re.search(
            r"\b(team|department|division|abteilung|vertriebsteam|service|equipe|équipe|departement|département)\b",
            value,
            flags=re.I,
        )
    )


def _is_latin_temporal_token(value: str) -> bool:
    """시간 표현(예: yesterday, heute 등)인지 판별합니다.

    시간 단어가 person으로 잘못 분류되는 오탐을 줄이기 위한 보조 필터입니다.

    예시:
        _is_latin_temporal_token("gestern") -> True
    """
    return bool(
        re.search(
            r"\b(yesterday|today|tomorrow|gestern|heute|morgen|hier|demain|aujourd'hui|aujourdhui)\b",
            value,
            flags=re.I,
        )
    )


def build_spacy_nlp(model_name: str, spacy_module: Any):
    """spaCy 모델을 로드하고 실패 시 `xx` blank 모델로 대체합니다.

    실서비스에서 모델 다운로드 누락/버전 불일치가 있어도 후처리 파이프라인이
    완전히 중단되지 않도록 방어적으로 동작합니다.

    Args:
        model_name: 로드할 spaCy 모델명.
        spacy_module: import된 spaCy 모듈 객체.

    Returns:
        spaCy Language 객체 또는 None.

    예시:
        build_spacy_nlp("xx_ent_wiki_sm", spacy)
    """
    if spacy_module is None:
        return None
    try:
        return spacy_module.load(model_name)
    except Exception:
        # 모델 다운로드가 없어도 후처리 파이프라인은 동작하도록 유지한다.
        return spacy_module.blank("xx")


def extract_single(extractor: Any, text: str):
    """spaCy 단일 추출 후 postprocess/guardrails를 적용합니다.

    처리 순서:
        1) 규칙 기반 단일 추출
        2) spaCy 신호 기반 조건 보정
        3) guardrails(구조/신뢰도 보정)

    예시:
        extract_single(extractor, "어제 보낸 쪽지 요약해줘")
    """
    result = extractor._extract_spacy_only(text)
    result = extractor._apply_spacy_postprocess_single(result, text)
    return extractor._apply_guardrails_single(result)


def extract_multi(extractor: Any, text: str):
    """spaCy split-by-verb 추출 후 postprocess/guardrails를 적용합니다.

    단일 추출과 동일하게 후처리/가드레일을 적용하되,
    액션별 condition 스코프를 유지하도록 multi 경로를 사용합니다.

    예시:
        extract_multi(extractor, "Summarize and send the message to Alice")
    """
    result = extractor._extract_by_verb_spacy_only(text)
    result = extractor._apply_spacy_postprocess_multi(result, text)
    return extractor._apply_guardrails_multi(result)


def build_spacy_single_payload(extractor: Any, text: str) -> dict[str, Any]:
    """spaCy/규칙 시그널을 이용해 단일 액션 payload를 구성합니다.

    language/verb/object/conditions/subject를 규칙 기반으로 채워
    LLM 없이도 스키마에 맞는 결과를 만들기 위한 함수입니다.

    Returns:
        `ExtractionResult`로 변환 가능한 dict payload.

    예시:
        build_spacy_single_payload(extractor, "배송지 주소를 오늘 변경해줘")
    """
    lang = extractor._detect_language(text)
    verb = extractor._pick_main_verb(text, lang)
    obj = extractor._pick_object(text, lang)
    conditions = extractor._dedupe_conditions(
        extractor._extract_time_conditions(text, lang)
        + extractor._extract_manner_conditions(text, lang)
        + extractor._extract_additional_conditions(text, lang)
        + extractor._spacy_refine_conditions([], text)
    )
    scoped = extractor._extract_action_scoped_entities(text, verb)
    for sender in sorted(scoped.get("sender_persons", set())):
        conditions.append({"type": "sender", "text": sender})
    for receiver in sorted(scoped.get("receiver_persons", set())):
        conditions.append({"type": "receiver", "text": receiver})
    for dep in sorted(scoped.get("receiver_departments", set())):
        conditions.append({"type": "receiver_department", "text": dep})
    conditions = extractor._dedupe_conditions(conditions)
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
    """요약/전송 스코프를 분리해 다중 액션 payload를 구성합니다.

    핵심 동작:
        - summarize 힌트가 있으면 1차 액션(요약) 생성
        - send 힌트/수신자 존재 여부로 2차 액션(전송) 생성 여부 결정
        - action별 condition(person/department/time/manner)를 분리 축적

    Returns:
        `MultiActionExtractionResult`로 변환 가능한 dict payload.

    예시:
        build_spacy_multi_payload(extractor, "정리해서 영업팀에 보내줘")
    """
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
    # 독일어 "zusammen..." 같은 어형 변형도 잡히도록 action-scope 힌트를 합친다.
    summarize_hints.extend(extractor.action_scope_verb_hints.get("summarize", []))
    summarize_hint = any(k and k.lower() in text_lower for k in summarize_hints)
    if summarize_hint:
        first_conditions = [*time_conds, *extra_conds]
        for p in sorted(sender_persons):
            first_conditions.append({"type": "person", "text": p})
        summarize_obj = extractor._pick_summarize_object(text, lang)
        summarize_verb = extractor._pick_lang_value_by_text(
            extractor.summarize_verb_by_lang,
            lang,
            text,
            "en",
            "summarize",
            case_insensitive_langs={"en", "de", "fr"},
        )
        for a in extractor._extract_object_action_modifiers(summarize_obj):
            first_conditions.append({"type": "action", "text": a})
        actions.append(
            {
                "subject": extractor._pick_subject(text, lang, summarize_obj, summarize_verb),
                "verb": summarize_verb,
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
        send_obj = extractor._pick_object(text, lang)
        for p in sorted(receiver_persons):
            second_conditions.append({"type": "person", "text": p})
        for d in sorted(receiver_departments):
            second_conditions.append({"type": "department", "text": d})
        actions.append(
            {
                "subject": extractor._pick_subject(text, lang, send_obj, send_verb),
                "verb": send_verb,
                "object": send_obj,
                "conditions": extractor._dedupe_conditions(second_conditions),
            }
        )

    return {
        "language": lang,
        "actions": actions,
        "confidence": extractor.spacy_only_multi_confidence,
    }


def extract_action_scoped_entities(extractor: Any, text: str, verb_text: str) -> dict[str, set[str]]:
    """행동 의도(요약/전송)에 맞춰 발신자/수신자 엔티티를 추출합니다.

    언어별 action-scope regex를 적용한 뒤,
    라틴권 노이즈/시간어/부서형 표현을 정제하여 person/department 집합을 반환합니다.

    Args:
        extractor: 규칙/설정을 보유한 추출기 인스턴스.
        text: 원문 요청 텍스트.
        verb_text: 현재 액션 문맥(예: summarize, send).

        Returns:
                {
                    "persons": set[str],
                    "departments": set[str],
                    "sender_persons": set[str],
                    "receiver_persons": set[str],
                    "receiver_departments": set[str],
                } 형태의 스코프 엔티티.

    예시:
        extract_action_scoped_entities(extractor, "... send to Alice and Sales team", "send")
    """
    persons: set[str] = set()
    departments: set[str] = set()
    lang = extractor._detect_language(text)

    lang_scope = extractor.action_scope_patterns_by_lang.get(lang, extractor.action_scope_patterns_by_lang.get("en", {}))

    def _collect(patterns: list[str], use_icase: bool) -> list[str]:
        """정규식 매칭 결과를 수집하고 tuple/string 결과를 정규화합니다.

        내부 유틸로, 패턴 배열을 순회하며 모든 매칭 토큰을 한 리스트로 합칩니다.

        예시:
            _collect([r"to\\s+([A-Z][a-z]+)"], True)
        """
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
    elif lang == "ko":
        # "이선정,홍길동에게"처럼 쉼표로 나열된 다중 수신자를 분해한다.
        for m in re.finditer(r"([가-힣]{2,4}(?:\s*,\s*[가-힣]{2,4})+)\s*에게", text):
            chunk = m.group(1).strip()
            for token in re.split(r"\s*,\s*", chunk):
                cand = token.strip()
                if cand:
                    receiver_names.append(cand)

    v = (verb_text or "").strip().lower()
    is_summarize = any(k in v for k in extractor.action_scope_verb_hints.get("summarize", []))
    is_send = any(k in v for k in extractor.action_scope_verb_hints.get("send", []))

    sender_persons = {p for p in sender if p}
    receiver_persons = {p for p in receiver_names if p}
    receiver_departments = {d for d in receiver_depts if d}

    if is_summarize:
        persons.update([p for p in sender if p])
    elif is_send:
        persons.update([p for p in receiver_names if p])
        departments.update([d for d in receiver_depts if d])
    else:
        persons.update([p for p in sender if p])
        persons.update([p for p in receiver_names if p])
        departments.update([d for d in receiver_depts if d])

    return {
        "persons": persons,
        "departments": departments,
        "sender_persons": sender_persons,
        "receiver_persons": receiver_persons,
        "receiver_departments": receiver_departments,
    }


def extract_object_action_modifiers(extractor: Any, object_text: str) -> list[str]:
    """object 구절 안에 들어있는 action 성격 수식어를 추출합니다.

    예: "보낸 쪽지"에서 "보낸"을 action 조건으로 분리.
    언어별 skip/action 토큰 정책을 함께 적용해 노이즈를 줄입니다.

    Returns:
        action 수식어 토큰 문자열 리스트.

    예시:
        extract_object_action_modifiers(extractor, "보낸 쪽지") -> ["보낸"]
    """
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
    """regex + spaCy NER 시그널로 person/department/action을 추출합니다.

    언어별 패턴 매칭 결과와 spaCy 엔터티(`PERSON`, `ORG`)를 결합하고,
    오탐 필터(불용어, 시간어, 비인명 토큰)를 적용해 최종 신호 집합을 만듭니다.

    Returns:
        {"persons": set[str], "departments": set[str], "actions": set[str]}.

    예시:
        extract_spacy_signals(extractor, "Send to Alice and Sales team", set(), set(), r"(team)$")
    """
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
    """전역/스코프 시그널로 condition 타입을 보정하고 중복을 제거합니다.

    보정 규칙:
        - 시그널에 있으면 person/department/action 타입으로 재분류
        - add_global_entities 플래그에 따라 전역/스코프 엔티티 추가
        - object 내부 action 수식어를 조건에 보강

    예시:
        spacy_refine_conditions(extractor, [{"type": "other", "text": "Alice"}], text)
    """
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

        if ctype in {"sender", "receiver", "receiver_department"}:
            updated.append({"type": ctype, "text": ctext})
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

    lang = extractor._detect_language(text)
    if lang == "ko":
        person_texts = {
            str(c.get("text", "")).strip()
            for c in updated
            if str(c.get("type", "")).strip() == "person" and str(c.get("text", "")).strip()
        }
        role_texts = {
            str(c.get("text", "")).strip()
            for c in updated
            if str(c.get("type", "")).strip() in {"sender", "receiver", "receiver_department"}
            and str(c.get("text", "")).strip()
        }
        normalized: list[dict] = []
        for c in updated:
            ctype = str(c.get("type", "")).strip()
            ctext = str(c.get("text", "")).strip()
            if ctype == "person" and ctext in role_texts:
                continue
            if ctype == "person" and ctext.endswith("이") and len(ctext) >= 3 and ctext[:-1] in person_texts:
                continue
            normalized.append(c)
        updated = normalized

    return extractor._dedupe_conditions(updated)
