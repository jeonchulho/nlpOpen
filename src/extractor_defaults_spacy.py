from __future__ import annotations

SPACY_REFINE_ACTION_EXCLUDED_TYPES = {"time", "location", "reason", "constraint"}

ACTION_SCOPE_PATTERNS_BY_LANG: dict[str, dict[str, list[str]]] = {
    "ko": {
        "sender": [
            r"([가-힣]{2,3})이가\s+보낸",
            r"((?<!이)[가-힣]{2,4})가\s+보낸",
            r"([가-힣]{2,3})이가\s+[^.!?]{0,40}?\s*보낸",
            r"((?<!이)[가-힣]{2,4})가\s+[^.!?]{0,40}?\s*보낸",
            r"([가-힣]{2,3})이가\s+(?:등록한|올린)",
            r"((?<!이)[가-힣]{2,4})가\s+(?:등록한|올린)",
        ],
        "receiver_names": [r"([가-힣]{2,4})\s*,\s*[가-힣A-Za-z0-9]+(?:팀|부|본부|센터|실|그룹)\s*에게"],
        "receiver_departments": [r"([가-힣A-Za-z0-9]+(?:팀|부|본부|센터|실|그룹))\s*에게"],
    },
    "ja": {
        "sender": [r"(?:の)?([\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFF]{2,12})が(?:送った|送信した)"],
        "receiver_names": [r"([\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFF]{2,12})\s*、\s*[\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFFA-Za-z0-9]+(?:部|チーム|本部|センター)に"],
        "receiver_departments": [r"([\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFFA-Za-z0-9]+(?:部|チーム|本部|センター))に"],
    },
    "zh": {
        "sender": [r"([\u4E00-\u9FFF]{2,8})(?:发送的|发来的|发送)"],
        "receiver_names": [r"给([\u4E00-\u9FFF]{2,8})和[\u4E00-\u9FFFA-Za-z0-9]+(?:部|团队|小组)"],
        "receiver_departments": [r"(?:给|和)([\u4E00-\u9FFFA-Za-z0-9]{1,12}(?:部|团队|小组))"],
    },
    "en": {
        "sender": [r"([A-Z][a-z]+)\s+(?:sent|has sent)"],
        "receiver_names": [r"to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"],
        "receiver_departments": [r"([A-Za-z]+(?:\s+[A-Za-z]+)?\s+(?:team|department|division))"],
    },
}

ACTION_SCOPE_VERB_HINTS = {
    "summarize": ["정리", "요약", "축약", "summar", "整理", "تلخيص", "zusammen", "résum"],
    "send": ["보내", "전달", "send", "sent", "送", "发送", "إرسال", "schick", "senden", "envoy"],
}

ACTION_SCOPE_CLEANUP = {
    "ja_split_delimiter": "の",
    "zh_department_split_delimiter": "和",
    "zh_department_suffix_pattern": r"(?:部|团队|小组)$",
}

OBJECT_ACTION_MODIFIER_PATTERNS_BY_LANG: dict[str, str] = {
    "ko": r"\b([가-힣]+(?:은|는|한|된|던)|보낸|받은|작성한|정리한|확인한|첨부한|등록한|요청한|취소한|변경한|예약한|발송한|도착한)\s+[가-힣]+",
    "en": r"\b(sent|received|written|summarized|summarised|attached|registered|requested|canceled|cancelled|changed|scheduled|shipped|delivered|gesendet|gesandte|zusammengefasste|envoye|envoyee|resume|resumee|annule)\b\s+[A-Za-z]+",
    "ja": r"([\u3040-\u30FF\u4E00-\u9FFF]+(?:した|された|送った|送信した|作成した))",
    "zh": r"([\u4E00-\u9FFF]+(?:的|发送的|整理的| 작성的))",
    "ar": r"(المرسل|المستلم|الملخص|المرفق|المكتوب)",
}

OBJECT_ACTION_MODIFIER_SKIP_TOKENS_BY_LANG: dict[str, list[str]] = {
    "ko": ["한", "된", "던", "는", "은"],
}

OBJECT_ACTION_MODIFIER_ACTION_TOKENS_BY_LANG: dict[str, list[str]] = {
    "ko": [
        "보낸",
        "받은",
        "작성한",
        "정리한",
        "확인한",
        "첨부한",
        "등록한",
        "요청한",
        "취소한",
        "변경한",
        "예약한",
        "발송한",
        "도착한",
    ]
}

OBJECT_ACTION_MODIFIER_ATTRIBUTE_SUFFIXES_BY_LANG: dict[str, list[str]] = {
    "ko": ["한", "은", "는", "된", "던"],
}

OBJECT_DETAIL_RULES_BY_LANG: dict[str, dict[str, object]] = {
    "ko": {
        "command_hint": r"(?:요약(?:해|해서)?|정리(?:해|해서)?|보내(?:줘|주세요)?|전달(?:해|해줘|해주세요)?|발송(?:해|해줘|해주세요)?|공유(?:해|해줘|해주세요)?|확인(?:해|해줘|해주세요)?|알려(?:줘|주세요)?)",
        "patterns": [
            r"([가-힣A-Za-z0-9]{2,12}\s+사용자\s*정보)(?=\s+{command_hint})",
            r"([가-힣A-Za-z0-9]{2,12}(?:가|이)\s+[가-힣]{1,15}(?:한|은|는|던|된)\s+[가-힣A-Za-z0-9]{1,20}(?:\s+[가-힣A-Za-z0-9]{1,20})?)(?=\s+{command_hint})",
            r"([가-힣]{1,15}(?:한|은|는|던|된)\s+[가-힣A-Za-z0-9]{1,20}(?:\s+[가-힣A-Za-z0-9]{1,20})?)(?=\s+{command_hint})",
            r"([가-힣A-Za-z0-9]{2,12}(?:가|이)\s+보낸\s+(?:채팅|쪽지|메시지|메일|이메일))",
            r"((?:보낸|받은|작성한|정리한)\s+(?:채팅|쪽지|메시지|메일|이메일))",
        ],
    },
    "en": {
        "command_hint": r"(?:summari[sz]e|send|forward|dispatch|share|review|check)",
        "patterns": [
            r"((?:message|messages|email|emails|chat(?:\s+messages?)?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:sent|received|written)(?:\s+(?:yesterday|today|tomorrow))?)(?=\s+(?:and\s+)?{command_hint})",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:sent|received|written)\s+(?:message|messages|email|emails|chat(?:\s+messages?)?))(?=\s+(?:and\s+)?{command_hint})",
            r"((?:important|urgent|latest)\s+[A-Za-z][A-Za-z\s]{1,20})(?=\s+{command_hint})",
        ],
    },
}

OBJECT_PHRASE_FALLBACK_BY_LANG: dict[str, dict[str, object]] = {
    "ko": {
        "capture_pattern": r"([가-힣A-Za-z0-9\s,~\-]{1,50}?)(?:을|를)\s+[가-힣A-Za-z0-9]+",
        "strip_prefix_pattern": r"^(?:(?:월|화|수|목|금|토|일)(?:요일)?(?:\s*,\s*(?:월|화|수|목|금|토|일)(?:요일)?)*|[0-9년월일\-\./~\s]+)에\s+",
        "head_suffix_map": {" 일정": "일정"},
    }
}

ADDITIONAL_CONDITION_PATTERNS_BY_LANG: dict[str, list[tuple[str, str]]] = {
    "ko": [
        ("reason", r"([가-힣A-Za-z0-9\s]+(?:때문에|라서))"),
        ("constraint", r"([가-힣A-Za-z0-9\s]+(?:한해서|이내(?:에만)?|이상(?:일 때만)?|이하|상태에서만|경우에만|후에만|때만))"),
        ("constraint", r"([가-힣A-Za-z0-9\s]+(?:이면|하면))"),
        ("location", r"([가-힣A-Za-z0-9\s]+(?:에서만|에서|으로만|으로))"),
    ]
}

SPACY_SIGNAL_PATTERNS_BY_LANG: dict[str, dict[str, str]] = {
    "ko": {
        "person": r"(?:^|\s|,)(?:([가-힣]{2,3})(?=이가\b)|((?<!이)[가-힣]{2,4})(?=가\b)|([가-힣]{2,4})(?=(?:이|님|씨|에게|한테|께|의|\s*,)\b))",
        "comma_name": r"(?:^|\s)([가-힣]{2,4})\s*,\s*[가-힣A-Za-z0-9]+(?:팀|부|본부|센터|실|그룹)",
        "action": r"([가-힣]+(?:해서|하고|해|한|하기|요청|부탁|보내줘|보내 주세요|보내))",
    },
    "ja": {
        "person": r"([\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFF]{2,12})(?=(?:が|さん|様|へ|に|の))",
        "comma_name": r"([\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFF]{2,12})\s*、\s*[\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFFA-Za-z0-9]+(?:部|チーム|本部|センター)",
        "action": r"(整理して|送って|送信|要約|キャンセル)",
    },
    "zh": {
        "person": r"([\u4E00-\u9FFF]{2,8})(?=(?:给|和|的|发送|发给))",
        "comma_name": r"([\u4E00-\u9FFF]{2,8})和[\u4E00-\u9FFFA-Za-z0-9]+(?:部|团队|小组)",
        "action": r"(整理|发送|取消|汇总)",
    },
    "en": {
        "person": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        "comma_name": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,\s*[A-Za-z][A-Za-z\s]+(?:team|department|division)\b",
        "action": r"\b(send|sent|cancel|summarize|summarise|整理|送信|发送|تلخيص|إرسال|stornieren|envoyer|résumer)\b",
    },
}
