from __future__ import annotations

SUPPORTED_LANGUAGES = {"ko", "en", "ja", "zh", "ar", "de", "fr"}
DEPARTMENT_SUFFIXES = r"(?:team|department|division|部|팀|부|本部|センター|グループ|équipe|département|abteilung|vertriebsteam|فريق|قسم)"

KOREAN_PERSON_STOPWORDS = {
    "어제",
    "그제",
    "오늘",
    "내일",
    "모레",
    "이번",
    "다음",
    "오전",
    "오후",
    "금일",
    "익일",
    "당일",
    "까지",
    "부터",
    "이후",
    "이전",
    "중",
    "개월",
    "시간",
    "분",
    "초",
    "날짜",
}

GENERIC_PERSON_STOPWORDS = {
    "yesterday",
    "today",
    "tomorrow",
    "gestern",
    "heute",
    "morgen",
    "hier",
    "demain",
    "aujourd'hui",
    "昨天",
    "今天",
    "明天",
    "昨日",
    "今日",
    "明日",
    "أمس",
    "اليوم",
    "غدا",
    "غدًا",
}

NON_PERSON_TOKENS_BY_LANG: dict[str, list[str]] = {
    "ko": [
        "배송",
        "주문",
        "결제",
        "환불",
        "구독",
        "주소",
        "보고서",
        "첨부",
        "채팅",
        "메시지",
        "이메일",
        "메일",
        "일정",
        "서비스",
        "계정",
        "쿠폰",
        "앱",
    ],
    "en": [
        "delivery",
        "order",
        "payment",
        "refund",
        "subscription",
        "address",
        "report",
        "attachment",
        "chat",
        "message",
        "email",
        "mail",
        "schedule",
        "service",
        "account",
        "coupon",
        "app",
    ],
}

SPACY_ONLY_SUBJECT_BY_LANG: dict[str, str] = {
    "ko": "고객",
    "en": "customer",
    "ja": "customer",
    "zh": "customer",
    "ar": "customer",
    "de": "customer",
    "fr": "customer",
}

SUBJECT_PICKER_RULES_BY_LANG: dict[str, dict[str, object]] = {
    "ko": {
        "topic_pattern": r"([가-힣A-Za-z0-9\s]{1,30}?)(?:은|는|이|가)\s",
        "topic_exclude": ["고객", "상담원"],
        "predicate_verb_keywords": [
            "가능",
            "되나요",
            "됩니다",
            "실패",
            "반복",
            "중단",
            "도착",
            "올라가",
            "오지",
            "오나요",
            "잠기",
        ],
        "object_suffix_subject_overrides": {
            "배송": "배송",
        },
        "object_skip_values": ["요청 대상", "request target", "target"],
    }
}

SPACY_ONLY_CONFIDENCE = 0.72
SPACY_ONLY_MULTI_CONFIDENCE = 0.7
