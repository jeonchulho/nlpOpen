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
    "what",
    "who",
    "when",
    "where",
    "which",
    "was",
    "qu",
    "que",
    "quoi",
    "qui",
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
        "쪽지",
        "채팅",
        "메시지",
        "이메일",
        "메일",
        "일정",
        "서비스",
        "계정",
        "쿠폰",
        "앱",
        "내용",
        "게시판",
    ],
    "en": [
        "what",
        "who",
        "when",
        "where",
        "which",
        "board",
        "content",
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
    "de": ["was", "wer", "wann", "wo", "board", "tafel", "inhalt"],
    "fr": ["qu", "que", "quoi", "qui", "quand", "ou", "où", "tableau", "contenu"],
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
        "priority_subject_patterns": [
            r"([가-힣]{2,3})이가\s+(?:올린|등록한|보낸)",
            r"((?<!이)[가-힣]{2,4})가\s+(?:올린|등록한|보낸)",
        ],
        "subject_cleanup_regexes": [
            [r"^(?:어제|그제|오늘|내일|모레|이번\s*주|다음\s*주|이번\s*달|다음\s*달)\s+", ""],
            [r"(.{2,}?)(?:은|는|이|가)$", r"\1"],
        ],
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
    },
    "en": {
        "topic_exclude": ["customer", "agent"],
        "priority_subject_patterns": [
            r"(?:what\s+did\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:post(?:ed)?|upload(?:ed)?|register(?:ed)?|send|sent|write|wrote)",
        ],
        "object_skip_values": ["request target", "target"],
    },
    "de": {
        "topic_exclude": ["kunde"],
        "priority_subject_patterns": [
            r"hat\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?).*\b(?:gesendet|gepostet|hochgeladen|registriert)",
            r"([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s+(?:hat\s+)?(?:gesendet|gepostet|hochgeladen|registriert)",
        ],
        "object_skip_values": ["request target", "target"],
    },
    "fr": {
        "topic_exclude": ["client"],
        "priority_subject_patterns": [
            r"qu[' ]?est-ce\s+qu[' ]?([A-ZÉÈÊÀÂÇÎÔÛ][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-ZÉÈÊÀÂÇÎÔÛ][A-Za-zÀ-ÖØ-öø-ÿ'-]+)?)\s+a\s+(?:envoye|envoyé|poste|posté|publie|publié|enregistre|enregistré)",
            r"([A-ZÉÈÊÀÂÇÎÔÛ][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-ZÉÈÊÀÂÇÎÔÛ][A-Za-zÀ-ÖØ-öø-ÿ'-]+)?)\s+(?:a\s+)?(?:envoye|envoyé|poste|posté|publie|publié|enregistre|enregistré)",
        ],
        "object_skip_values": ["request target", "target"],
    },
    "ja": {
        "priority_subject_patterns": [
            r"([\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFF]{2,12})(?=が(?:投稿した|登録した|送った|送信した))",
        ],
        "object_skip_values": ["request target", "target"],
    },
    "zh": {
        "priority_subject_patterns": [
            r"([\u4E00-\u9FFF]{2,8})(?=(?:发布的|上传的|登记的|发送的))",
        ],
        "object_skip_values": ["request target", "target"],
    }
}

SPACY_ONLY_CONFIDENCE = 0.72
SPACY_ONLY_MULTI_CONFIDENCE = 0.7
