from __future__ import annotations

TIME_PATTERNS_COMMON = [
    r"\d{4}[\-/\.]\d{1,2}[\-/\.]\d{1,2}\s*[~\-]\s*\d{4}[\-/\.]\d{1,2}[\-/\.]\d{1,2}",
    r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*[~\-]\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일",
    r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*부터\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*까지",
]

TIME_PATTERNS_BY_LANG = {
    "ko": [r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*이후", r"어제", r"그제", r"오늘", r"내일", r"모레"],
    "en": [r"yesterday", r"today", r"tomorrow", r"from\s+\d{4}[\-/\.]\d{1,2}[\-/\.]\d{1,2}\s+to\s+\d{4}[\-/\.]\d{1,2}[\-/\.]\d{1,2}"],
    "ja": [r"昨日", r"今日", r"明日"],
    "zh": [r"昨天", r"今天", r"明天"],
    "ar": [r"أمس", r"اليوم", r"غدًا|غدا"],
    "de": [r"gestern", r"heute", r"morgen"],
    "fr": [r"hier", r"aujourd'hui", r"demain"],
}

TIME_CASE_INSENSITIVE_LANGS = {"en", "de", "fr"}
