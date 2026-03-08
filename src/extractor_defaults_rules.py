from __future__ import annotations

VERB_PATTERNS_BY_LANG: dict[str, list[str]] = {
    "ko": [
        r"보내\s*주세요",
        r"보내줘",
        r"보내",
        r"취소하겠습니다",
        r"취소해\s*주세요",
        r"정리해서",
    ],
    "ja": [r"送って", r"送信", r"キャンセル", r"まとめて"],
    "zh": [r"发送", r"取消", r"整理"],
    "ar": [r"أرسل", r"إرسال", r"إلغاء", r"تلخيص"],
    "de": [r"senden", r"schicken", r"stornieren", r"zusammenfassen"],
    "fr": [r"envoyer", r"annuler", r"résumer"],
    "en": [r"send", r"cancel", r"summari[sz]e"],
}

DEFAULT_VERB_BY_LANG: dict[str, str] = {
    "ko": "요청",
    "ja": "依頼",
    "zh": "请求",
    "ar": "طلب",
    "de": "anfrage",
    "fr": "demande",
    "en": "request",
}

OBJECT_RULES_BY_LANG: dict[str, list[tuple[str, str]]] = {
    "ko": [
        (r"(?=.*쪽지)(?=.*정리)", "쪽지 정리본"),
        (r"쪽지", "쪽지"),
        (r"(?=.*주문)(?=.*전부)", "주문 전부"),
        (r"주문", "주문"),
    ],
    "en": [
        (r"message", "summarized message"),
        (r"order", "order"),
    ],
}

DEFAULT_OBJECT_BY_LANG: dict[str, str] = {
    "ko": "요청 대상",
    "ja": "依頼対象",
    "zh": "请求对象",
    "ar": "هدف الطلب",
    "de": "Anfrageobjekt",
    "fr": "objet de la demande",
    "en": "request target",
}

MANNER_TOKENS_BY_LANG: dict[str, list[str]] = {
    "ko": ["쪽지로", "문자로", "이메일로", "PDF로"],
    "en": ["by email", "via message"],
    "ja": ["メールで"],
    "zh": ["通过消息"],
    "ar": ["بواسطة رسالة"],
    "de": ["per E-Mail"],
    "fr": ["par message"],
}

SUMMARIZE_HINT_KEYWORDS = ["정리", "summar", "整理", "تلخيص", "zusammen", "résum"]

SUMMARIZE_VERB_BY_LANG = {
    "ko": "정리해서",
    "en": "summarize",
    "ja": "summarize",
    "zh": "summarize",
    "ar": "summarize",
    "de": "summarize",
    "fr": "summarize",
}

SUMMARIZE_OBJECT_RULES_BY_LANG: dict[str, list[tuple[str, str]]] = {
    "ko": [(r"(?=.*쪽지)(?=.*보낸)", "쪽지"), (r"쪽지", "쪽지"), (r"메시지", "메시지")],
    "en": [(r"message", "message"), (r"email", "email"), (r"chat", "chat message")],
    "ja": [(r"メッセージ", "メッセージ"), (r"メール", "メール")],
    "zh": [(r"消息", "消息"), (r"邮件", "邮件")],
    "ar": [(r"رسالة", "رسالة"), (r"بريد", "بريد")],
    "de": [(r"Nachricht", "Nachricht"), (r"E-Mail", "E-Mail")],
    "fr": [(r"message", "message"), (r"e-mail|mail", "e-mail")],
}
