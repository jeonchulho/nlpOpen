from __future__ import annotations

VERB_PATTERNS_BY_LANG: dict[str, list[str]] = {
    "ko": [
        r"보내\s*주세요",
        r"보내줘",
        r"보내",
        r"취소하겠습니다",
        r"취소해\s*주세요",
        r"정리해서",
        r"알려줘",
        r"알려\s*줘",
        r"알려\s*주세요",
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
        (r"게시판\s*내용", "게시판 내용"),
        (r"게시판", "게시판"),
        (r"(?=.*주문)(?=.*전부)", "주문 전부"),
        (r"주문", "주문"),
    ],
    "en": [
        (r"board\s*content|content\s+on\s+the\s+board|on\s+the\s+board", "board content"),
        (r"message", "summarized message"),
        (r"order", "order"),
    ],
    "de": [
        (r"board\s*inhalt|inhalt\s+im\s+board|im\s+board", "Board-Inhalt"),
        (r"Nachricht", "zusammengefasste Nachricht"),
        (r"E-Mail", "E-Mail"),
        (r"Adresse|Lieferadresse", "Lieferadresse"),
        (r"Zahlung", "Zahlung"),
        (r"Rückerstattung", "Rückerstattung"),
        (r"Bericht", "Bericht"),
        (r"Anhang", "Anhang"),
        (r"Bestellung", "Bestellung"),
    ],
    "fr": [
        (r"contenu\s+du\s+tableau|tableau", "contenu du tableau"),
        (r"message", "message résumé"),
        (r"e-mail|mail", "e-mail"),
        (r"adresse|adresse de livraison", "adresse de livraison"),
        (r"paiement", "paiement"),
        (r"remboursement", "remboursement"),
        (r"abonnement", "abonnement"),
        (r"rapport", "rapport"),
        (r"pièce jointe", "pièce jointe"),
        (r"commande", "commande"),
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

QUESTION_PATTERNS_BY_LANG: dict[str, list[str]] = {
    "ko": [r"뭐지\?*$", r"뭐야\?*$", r"무엇.*(인가요|인가|인지)\?*$"],
    "en": [r"\?$", r"\bwhat\b", r"\bwho\b", r"\bwhen\b", r"\bwhere\b", r"\bwhich\b"],
    "de": [r"\?$", r"\bwas\b", r"\bwer\b", r"\bwann\b", r"\bwo\b"],
    "fr": [r"\?$", r"\bqu[' ]?est-ce\b", r"\bqui\b", r"\bquand\b", r"\bo[uù]\b"],
    "ja": [r"\?$", r"何", r"誰", r"いつ", r"どこ"],
    "zh": [r"[?？]$", r"什么", r"谁", r"何时", r"哪里"],
    "ar": [r"[?؟]$", r"ما", r"من", r"متى", r"أين"],
}

QUERY_VERB_BY_LANG: dict[str, str] = {
    "ko": "조회",
    "en": "query",
    "de": "abfrage",
    "fr": "requete",
    "ja": "照会",
    "zh": "查询",
    "ar": "استعلام",
}

OBJECT_BARE_PHRASE_FALLBACK_BY_LANG: dict[str, dict[str, object]] = {
    "ko": {
        "max_length": 40,
        "allowed_pattern": r"[가-힣A-Za-z0-9\s]+",
        "must_end_with": [r"정보$", r"내용$", r"현황$", r"상태$"],
    },
    "en": {
        "max_length": 64,
        "allowed_pattern": r"[A-Za-z0-9\s'\-]+",
        "must_end_with": [r"\b(info|information|status|summary|content|details?)$"],
    },
    "de": {
        "max_length": 64,
        "allowed_pattern": r"[A-Za-zÄÖÜäöüß0-9\s'\-]+",
        "must_end_with": [r"\b(info|information|status|inhalt|details?)$"],
    },
    "fr": {
        "max_length": 64,
        "allowed_pattern": r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\s'\-]+",
        "must_end_with": [r"\b(info|information|statut|contenu|details?)$"],
    },
}

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
