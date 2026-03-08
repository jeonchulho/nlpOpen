from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

try:
    from extractor_llm_backend import build_llm as llm_build_backend, extract_multi as llm_extract_multi_backend, extract_single as llm_extract_single_backend
    from extractor_spacy_backend import build_spacy_multi_payload, build_spacy_nlp as spacy_build_backend, build_spacy_single_payload, extract_action_scoped_entities as spacy_extract_action_scoped_entities_backend, extract_multi as spacy_extract_multi_backend, extract_object_action_modifiers as spacy_extract_object_action_modifiers_backend, extract_single as spacy_extract_single_backend, extract_spacy_signals as spacy_extract_signals_backend, spacy_refine_conditions as spacy_refine_conditions_backend
    from extractor_common import detect_language as common_detect_language, pick_lang_value as common_pick_lang_value, pick_lang_value_by_text as common_pick_lang_value_by_text, pick_main_verb as common_pick_main_verb
    from extractor_defaults import ACTION_SCOPE_CLEANUP, ACTION_SCOPE_PATTERNS_BY_LANG, ACTION_SCOPE_VERB_HINTS, ADDITIONAL_CONDITION_PATTERNS_BY_LANG, DEFAULT_OBJECT_BY_LANG, DEFAULT_VERB_BY_LANG, DEPARTMENT_SUFFIXES, GENERIC_PERSON_STOPWORDS, KOREAN_PERSON_STOPWORDS, MANNER_TOKENS_BY_LANG, NON_PERSON_TOKENS_BY_LANG, OBJECT_ACTION_MODIFIER_ACTION_TOKENS_BY_LANG, OBJECT_ACTION_MODIFIER_ATTRIBUTE_SUFFIXES_BY_LANG, OBJECT_ACTION_MODIFIER_PATTERNS_BY_LANG, OBJECT_ACTION_MODIFIER_SKIP_TOKENS_BY_LANG, OBJECT_BARE_PHRASE_FALLBACK_BY_LANG, OBJECT_DETAIL_RULES_BY_LANG, OBJECT_PHRASE_FALLBACK_BY_LANG, OBJECT_RULES_BY_LANG, QUERY_VERB_BY_LANG, QUESTION_PATTERNS_BY_LANG, SPACY_ONLY_CONFIDENCE, SPACY_ONLY_MULTI_CONFIDENCE, SPACY_ONLY_SUBJECT_BY_LANG, SPACY_REFINE_ACTION_EXCLUDED_TYPES, SPACY_SIGNAL_DEPARTMENT_SUFFIX_PATTERN_JA, SPACY_SIGNAL_DEPT_PATTERN, SPACY_SIGNAL_JA_PERSON_PREFIXES, SPACY_SIGNAL_LATIN_TIME_SUFFIX_PATTERN, SPACY_SIGNAL_PATTERNS_BY_LANG, SPACY_SIGNAL_PERSON_TIME_SUFFIX_PATTERN, SUBJECT_PICKER_RULES_BY_LANG, SUMMARIZE_HINT_KEYWORDS, SUMMARIZE_OBJECT_RULES_BY_LANG, SUMMARIZE_VERB_BY_LANG, SUPPORTED_LANGUAGES, TIME_CASE_INSENSITIVE_LANGS, TIME_PATTERNS_BY_LANG, TIME_PATTERNS_COMMON, VERB_PATTERNS_BY_LANG
except ImportError:  # pragma: no cover
    from src.extractor_llm_backend import build_llm as llm_build_backend, extract_multi as llm_extract_multi_backend, extract_single as llm_extract_single_backend
    from src.extractor_spacy_backend import build_spacy_multi_payload, build_spacy_nlp as spacy_build_backend, build_spacy_single_payload, extract_action_scoped_entities as spacy_extract_action_scoped_entities_backend, extract_multi as spacy_extract_multi_backend, extract_object_action_modifiers as spacy_extract_object_action_modifiers_backend, extract_single as spacy_extract_single_backend, extract_spacy_signals as spacy_extract_signals_backend, spacy_refine_conditions as spacy_refine_conditions_backend
    from src.extractor_common import detect_language as common_detect_language, pick_lang_value as common_pick_lang_value, pick_lang_value_by_text as common_pick_lang_value_by_text, pick_main_verb as common_pick_main_verb
    from src.extractor_defaults import ACTION_SCOPE_CLEANUP, ACTION_SCOPE_PATTERNS_BY_LANG, ACTION_SCOPE_VERB_HINTS, ADDITIONAL_CONDITION_PATTERNS_BY_LANG, DEFAULT_OBJECT_BY_LANG, DEFAULT_VERB_BY_LANG, DEPARTMENT_SUFFIXES, GENERIC_PERSON_STOPWORDS, KOREAN_PERSON_STOPWORDS, MANNER_TOKENS_BY_LANG, NON_PERSON_TOKENS_BY_LANG, OBJECT_ACTION_MODIFIER_ACTION_TOKENS_BY_LANG, OBJECT_ACTION_MODIFIER_ATTRIBUTE_SUFFIXES_BY_LANG, OBJECT_ACTION_MODIFIER_PATTERNS_BY_LANG, OBJECT_ACTION_MODIFIER_SKIP_TOKENS_BY_LANG, OBJECT_BARE_PHRASE_FALLBACK_BY_LANG, OBJECT_DETAIL_RULES_BY_LANG, OBJECT_PHRASE_FALLBACK_BY_LANG, OBJECT_RULES_BY_LANG, QUERY_VERB_BY_LANG, QUESTION_PATTERNS_BY_LANG, SPACY_ONLY_CONFIDENCE, SPACY_ONLY_MULTI_CONFIDENCE, SPACY_ONLY_SUBJECT_BY_LANG, SPACY_REFINE_ACTION_EXCLUDED_TYPES, SPACY_SIGNAL_DEPARTMENT_SUFFIX_PATTERN_JA, SPACY_SIGNAL_DEPT_PATTERN, SPACY_SIGNAL_JA_PERSON_PREFIXES, SPACY_SIGNAL_LATIN_TIME_SUFFIX_PATTERN, SPACY_SIGNAL_PATTERNS_BY_LANG, SPACY_SIGNAL_PERSON_TIME_SUFFIX_PATTERN, SUBJECT_PICKER_RULES_BY_LANG, SUMMARIZE_HINT_KEYWORDS, SUMMARIZE_OBJECT_RULES_BY_LANG, SUMMARIZE_VERB_BY_LANG, SUPPORTED_LANGUAGES, TIME_CASE_INSENSITIVE_LANGS, TIME_PATTERNS_BY_LANG, TIME_PATTERNS_COMMON, VERB_PATTERNS_BY_LANG

try:
    import instructor
except ImportError:  # pragma: no cover
    instructor = None

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    ChatOllama = None

try:
    from langchain_community.chat_models import ChatLiteLLM
except ImportError:  # pragma: no cover
    ChatLiteLLM = None

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None


def _as_float(value: object, default: float) -> float:
    """값을 float으로 안전하게 변환하고 실패 시 기본값을 반환합니다.

    예시:
        _as_float("0.7", 0.5) -> 0.7
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    """문자열/숫자 형태의 bool 유사 값을 안전하게 변환합니다.

    예시:
        _as_bool("yes", False) -> True
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_str_list(value: object) -> list[str]:
    """스칼라/리스트 입력을 비어 있지 않은 문자열 리스트로 정규화합니다.

    예시:
        _as_str_list([" send ", "forward"]) -> ["send", "forward"]
    """
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return []

DEFAULT_RULES_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "extraction_rules.json"


def _load_extraction_rules_config(path: str) -> dict:
    """JSON 규칙 파일을 로드하고 최상위 스키마를 검증합니다.

    예시:
        _load_extraction_rules_config("config/extraction_rules.json")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rules config file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("rules config root must be a JSON object")
    return data


def _normalize_rule_tables(config: dict | None) -> tuple[
    dict[str, list[str]],
    dict[str, str],
    dict[str, list[tuple[str, str]]],
    dict[str, str],
    dict[str, list[str]],
]:
    """내장 기본 규칙과 동적 config override를 병합합니다.

    Args:
        config: 규칙 JSON dict. None이면 내장 기본값만 사용합니다.

    Returns:
        (verb_patterns, default_verbs, object_rules, default_objects, manner_tokens)
        형태의 병합된 규칙 테이블 튜플.

    Raises:
        없음(구조가 맞지 않는 항목은 무시하고 기본값을 유지).

    예시:
        _normalize_rule_tables({"default_verb_by_lang": {"en": ["ask"]}})
    """
    verb_patterns = {k: list(v) for k, v in VERB_PATTERNS_BY_LANG.items()}
    default_verbs = dict(DEFAULT_VERB_BY_LANG)
    object_rules = {k: list(v) for k, v in OBJECT_RULES_BY_LANG.items()}
    default_objects = dict(DEFAULT_OBJECT_BY_LANG)
    manner_tokens = {k: list(v) for k, v in MANNER_TOKENS_BY_LANG.items()}

    if not config:
        return verb_patterns, default_verbs, object_rules, default_objects, manner_tokens

    vp = config.get("verb_patterns_by_lang", {})
    if isinstance(vp, dict):
        for lang, patterns in vp.items():
            if isinstance(patterns, list):
                verb_patterns[str(lang)] = [str(p) for p in patterns]

    dv = config.get("default_verb_by_lang", {})
    if isinstance(dv, dict):
        for lang, value in dv.items():
            vals = _as_str_list(value)
            if vals:
                default_verbs[str(lang)] = vals[0]

    orules = config.get("object_rules_by_lang", {})
    if isinstance(orules, dict):
        for lang, pairs in orules.items():
            if isinstance(pairs, list):
                normalized_pairs: list[tuple[str, str]] = []
                for item in pairs:
                    if (
                        isinstance(item, list)
                        and len(item) == 2
                        and isinstance(item[0], str)
                        and isinstance(item[1], str)
                    ):
                        normalized_pairs.append((item[0], item[1]))
                object_rules[str(lang)] = normalized_pairs

    dobj = config.get("default_object_by_lang", {})
    if isinstance(dobj, dict):
        for lang, value in dobj.items():
            vals = _as_str_list(value)
            if vals:
                default_objects[str(lang)] = vals[0]

    mt = config.get("manner_tokens_by_lang", {})
    if isinstance(mt, dict):
        for lang, tokens in mt.items():
            if isinstance(tokens, list):
                manner_tokens[str(lang)] = [str(t) for t in tokens]

    return verb_patterns, default_verbs, object_rules, default_objects, manner_tokens


class Condition(BaseModel):
    type: Literal[
        "time",
        "location",
        "manner",
        "reason",
        "constraint",
        "action",
        "person",
        "sender",
        "receiver",
        "receiver_department",
        "department",
        "other",
    ] = Field(
        ..., description="조건 유형"
    )
    text: str = Field(..., description="원문에서 추출한 조건 표현")


class ExtractionResult(BaseModel):
    language: str = Field(
        ...,
        description="입력 문장의 언어. ko/en/ja/zh/ar/de/fr 중 하나를 우선 사용",
    )
    subject: str = Field(..., description="행위 주체")
    verb: str = Field(..., description="핵심 동사")
    object: str = Field(..., description="행위 대상")
    conditions: list[Condition] = Field(default_factory=list, description="조건 정보 목록")
    confidence: float = Field(..., ge=0.0, le=1.0)


class VerbAction(BaseModel):
    subject: str = Field(..., description="해당 동작의 주어")
    verb: str = Field(..., description="해당 동작의 핵심 동사")
    object: str = Field(..., description="해당 동작의 목적어")
    conditions: list[Condition] = Field(default_factory=list, description="해당 동작에만 연결되는 조건")


class MultiActionExtractionResult(BaseModel):
    language: str = Field(..., description="입력 문장의 언어")
    actions: list[VerbAction] = Field(default_factory=list, description="동사별 분리 결과")
    confidence: float = Field(..., ge=0.0, le=1.0)


EXTRACTION_SYSTEM_PROMPT = """
너는 다국어 고객문의 문장에서 구조화 정보를 매우 정확하게 추출하는 분석기다.

목표:
- 주어(subject), 동사(verb), 목적어(object), 조건정보(conditions) 추출

지원 언어:
- 한국어(ko), 영어(en), 일본어(ja), 중국어(zh), 아랍어(ar), 독일어(de), 프랑스어(fr)

엄격한 규칙:
1) 반드시 원문 의미를 보존한다.
2) subject/verb/object는 짧고 명확한 구문으로 추출한다.
3) 조건정보는 시간/장소/방법/이유/제약/행위/사람/부서를 최대한 빠짐없이 분리한다.
4) 발신/수신 관계가 명확하면 person 대신 역할 타입을 우선 사용한다.
    - 보낸 주체: sender
    - 받는 사람: receiver
    - 받는 부서/팀: receiver_department
5) 사람명, 조직명(팀/부/본부 등), 보조 행위(정리/검토/승인 등)는 조건에 각각 person/department/action으로 분류한다.
    단, sender/receiver/receiver_department가 성립하면 해당 역할 타입을 우선한다.
5) 근거가 부족하면 confidence를 낮춘다.
6) 추측 금지. 원문에 없는 정보는 넣지 않는다.
7) 출력은 반드시 스키마에 맞는 JSON 구조만 반환한다.
""".strip()


EXTRACTION_HUMAN_PROMPT = """
아래 고객문의에서 정보를 추출해라.

[다국어 예시]
1) ko: "내일 오전까지 주문을 취소하고 싶어요."
    -> subject="고객", verb="취소하고 싶어요", object="주문", conditions=[(type=time, text=내일 오전까지)]
2) en: "Please refund my payment by Friday."
    -> subject="customer", verb="refund", object="payment", conditions=[(type=time, text=by Friday)]
3) ja: "明日の午後までに配送先を変更したいです。"
    -> subject="顧客", verb="変更したいです", object="配送先", conditions=[(type=time, text=明日の午後までに)]
4) zh: "我想在今天晚上之前取消订单。"
    -> subject="客户", verb="取消", object="订单", conditions=[(type=time, text=今天晚上之前)]
5) ar: "أريد تغيير عنوان التسليم قبل الغد."
    -> subject="العميل", verb="تغيير", object="عنوان التسليم", conditions=[(type=time, text=قبل الغد)]
6) de: "Ich möchte die Lieferung nur an die Firmenadresse senden lassen."
    -> subject="Kunde", verb="senden lassen", object="Lieferung", conditions=[(type=constraint, text=nur an die Firmenadresse)]
7) fr: "Merci d'annuler l'abonnement à partir du mois prochain."
    -> subject="client", verb="annuler", object="abonnement", conditions=[(type=time, text=à partir du mois prochain)]
8) ko: "전철호가 보낸 쪽지 정리해서 이선정, 영업팀에게 보내줘"
    -> subject="고객", verb="보내줘", object="쪽지 정리본", conditions=[(type=sender, text=전철호), (type=receiver, text=이선정), (type=receiver_department, text=영업팀), (type=action, text=정리해서)]
9) en: "What message did John send to Alice and Bob today?"
    -> subject="customer", verb="query", object="message", conditions=[(type=time, text=today), (type=sender, text=John), (type=receiver, text=Alice), (type=receiver, text=Bob)]

[입력 문장]
{text}
""".strip()


REFINE_SYSTEM_PROMPT = """
너는 정보추출 결과를 교정하는 검수기다.

규칙:
1) 원문과 1차 추출 결과를 비교해 누락/오추출을 교정한다.
2) subject/verb/object의 역할이 틀리면 바로잡는다.
3) 조건정보를 가능한 한 세분화해서 보완한다.
    발신/수신 관계가 있으면 person/department보다 sender/receiver/receiver_department를 우선한다.
4) 과도한 확신(confidence)은 낮춘다.
5) 출력은 반드시 동일 스키마 JSON만 반환한다.
""".strip()


REFINE_HUMAN_PROMPT = """
[원문]
{text}

[1차 추출 결과]
{first_pass_json}
""".strip()


MULTI_ACTION_SYSTEM_PROMPT = """
너는 다국어 고객문의 문장을 동사(행위) 단위로 분리하는 정보추출기다.

목표:
- 한 문장에 동사가 여러 개면 actions 배열에 각각 분리해서 넣는다.
- 각 action에 subject/verb/object/conditions를 독립적으로 추출한다.

규칙:
1) 동작 순서는 원문 등장 순서를 따른다.
2) 공통 조건은 각 동작에 필요한 범위에서 중복 포함할 수 있다.
3) 접속 구조("~해서", "그리고", "후에", "then")를 근거로 동작을 분리한다.
4) 발신/수신 관계가 명확하면 person/department보다 sender/receiver/receiver_department를 우선한다.
5) 원문에 없는 동작을 생성하지 않는다.
6) 출력은 반드시 지정된 JSON 스키마를 따른다.
""".strip()


MULTI_ACTION_HUMAN_PROMPT = """
[입력 문장]
{text}
""".strip()


class MultilingualSVOExtractor:
    """다국어 요청에서 SVO + 조건(condition)을 추출하는 핵심 추출기입니다.

    지원 모드:
        - LLM 모드(openai/ollama/litellm): 1차 추출 + 정제(refine) 2단계
        - spaCy 모드(spacy): 규칙 기반 추출 + spaCy 신호 보정

    보조 기능:
        - guardrails: 출력 구조/신뢰도 보정
        - spacy postprocess: person/department/action 타입 재분류
        - split-by-verb: 복합 요청을 다중 액션으로 분해
    """

    def __init__(
        self,
        provider: Literal["openai", "ollama", "litellm", "spacy"] = "openai",
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        extra_rules: str = "",
        ollama_base_url: str = "http://localhost:11434",
        use_instructor: bool = False,
        use_guardrails: bool = False,
        use_spacy_postprocess: bool | None = None,
        spacy_model: str = "xx_ent_wiki_sm",
        litellm_api_base: str = "",
        litellm_api_key: str = "",
        rules_config_file: str = "",
    ) -> None:
        """추출기 실행 환경과 동적 규칙 테이블을 초기화합니다.

        Args:
            provider: 추출 provider(`openai`, `ollama`, `litellm`, `spacy`).
            model: LLM 모델명(LLM provider에서 사용).
            temperature: LLM 온도.
            extra_rules: 시스템 프롬프트에 주입할 추가 규칙 텍스트.
            ollama_base_url: Ollama 서버 URL.
            use_instructor: OpenAI + Instructor 강제 스키마 모드 사용 여부.
            use_guardrails: 결과 보정 로직 활성화 여부.
            use_spacy_postprocess: spaCy 후처리 보정 활성화 여부.
            spacy_model: spaCy 모델명.
            litellm_api_base: LiteLLM API base URL.
            litellm_api_key: LiteLLM API key.
            rules_config_file: 동적 규칙 JSON 경로(미지정 시 기본값 사용).

        Raises:
            ValueError: provider/use_instructor 조합이 유효하지 않을 때.
            ImportError: 필요한 의존성이 설치되지 않았을 때.
            EnvironmentError: API 키가 필요하지만 설정되지 않았을 때.
            FileNotFoundError: 규칙 파일 경로가 잘못되었을 때.
        """
        load_dotenv()
        self.provider = provider
        self.model = model
        self.use_instructor = use_instructor
        self.use_guardrails = use_guardrails
        if use_spacy_postprocess is None:
            self.use_spacy_postprocess = provider == "spacy"
        else:
            self.use_spacy_postprocess = bool(use_spacy_postprocess)
        self.spacy_nlp = self._build_spacy_nlp(spacy_model) if self.use_spacy_postprocess else None

        resolved_rules_path = rules_config_file or str(DEFAULT_RULES_CONFIG_PATH)
        rules_config = _load_extraction_rules_config(resolved_rules_path) if resolved_rules_path else None
        (
            self.verb_patterns_by_lang,
            self.default_verb_by_lang,
            self.object_rules_by_lang,
            self.default_object_by_lang,
            self.manner_tokens_by_lang,
        ) = _normalize_rule_tables(rules_config)
        self.rules_config = rules_config or {}

        self.default_verb_candidates_by_lang: dict[str, list[str]] = {
            str(k): [str(v)] for k, v in DEFAULT_VERB_BY_LANG.items()
        }
        cfg_default_verbs = self.rules_config.get("default_verb_by_lang", {})
        if isinstance(cfg_default_verbs, dict):
            for lang, value in cfg_default_verbs.items():
                vals = _as_str_list(value)
                if vals:
                    self.default_verb_candidates_by_lang[str(lang)] = vals

        self.question_patterns_by_lang: dict[str, list[str]] = {
            str(k): [str(x) for x in v] for k, v in QUESTION_PATTERNS_BY_LANG.items()
        }
        cfg_q_patterns = self.rules_config.get("question_patterns_by_lang", {})
        if isinstance(cfg_q_patterns, dict):
            for lang, patterns in cfg_q_patterns.items():
                vals = _as_str_list(patterns)
                if vals:
                    self.question_patterns_by_lang[str(lang)] = vals

        self.query_verb_by_lang: dict[str, list[str]] = {
            str(k): [str(v)] for k, v in QUERY_VERB_BY_LANG.items()
        }
        cfg_query_verbs = self.rules_config.get("query_verb_by_lang", {})
        if isinstance(cfg_query_verbs, dict):
            for lang, value in cfg_query_verbs.items():
                vals = _as_str_list(value)
                if vals:
                    self.query_verb_by_lang[str(lang)] = vals

        self.default_object_candidates_by_lang: dict[str, list[str]] = {
            str(k): [str(v)] for k, v in DEFAULT_OBJECT_BY_LANG.items()
        }
        cfg_default_objects = self.rules_config.get("default_object_by_lang", {})
        if isinstance(cfg_default_objects, dict):
            for lang, value in cfg_default_objects.items():
                vals = _as_str_list(value)
                if vals:
                    self.default_object_candidates_by_lang[str(lang)] = vals

        self.subject_by_lang: dict[str, list[str]] = {
            str(k): [str(v)] for k, v in SPACY_ONLY_SUBJECT_BY_LANG.items()
        }
        cfg_subjects = self.rules_config.get("subject_by_lang", {})
        if isinstance(cfg_subjects, dict):
            for k, v in cfg_subjects.items():
                vals = _as_str_list(v)
                if vals:
                    self.subject_by_lang[str(k)] = vals

        self.subject_picker_rules_by_lang: dict[str, dict[str, object]] = {
            str(k): dict(v) for k, v in SUBJECT_PICKER_RULES_BY_LANG.items()
        }
        cfg_subject_picker = self.rules_config.get("subject_picker_rules_by_lang", {})
        if isinstance(cfg_subject_picker, dict):
            for lang, rule_obj in cfg_subject_picker.items():
                if not isinstance(rule_obj, dict):
                    continue
                base = dict(self.subject_picker_rules_by_lang.get(str(lang), {}))
                base.update(rule_obj)
                self.subject_picker_rules_by_lang[str(lang)] = base
        self.spacy_only_confidence = _as_float(
            self.rules_config.get("spacy_only_confidence", SPACY_ONLY_CONFIDENCE), SPACY_ONLY_CONFIDENCE
        )
        self.spacy_only_multi_confidence = _as_float(
            self.rules_config.get("spacy_only_multi_confidence", SPACY_ONLY_MULTI_CONFIDENCE),
            SPACY_ONLY_MULTI_CONFIDENCE,
        )

        self.summarize_hint_keywords_by_lang: dict[str, list[str]] = {"default": list(SUMMARIZE_HINT_KEYWORDS)}
        raw_hint_keywords = self.rules_config.get("summarize_hint_keywords", SUMMARIZE_HINT_KEYWORDS)
        if isinstance(raw_hint_keywords, dict):
            parsed: dict[str, list[str]] = {}
            for key, value in raw_hint_keywords.items():
                vals = _as_str_list(value)
                if vals:
                    parsed[str(key)] = vals
            if parsed:
                self.summarize_hint_keywords_by_lang = parsed
        else:
            vals = _as_str_list(raw_hint_keywords)
            if vals:
                self.summarize_hint_keywords_by_lang = {"default": vals}
        self.summarize_verb_by_lang: dict[str, list[str]] = {
            str(k): [str(v)] for k, v in SUMMARIZE_VERB_BY_LANG.items()
        }
        cfg_sum_verbs = self.rules_config.get("summarize_verb_by_lang", {})
        if isinstance(cfg_sum_verbs, dict):
            for k, v in cfg_sum_verbs.items():
                vals = _as_str_list(v)
                if vals:
                    self.summarize_verb_by_lang[str(k)] = vals
        self.summarize_object_rules_by_lang = {k: list(v) for k, v in SUMMARIZE_OBJECT_RULES_BY_LANG.items()}
        cfg_obj_rules = self.rules_config.get("summarize_object_rules_by_lang", {})
        if isinstance(cfg_obj_rules, dict):
            for lang, pairs in cfg_obj_rules.items():
                if isinstance(pairs, list):
                    normalized_pairs: list[tuple[str, str]] = []
                    for item in pairs:
                        if isinstance(item, list) and len(item) == 2:
                            normalized_pairs.append((str(item[0]), str(item[1])))
                    self.summarize_object_rules_by_lang[str(lang)] = normalized_pairs

        self.time_patterns_common = list(self.rules_config.get("time_patterns_common", TIME_PATTERNS_COMMON))
        self.time_patterns_by_lang = {k: list(v) for k, v in TIME_PATTERNS_BY_LANG.items()}
        cfg_time_by_lang = self.rules_config.get("time_patterns_by_lang", {})
        if isinstance(cfg_time_by_lang, dict):
            for lang, patterns in cfg_time_by_lang.items():
                if isinstance(patterns, list):
                    self.time_patterns_by_lang[str(lang)] = [str(p) for p in patterns]
        self.time_case_insensitive_langs = set(
            str(x) for x in self.rules_config.get("time_case_insensitive_langs", list(TIME_CASE_INSENSITIVE_LANGS))
        )

        self.spacy_refine_action_excluded_types = set(
            str(x) for x in self.rules_config.get("spacy_refine_action_excluded_types", list(SPACY_REFINE_ACTION_EXCLUDED_TYPES))
        )

        self.action_scope_patterns_by_lang = {
            k: {kk: list(vv) for kk, vv in v.items()} for k, v in ACTION_SCOPE_PATTERNS_BY_LANG.items()
        }
        cfg_scope = self.rules_config.get("action_scope_patterns_by_lang", {})
        if isinstance(cfg_scope, dict):
            for lang, scopes in cfg_scope.items():
                if isinstance(scopes, dict):
                    self.action_scope_patterns_by_lang[str(lang)] = {
                        str(k): [str(p) for p in v] for k, v in scopes.items() if isinstance(v, list)
                    }
        self.action_scope_verb_hints = {
            k: list(v) for k, v in ACTION_SCOPE_VERB_HINTS.items()
        }
        cfg_scope_hints = self.rules_config.get("action_scope_verb_hints", {})
        if isinstance(cfg_scope_hints, dict):
            for key, vals in cfg_scope_hints.items():
                if isinstance(vals, list):
                    self.action_scope_verb_hints[str(key)] = [str(x) for x in vals]
        cleanup = dict(ACTION_SCOPE_CLEANUP)
        cleanup.update({str(k): str(v) for k, v in self.rules_config.get("action_scope_cleanup", {}).items()})
        self.action_scope_cleanup = cleanup

        self.object_action_modifier_patterns_by_lang = dict(OBJECT_ACTION_MODIFIER_PATTERNS_BY_LANG)
        self.object_action_modifier_patterns_by_lang.update(
            {str(k): str(v) for k, v in self.rules_config.get("object_action_modifier_patterns_by_lang", {}).items()}
        )
        self.object_action_modifier_skip_tokens_by_lang = {
            k: list(v) for k, v in OBJECT_ACTION_MODIFIER_SKIP_TOKENS_BY_LANG.items()
        }
        cfg_skip = self.rules_config.get("object_action_modifier_skip_tokens_by_lang", {})
        if isinstance(cfg_skip, dict):
            for lang, vals in cfg_skip.items():
                if isinstance(vals, list):
                    self.object_action_modifier_skip_tokens_by_lang[str(lang)] = [str(x) for x in vals]

        self.object_action_modifier_action_tokens_by_lang = {
            k: list(v) for k, v in OBJECT_ACTION_MODIFIER_ACTION_TOKENS_BY_LANG.items()
        }
        cfg_action_tokens = self.rules_config.get("object_action_modifier_action_tokens_by_lang", {})
        if isinstance(cfg_action_tokens, dict):
            for lang, vals in cfg_action_tokens.items():
                if isinstance(vals, list):
                    self.object_action_modifier_action_tokens_by_lang[str(lang)] = [str(x) for x in vals]

        self.object_action_modifier_attribute_suffixes_by_lang = {
            k: list(v) for k, v in OBJECT_ACTION_MODIFIER_ATTRIBUTE_SUFFIXES_BY_LANG.items()
        }
        cfg_attr_suffixes = self.rules_config.get("object_action_modifier_attribute_suffixes_by_lang", {})
        if isinstance(cfg_attr_suffixes, dict):
            for lang, vals in cfg_attr_suffixes.items():
                if isinstance(vals, list):
                    self.object_action_modifier_attribute_suffixes_by_lang[str(lang)] = [str(x) for x in vals]

        self.object_detail_rules_by_lang = {k: dict(v) for k, v in OBJECT_DETAIL_RULES_BY_LANG.items()}
        cfg_object_detail = self.rules_config.get("object_detail_rules_by_lang", {})
        if isinstance(cfg_object_detail, dict):
            for lang, rule_obj in cfg_object_detail.items():
                if isinstance(rule_obj, dict):
                    base = dict(self.object_detail_rules_by_lang.get(str(lang), {}))
                    base.update(rule_obj)
                    self.object_detail_rules_by_lang[str(lang)] = base

        self.object_phrase_fallback_by_lang = {k: dict(v) for k, v in OBJECT_PHRASE_FALLBACK_BY_LANG.items()}
        cfg_object_phrase = self.rules_config.get("object_phrase_fallback_by_lang", {})
        if isinstance(cfg_object_phrase, dict):
            for lang, rule_obj in cfg_object_phrase.items():
                if isinstance(rule_obj, dict):
                    base = dict(self.object_phrase_fallback_by_lang.get(str(lang), {}))
                    base.update(rule_obj)
                    self.object_phrase_fallback_by_lang[str(lang)] = base

        self.object_bare_phrase_fallback_by_lang = {
            str(k): dict(v) for k, v in OBJECT_BARE_PHRASE_FALLBACK_BY_LANG.items()
        }
        cfg_bare_object = self.rules_config.get("object_bare_phrase_fallback_by_lang", {})
        if isinstance(cfg_bare_object, dict):
            for lang, rule_obj in cfg_bare_object.items():
                if isinstance(rule_obj, dict):
                    base = dict(self.object_bare_phrase_fallback_by_lang.get(str(lang), {}))
                    base.update(rule_obj)
                    self.object_bare_phrase_fallback_by_lang[str(lang)] = base

        self.additional_condition_patterns_by_lang = {
            k: list(v) for k, v in ADDITIONAL_CONDITION_PATTERNS_BY_LANG.items()
        }
        cfg_additional = self.rules_config.get("additional_condition_patterns_by_lang", {})
        if isinstance(cfg_additional, dict):
            for lang, pairs in cfg_additional.items():
                if isinstance(pairs, list):
                    normalized_pairs: list[tuple[str, str]] = []
                    for item in pairs:
                        if isinstance(item, list) and len(item) == 2:
                            normalized_pairs.append((str(item[0]), str(item[1])))
                    self.additional_condition_patterns_by_lang[str(lang)] = normalized_pairs

        self.spacy_signal_patterns_by_lang = {
            k: dict(v) for k, v in SPACY_SIGNAL_PATTERNS_BY_LANG.items()
        }
        cfg_spacy_signals = self.rules_config.get("spacy_signal_patterns_by_lang", {})
        if isinstance(cfg_spacy_signals, dict):
            for lang, patt in cfg_spacy_signals.items():
                if isinstance(patt, dict):
                    self.spacy_signal_patterns_by_lang[str(lang)] = {
                        str(k): str(v) for k, v in patt.items()
                    }
        self.spacy_signal_dept_pattern = str(
            self.rules_config.get("spacy_signal_dept_pattern", SPACY_SIGNAL_DEPT_PATTERN)
        )
        self.spacy_signal_department_suffix_pattern_ja = str(
            self.rules_config.get("spacy_signal_department_suffix_pattern_ja", SPACY_SIGNAL_DEPARTMENT_SUFFIX_PATTERN_JA)
        )
        self.spacy_signal_ja_person_prefixes = [
            str(x) for x in self.rules_config.get("spacy_signal_ja_person_prefixes", SPACY_SIGNAL_JA_PERSON_PREFIXES)
        ]
        self.spacy_signal_person_time_suffix_pattern = str(
            self.rules_config.get("spacy_signal_person_time_suffix_pattern", SPACY_SIGNAL_PERSON_TIME_SUFFIX_PATTERN)
        )
        self.spacy_signal_latin_time_suffix_pattern = str(
            self.rules_config.get("spacy_signal_latin_time_suffix_pattern", SPACY_SIGNAL_LATIN_TIME_SUFFIX_PATTERN)
        )

        self.non_person_tokens_by_lang = {k: list(v) for k, v in NON_PERSON_TOKENS_BY_LANG.items()}
        cfg_non_person = self.rules_config.get("non_person_tokens_by_lang", {})
        if isinstance(cfg_non_person, dict):
            for lang, vals in cfg_non_person.items():
                if isinstance(vals, list):
                    self.non_person_tokens_by_lang[str(lang)] = [str(x).strip() for x in vals if str(x).strip()]

        self.instructor_client = None
        if self.use_instructor:
            if provider != "openai":
                raise ValueError("use_instructor is only supported with provider=openai")
            if instructor is None:
                raise ImportError(
                    "instructor is not installed. Install dependencies with: pip install -r requirements.txt"
                )
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError("OPENAI_API_KEY is not set.")
            self.instructor_client = instructor.from_openai(OpenAI())

        self.llm = None
        if provider != "spacy":
            self.llm = self._build_llm(
                provider=provider,
                model=model,
                temperature=temperature,
                ollama_base_url=ollama_base_url,
                litellm_api_base=litellm_api_base,
                litellm_api_key=litellm_api_key,
            )

        extract_system_prompt = EXTRACTION_SYSTEM_PROMPT
        if extra_rules.strip():
            extract_system_prompt = (
                f"{EXTRACTION_SYSTEM_PROMPT}\n\n[추가 교정 규칙]\n{extra_rules.strip()}"
            )

        self.extract_prompt = ChatPromptTemplate.from_messages(
            [("system", extract_system_prompt), ("human", EXTRACTION_HUMAN_PROMPT)]
        )
        self.refine_prompt = ChatPromptTemplate.from_messages(
            [("system", REFINE_SYSTEM_PROMPT), ("human", REFINE_HUMAN_PROMPT)]
        )
        self.multi_action_prompt = ChatPromptTemplate.from_messages(
            [("system", MULTI_ACTION_SYSTEM_PROMPT), ("human", MULTI_ACTION_HUMAN_PROMPT)]
        )

        self.extract_chain = None
        self.refine_chain = None
        self.multi_action_chain = None
        if self.llm is not None:
            self.extract_chain = self.extract_prompt | self.llm.with_structured_output(ExtractionResult)
            self.refine_chain = self.refine_prompt | self.llm.with_structured_output(ExtractionResult)
            self.multi_action_chain = self.multi_action_prompt | self.llm.with_structured_output(
                MultiActionExtractionResult
            )

    @staticmethod
    def _pick_lang_value(values_by_lang: dict[str, list[str]], lang: str, default_lang: str, fallback: str) -> str:
        """언어별 값 맵에서 우선순위(`lang` -> `default_lang` -> `fallback`)로 값을 선택합니다.

        Args:
            values_by_lang: 언어 코드별 후보 문자열 리스트.
            lang: 현재 판별된 언어 코드.
            default_lang: 현재 언어에 값이 없을 때 사용할 기본 언어 코드.
            fallback: 모든 후보가 없을 때 마지막으로 사용할 값.

        예시:
            self._pick_lang_value({"en": ["customer"]}, "fr", "en", "user") -> "customer"
        """
        return common_pick_lang_value(values_by_lang, lang, default_lang, fallback)

    @staticmethod
    def _pick_lang_value_by_text(
        values_by_lang: dict[str, list[str]],
        lang: str,
        text: str,
        default_lang: str,
        fallback: str,
        case_insensitive_langs: set[str] | None = None,
    ) -> str:
        """언어별 후보 패턴 중 텍스트에 실제로 매칭되는 값을 우선 선택합니다.

        매칭이 없으면 `default_lang` 후보의 첫 값을 사용하고, 그마저 없으면 fallback을 반환합니다.

        예시:
            self._pick_lang_value_by_text({"en": ["send", "forward"]}, "en", "please forward", "en", "request")
        """
        return common_pick_lang_value_by_text(
            values_by_lang=values_by_lang,
            lang=lang,
            text=text,
            default_lang=default_lang,
            fallback=fallback,
            case_insensitive_langs=case_insensitive_langs,
        )

    @staticmethod
    def _build_llm(
        provider: Literal["openai", "ollama", "litellm", "spacy"],
        model: str,
        temperature: float,
        ollama_base_url: str,
        litellm_api_base: str,
        litellm_api_key: str,
    ):
        """provider 설정에 맞는 LLM 클라이언트를 생성합니다.

        `spacy` provider는 LLM을 사용하지 않으므로 `None`을 반환합니다.

        Returns:
            ChatOpenAI/ChatOllama/ChatLiteLLM 인스턴스 또는 None.
        """
        return llm_build_backend(
            provider=provider,
            model=model,
            temperature=temperature,
            ollama_base_url=ollama_base_url,
            litellm_api_base=litellm_api_base,
            litellm_api_key=litellm_api_key,
            chat_openai_cls=ChatOpenAI,
            chat_ollama_cls=ChatOllama,
            chat_litellm_cls=ChatLiteLLM,
        )

    def extract(self, text: str) -> ExtractionResult:
        """입력 텍스트에서 단일 액션을 추출합니다.

        예시:
            extractor.extract("Please cancel my subscription by Friday.")
        """
        if self.provider == "spacy":
            return spacy_extract_single_backend(self, text)
        return llm_extract_single_backend(self, text)

    def extract_by_verb(self, text: str) -> MultiActionExtractionResult:
        """동사 단위로 분해하여 다중 액션을 추출합니다.

        예시:
            extractor.extract_by_verb("Summarize and send the message to Alice.")
        """
        if self.provider == "spacy":
            return spacy_extract_multi_backend(self, text)
        return llm_extract_multi_backend(self, text)

    def _extract_spacy_only(self, text: str) -> ExtractionResult:
        """spaCy 규칙 기반 단일 추출 payload를 `ExtractionResult`로 변환합니다.

        이 함수는 LLM 경로를 완전히 우회합니다.
        """
        payload = build_spacy_single_payload(self, text)
        return ExtractionResult(**payload)

    def _extract_by_verb_spacy_only(self, text: str) -> MultiActionExtractionResult:
        """spaCy 규칙 기반 split-by-verb payload를 모델 객체로 변환합니다.

        `actions` 항목은 `VerbAction`으로 재검증하여 타입 안전성을 확보합니다.
        """
        payload = build_spacy_multi_payload(self, text)
        actions = [VerbAction(**a) for a in payload.get("actions", [])]
        return MultiActionExtractionResult(
            language=str(payload.get("language", "en")),
            actions=actions,
            confidence=float(payload.get("confidence", self.spacy_only_multi_confidence)),
        )

    def _pick_summarize_object(self, text: str, lang: str) -> str:
        """요약 의도에 맞는 object 라벨을 언어별 규칙로 선택합니다.

        예시:
            self._pick_summarize_object("Summarize the message", "en")
        """
        detailed = self._pick_detailed_object(text, lang)
        if detailed:
            return detailed
        for pattern, label in self.summarize_object_rules_by_lang.get(lang, []):
            flags = re.I if re.search(r"[A-Za-z]", pattern) else 0
            if re.search(pattern, text, flags=flags):
                return label
        return self._pick_object(text, lang)

    def _get_summarize_hint_keywords(self, lang: str) -> list[str]:
        """default/global/language 스코프의 요약 힌트 키워드를 병합합니다.

        예시:
            self._get_summarize_hint_keywords("en")
        """
        merged: list[str] = []
        for key in ("default", "all", "*", lang):
            for kw in self.summarize_hint_keywords_by_lang.get(key, []):
                if kw not in merged:
                    merged.append(kw)
        return merged

    @staticmethod
    def _detect_language(text: str) -> str:
        """공용 휴리스틱 탐지기로 요청 언어를 판별합니다.

        예시:
            self._detect_language("Résume les messages") -> "fr"
        """
        return common_detect_language(text)

    def _pick_main_verb(self, text: str, lang: str = "ko") -> str:
        """설정된 정규식 규칙에서 최적의 핵심 동사를 선택합니다.

        예시:
            self._pick_main_verb("Please send the report", "en") -> "send"
        """
        picked = common_pick_main_verb(
            text=text,
            lang=lang,
            verb_patterns_by_lang=self.verb_patterns_by_lang,
            default_verb_candidates_by_lang=self.default_verb_candidates_by_lang,
        )
        default_verb = self._pick_lang_value(self.default_verb_candidates_by_lang, lang, "en", "request")
        if picked == default_verb:
            patterns = self.question_patterns_by_lang.get(lang) or self.question_patterns_by_lang.get("en", [])
            for patt in patterns:
                try:
                    if re.search(patt, text, flags=re.I if lang in {"en", "de", "fr"} else 0):
                        return self._pick_lang_value(self.query_verb_by_lang, lang, "en", "query")
                except re.error:
                    continue
            surface_verb = self._infer_surface_verb_on_fallback(text, lang)
            if surface_verb:
                return surface_verb
        return picked

    def _infer_surface_verb_on_fallback(self, text: str, lang: str) -> str:
        """기본 동사 fallback 상황에서 원문 표면 술어를 최대한 그대로 반환합니다.

        우선순위:
            1) spaCy가 있으면 ROOT 토큰
            2) 언어별 종결형/말미 토큰 휴리스틱
        """
        normalized = (text or "").strip()
        if not normalized:
            return ""

        if self.spacy_nlp is not None:
            try:
                doc = self.spacy_nlp(normalized)
                for tok in doc:
                    if (tok.dep_ or "").upper() == "ROOT" and tok.text.strip():
                        return tok.text.strip()
                for tok in reversed(doc):
                    if (tok.pos_ or "").upper() in {"VERB", "AUX"} and tok.text.strip():
                        return tok.text.strip()
            except Exception:
                # POS/DEP가 없는 모델이어도 추출 파이프라인은 계속 유지한다.
                pass

        if lang == "ko":
            tokens = re.findall(r"[가-힣]+", normalized)
            if not tokens:
                return ""
            endings = (
                "했습니다",
                "하였다",
                "했다",
                "합니다",
                "한다",
                "됩니다",
                "되었다",
                "됐다",
                "였습니다",
                "였다",
                "입니다",
                "이다",
            )
            for tok in reversed(tokens):
                if len(tok) >= 2 and tok.endswith(endings):
                    return tok
            return tokens[-1]

        if lang in {"en", "de", "fr"}:
            latin_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+", normalized)
            if latin_tokens:
                return latin_tokens[-1]

        return ""

    def _pick_object(self, text: str, lang: str = "ko") -> str:
        """detailed/object/fallback 규칙 계층으로 canonical object를 선택합니다.

        예시:
            self._pick_object("Update the shipping address", "en")
        """
        detailed = self._pick_detailed_object(text, lang)
        if detailed:
            return detailed

        lang_rules = self.object_rules_by_lang.get(lang, [])
        common_rules = self.object_rules_by_lang.get("en", []) if lang != "en" else []
        for pattern, label in [*lang_rules, *common_rules]:
            flags = re.I if re.search(r"[A-Za-z]", pattern) else 0
            if re.search(pattern, text, flags=flags):
                return label

        fallback_obj = self._pick_object_phrase_fallback(text, lang)
        if fallback_obj:
            return fallback_obj

        bare = self._pick_bare_object_phrase_fallback(text, lang)
        if bare:
            return bare

        return self._pick_lang_value(self.default_object_candidates_by_lang, lang, "en", "request target")

    def _pick_bare_object_phrase_fallback(self, text: str, lang: str) -> str:
        """동사 없는 단문/명사구 입력에서 언어별 규칙로 object를 보존합니다."""
        rule_obj = self.object_bare_phrase_fallback_by_lang.get(lang) or {}
        if not isinstance(rule_obj, dict):
            return ""

        bare = re.sub(r"[\s\u3000]+", " ", text).strip()
        bare = re.sub(r"[\.!?؟]+$", "", bare).strip()
        if not bare:
            return ""

        max_length = int(rule_obj.get("max_length", 0) or 0)
        if max_length > 0 and len(bare) > max_length:
            return ""

        allowed_pattern = str(rule_obj.get("allowed_pattern", "")).strip()
        if allowed_pattern and not re.fullmatch(allowed_pattern, bare):
            return ""

        endings = [str(x).strip() for x in rule_obj.get("must_end_with", []) if str(x).strip()]
        if endings:
            matched = False
            for patt in endings:
                flags = re.I if lang in {"en", "de", "fr"} else 0
                try:
                    if re.search(patt, bare, flags=flags):
                        matched = True
                        break
                except re.error:
                    continue
            if not matched:
                return ""

        return bare

    def _pick_detailed_object(self, text: str, lang: str) -> str:
        """canonical 정규화 전에 detailed object 구절을 추출합니다.

        예시:
            self._pick_detailed_object("messages John sent yesterday summarize", "en")
        """
        rule_obj = self.object_detail_rules_by_lang.get(lang) or self.object_detail_rules_by_lang.get("en") or {}
        command_hint = str(rule_obj.get("command_hint", "")).strip()
        patterns = [str(p) for p in rule_obj.get("patterns", []) if str(p).strip()]

        for raw in patterns:
            patt = raw.replace("{command_hint}", command_hint)
            flags = re.I if re.search(r"[A-Za-z]", patt) else 0
            m = re.search(patt, text, flags=flags)
            if m:
                detailed = re.sub(r"\s+", " ", m.group(1).strip())
                return self._normalize_detailed_object(detailed, lang)

        return ""

    @staticmethod
    def _normalize_detailed_object(detailed: str, lang: str) -> str:
        """detailed object 구절을 안정적인 canonical 라벨로 정규화합니다.

        예시:
            _normalize_detailed_object("messages John sent yesterday", "en") -> "message"
        """
        cand = re.sub(r"\s+", " ", (detailed or "").strip())
        if not cand:
            return ""

        if lang == "en":
            # Canonicalize overly specific EN phrases like
            # "messages John sent yesterday" -> "message".
            if re.search(r"\bchat\s+messages?\b", cand, flags=re.I):
                return "chat message"
            if re.search(r"\bemails?\b", cand, flags=re.I):
                return "email"
            if re.search(r"\bmessages?\b", cand, flags=re.I):
                return "message"

        if lang == "ko":
            if re.search(r"게시판\s*내용", cand):
                return "게시판 내용"
            if "게시판" in cand:
                return "게시판"

        if lang == "de":
            if re.search(r"\b(chat[-\s]?nachrichten?)\b", cand, flags=re.I):
                return "Nachricht"
            if re.search(r"\b(e-?mails?)\b", cand, flags=re.I):
                return "E-Mail"
            if re.search(r"\b(nachrichten?)\b", cand, flags=re.I):
                return "Nachricht"

        if lang == "fr":
            if re.search(r"\b(messages?\s+de\s+chat|chat)\b", cand, flags=re.I):
                return "message"
            if re.search(r"\b(e-?mails?|courriels?)\b", cand, flags=re.I):
                return "e-mail"
            if re.search(r"\b(messages?)\b", cand, flags=re.I):
                return "message"

        return cand

    def _pick_object_phrase_fallback(self, text: str, lang: str) -> str:
        """직접 object 규칙이 실패했을 때 fallback 구절 추출을 수행합니다.

        예시:
            self._pick_object_phrase_fallback("배송지 주소를 변경", "ko")
        """
        rule_obj = self.object_phrase_fallback_by_lang.get(lang, {})
        capture_pattern = str(rule_obj.get("capture_pattern", "")).strip()
        if not capture_pattern:
            return ""

        flags = re.I if re.search(r"[A-Za-z]", capture_pattern) else 0
        m = re.search(capture_pattern, text, flags=flags)
        if not m:
            return ""

        cand = m.group(1).strip()
        strip_prefix_pattern = str(rule_obj.get("strip_prefix_pattern", "")).strip()
        if strip_prefix_pattern:
            cand = re.sub(strip_prefix_pattern, "", cand).strip()

        head_suffix_map = rule_obj.get("head_suffix_map", {})
        if isinstance(head_suffix_map, dict):
            for suffix, replacement in head_suffix_map.items():
                if cand.endswith(str(suffix)):
                    return str(replacement)

        if cand:
            return cand
        return ""

    def _pick_subject(self, text: str, lang: str, obj: str, verb: str) -> str:
        """주어(subject)를 토픽 패턴/술어 규칙/기본값 순서로 선택합니다.

        동작 개요:
            1) 언어별 topic_pattern으로 명시된 주제를 우선 추출
            2) 특정 술어(가능/실패/중단 등)일 때 object를 주어로 승격
            3) 아무 조건도 맞지 않으면 언어 기본 subject 사용

        예시:
            self._pick_subject("배송이 지연됩니다", "ko", "배송", "지연됩니다") -> "배송"
        """
        fallback = self._pick_lang_value(self.subject_by_lang, lang, "en", "customer")
        rules = self.subject_picker_rules_by_lang.get(lang)
        if not isinstance(rules, dict):
            return fallback

        topic_pattern = str(rules.get("topic_pattern", "")).strip()
        topic_group = int(rules.get("topic_group", 1) or 1)
        topic_exclude = {str(x).strip().casefold() for x in rules.get("topic_exclude", []) if str(x).strip()}
        predicate_verbs = [str(x).strip() for x in rules.get("predicate_verb_keywords", []) if str(x).strip()]
        predicate_patterns = [str(x).strip() for x in rules.get("predicate_verb_patterns", []) if str(x).strip()]
        priority_subject_patterns_raw = rules.get("priority_subject_patterns", [])
        priority_subject_patterns: list[tuple[str, int]] = []
        if isinstance(priority_subject_patterns_raw, list):
            for item in priority_subject_patterns_raw:
                if isinstance(item, str) and item.strip():
                    priority_subject_patterns.append((item.strip(), 1))
                elif isinstance(item, dict):
                    patt = str(item.get("pattern", "")).strip()
                    grp = int(item.get("group", 1) or 1)
                    if patt:
                        priority_subject_patterns.append((patt, grp))
        subject_cleanup_regexes_raw = rules.get("subject_cleanup_regexes", [])
        subject_cleanup_regexes: list[tuple[str, str]] = []
        if isinstance(subject_cleanup_regexes_raw, list):
            for item in subject_cleanup_regexes_raw:
                if isinstance(item, list) and len(item) == 2:
                    patt = str(item[0]).strip()
                    repl = str(item[1])
                    if patt:
                        subject_cleanup_regexes.append((patt, repl))
                elif isinstance(item, dict):
                    patt = str(item.get("pattern", "")).strip()
                    repl = str(item.get("replacement", ""))
                    if patt:
                        subject_cleanup_regexes.append((patt, repl))
        object_skip_values = {
            str(x).strip().casefold() for x in rules.get("object_skip_values", []) if str(x).strip()
        }
        suffix_overrides_raw = rules.get("object_suffix_subject_overrides", {})
        suffix_overrides = dict(suffix_overrides_raw) if isinstance(suffix_overrides_raw, dict) else {}
        allow_subject_from_object = _as_bool(rules.get("allow_subject_from_object", True), True)

        def _normalize_subject_candidate(cand: str) -> str:
            out = re.sub(r"\s+", " ", (cand or "").strip())
            for patt, repl in subject_cleanup_regexes:
                out = re.sub(patt, repl, out, flags=0).strip()
            return out

        for patt, grp in priority_subject_patterns:
            try:
                m = re.search(patt, text, flags=0)
            except re.error:
                continue
            if not m:
                continue
            if m.lastindex and m.lastindex >= grp:
                cand = m.group(grp).strip()
            else:
                cand = m.group(0).strip()
            cand = _normalize_subject_candidate(cand)
            if cand and cand.casefold() not in topic_exclude:
                return cand

        if topic_pattern:
            try:
                topic_matches = list(re.finditer(topic_pattern, text))
                if topic_matches:
                    matched = topic_matches[-1]
                    if matched.lastindex and matched.lastindex >= topic_group:
                        cand = matched.group(topic_group).strip()
                    else:
                        cand = matched.group(0).strip()
                    cand = _normalize_subject_candidate(cand)
                    if cand and cand.casefold() not in topic_exclude:
                        return cand
            except re.error:
                # 사용자 제공 정규식 오류가 있어도 전체 추출 파이프라인은 유지한다.
                pass

        obj_norm = (obj or "").strip()
        verb_norm = (verb or "").casefold()
        should_use_object = any(k.casefold() in verb_norm for k in predicate_verbs)
        if not should_use_object and predicate_patterns:
            for patt in predicate_patterns:
                flags = re.I if re.search(r"[A-Za-z]", patt) else 0
                if re.search(patt, verb, flags=flags):
                    should_use_object = True
                    break

        if allow_subject_from_object and obj_norm and obj_norm.casefold() not in object_skip_values and should_use_object:
            for suffix, subj in suffix_overrides.items():
                if obj_norm.endswith(str(suffix)):
                    return str(subj)
            return obj_norm

        return fallback

    def _extract_time_conditions(self, text: str, lang: str = "ko") -> list[dict]:
        """시간 조건(time)을 추출하고, 포괄 관계가 있는 중복 표현을 정리합니다.

        예: "월,화요일"이 있으면 하위 토큰("월", "화요일")은 제거합니다.
        """
        conds: list[dict] = []
        patterns = [*self.time_patterns_common, *self.time_patterns_by_lang.get(lang, [])]
        flags = re.I if lang in self.time_case_insensitive_langs else 0
        for p in patterns:
            for m in re.finditer(p, text, flags=flags):
                conds.append({"type": "time", "text": m.group(0).strip()})

        # 더 긴 기간/요일 묶음 표현을 우선 유지하고 하위 토큰을 제거한다.
        deduped = self._dedupe_conditions(conds)
        texts = [str(c.get("text", "")).strip() for c in deduped]
        ordered = sorted(texts, key=len, reverse=True)

        def _normalize_for_compare(value: str) -> str:
            # 공백/쉼표 차이를 정규화해 동일한 기간 표현을 같은 값으로 비교한다.
            return re.sub(r"[\s,]", "", value)

        kept: list[str] = []
        kept_norm: list[str] = []
        for t in ordered:
            if not t:
                continue
            norm_t = _normalize_for_compare(t)
            if any((norm_t in k and norm_t != k) or norm_t == k for k in kept_norm):
                continue
            kept.append(t)
            kept_norm.append(norm_t)

        kept_set = set(kept)
        return [c for c in deduped if str(c.get("text", "")).strip() in kept_set]

    def _extract_manner_conditions(self, text: str, lang: str = "ko") -> list[dict]:
        """전달 방식/형태(manner) 조건을 추출합니다.

        예: "이메일로", "by email", "PDF로" 등.
        요약 의도 키워드도 action 분해를 위해 manner로 함께 기록합니다.
        """
        conds: list[dict] = []
        tokens = self.manner_tokens_by_lang.get(lang, [])
        if lang != "en":
            tokens = [*tokens, *self.manner_tokens_by_lang.get("en", [])]
        for token in tokens:
            if token in text:
                conds.append({"type": "manner", "text": token})

        # 요청 분해(split-by-verb)를 위해 요약 의도를 manner 조건으로도 남긴다.
        lower_text = text.lower()
        summarize_hints = self._get_summarize_hint_keywords(lang)
        for hint in summarize_hints:
            if hint and hint.lower() in lower_text:
                conds.append({"type": "manner", "text": hint})

        return conds

    def _extract_additional_conditions(self, text: str, lang: str = "ko") -> list[dict]:
        """추가 조건(reason/constraint/location 등)을 언어별 패턴으로 추출합니다."""
        conds: list[dict] = []
        patterns = self.additional_condition_patterns_by_lang.get(lang, [])
        if not patterns:
            return conds

        for ctype, patt in patterns:
            flags = re.I if re.search(r"[A-Za-z]", patt) else 0
            for m in re.finditer(patt, text, flags=flags):
                token = m.group(1).strip()
                if token:
                    conds.append({"type": ctype, "text": token})
        return self._dedupe_conditions(conds)

    @staticmethod
    def _build_spacy_nlp(model_name: str):
        """spaCy 모델을 초기화합니다. 모델이 없으면 backend fallback 로직을 따릅니다."""
        return spacy_build_backend(model_name=model_name, spacy_module=spacy)

    def _apply_spacy_postprocess_single(self, result: ExtractionResult, text: str) -> ExtractionResult:
        """단일 추출 결과에 spaCy 기반 condition 보정 후처리를 적용합니다.

        `object_text`를 함께 넘겨 object 내부 action 수식어까지 반영합니다.
        """
        if not self.use_spacy_postprocess:
            return result

        data = result.model_dump()
        data["conditions"] = self._spacy_refine_conditions(
            data.get("conditions", []),
            text,
            object_text=str(data.get("object", "")),
        )
        return ExtractionResult(**data)

    def _apply_spacy_postprocess_multi(
        self, result: MultiActionExtractionResult, text: str
    ) -> MultiActionExtractionResult:
        """다중 액션 결과 각 항목에 spaCy 후처리를 적용합니다.

        multi 모드에서는 action별 스코프가 달라서 `add_global_entities=False`를 사용합니다.
        """
        if not self.use_spacy_postprocess:
            return result

        data = result.model_dump()
        actions = []
        for action in data.get("actions", []):
            action_data = dict(action)
            action_data["conditions"] = self._spacy_refine_conditions(
                action_data.get("conditions", []),
                text,
                object_text=str(action_data.get("object", "")),
                verb_text=str(action_data.get("verb", "")),
                add_global_entities=False,
            )
            actions.append(action_data)
        data["actions"] = actions
        return MultiActionExtractionResult(**data)

    def _spacy_refine_conditions(
        self,
        conditions: list[dict],
        text: str,
        object_text: str = "",
        verb_text: str = "",
        add_global_entities: bool = True,
    ) -> list[dict]:
        """spaCy backend의 condition 타입 보정 로직을 호출하는 래퍼입니다.

        내부적으로 person/department/action 재분류와 중복 제거를 수행합니다.
        """
        return spacy_refine_conditions_backend(
            extractor=self,
            conditions=conditions,
            text=text,
            object_text=object_text,
            verb_text=verb_text,
            add_global_entities=add_global_entities,
            korean_person_stopwords=KOREAN_PERSON_STOPWORDS,
            generic_person_stopwords=GENERIC_PERSON_STOPWORDS,
            department_suffixes=DEPARTMENT_SUFFIXES,
        )

    def _extract_action_scoped_entities(self, text: str, verb_text: str) -> dict[str, set[str]]:
        """동사 문맥(요약/전송)에 맞춰 action 스코프 엔티티를 추출합니다.

        Returns:
            {
              "persons": set[str],
              "departments": set[str],
              "sender_persons": set[str],
              "receiver_persons": set[str],
              "receiver_departments": set[str],
            } 형태의 엔티티 맵.
        """
        return spacy_extract_action_scoped_entities_backend(self, text, verb_text)

    def _extract_object_action_modifiers(self, object_text: str) -> list[str]:
        """object 문자열 내부의 action 수식어 토큰을 추출합니다.

        예: "보낸 쪽지" -> ["보낸"]
        """
        return spacy_extract_object_action_modifiers_backend(self, object_text)

    def _extract_spacy_signals(self, text: str) -> dict[str, set[str]]:
        """spaCy + 규칙 기반 시그널(person/department/action)을 추출합니다.

        Returns:
            {"persons": set[str], "departments": set[str], "actions": set[str]}.
        """
        return spacy_extract_signals_backend(
            extractor=self,
            text=text,
            korean_person_stopwords=KOREAN_PERSON_STOPWORDS,
            generic_person_stopwords=GENERIC_PERSON_STOPWORDS,
            department_suffixes=DEPARTMENT_SUFFIXES,
        )

    def _apply_guardrails_single(self, result: ExtractionResult) -> ExtractionResult:
        """단일 추출 결과에 안전 가드레일을 적용합니다.

        언어값 정규화, 필드 공백 제거, condition 중복 제거, confidence 범위 보정(0~1)을 수행합니다.
        핵심 필드(subject/verb/object)가 비면 원본 결과를 유지해 구조 손상을 방지합니다.
        """
        if not self.use_guardrails:
            return result

        data = result.model_dump()
        data["language"] = self._normalize_language(data.get("language", ""))
        data["subject"] = data.get("subject", "").strip()
        data["verb"] = data.get("verb", "").strip()
        data["object"] = data.get("object", "").strip()
        data["conditions"] = self._dedupe_conditions(data.get("conditions", []))
        data["confidence"] = self._clamp_confidence(data.get("confidence", 0.0))

        # 핵심 필드가 비어버리면 무효 구조를 내보내지 않고 원본 결과를 유지한다.
        if not data["subject"] or not data["verb"] or not data["object"]:
            return result

        return ExtractionResult(**data)

    def _apply_guardrails_multi(self, result: MultiActionExtractionResult) -> MultiActionExtractionResult:
        """다중 액션 결과에 안전 가드레일을 적용합니다.

        비어 있는 액션은 제거하고, 남은 액션의 조건을 정규화/중복 제거합니다.
        """
        if not self.use_guardrails:
            return result

        data = result.model_dump()
        data["language"] = self._normalize_language(data.get("language", ""))
        cleaned_actions = []
        for action in data.get("actions", []):
            subject = action.get("subject", "").strip()
            verb = action.get("verb", "").strip()
            obj = action.get("object", "").strip()
            if not subject or not verb or not obj:
                continue
            cleaned_actions.append(
                {
                    "subject": subject,
                    "verb": verb,
                    "object": obj,
                    "conditions": self._dedupe_conditions(action.get("conditions", [])),
                }
            )

        if cleaned_actions:
            data["actions"] = cleaned_actions
        data["confidence"] = self._clamp_confidence(data.get("confidence", 0.0))
        return MultiActionExtractionResult(**data)

    @staticmethod
    def _normalize_language(language: str) -> str:
        """언어 코드를 정규화하고 미지원 값은 `en`으로 fallback합니다.

        입력 공백/대소문자를 정리한 뒤 지원 언어 집합에 없으면
        운영 안정성을 위해 기본값 `en`을 반환합니다.

        예시:
            _normalize_language("EN") -> "en"
            _normalize_language("xx") -> "en"
        """
        lang = (language or "").strip().lower()
        return lang if lang in SUPPORTED_LANGUAGES else "en"

    @staticmethod
    def _clamp_confidence(confidence: float) -> float:
        """confidence 값을 0.0~1.0 범위로 제한합니다.

        숫자 변환이 실패하면 0.0을 반환합니다.

        예시:
            _clamp_confidence(1.2) -> 1.0
            _clamp_confidence(-0.1) -> 0.0
        """
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _dedupe_conditions(conditions: list[dict]) -> list[dict]:
        """(type, text) 기준으로 condition 중복을 제거합니다.

        비어 있는 text는 버리고, 최초 등장 순서를 보존합니다.

        예시:
            [{"type": "time", "text": "어제"}, {"type": "time", "text": "어제"}]
            -> [{"type": "time", "text": "어제"}]
        """
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for c in conditions:
            ctype = str(c.get("type", "other")).strip() or "other"
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            key = (ctype, text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"type": ctype, "text": text})
        return deduped

    def _extract_with_instructor(self, text: str) -> ExtractionResult:
        """Instructor 백엔드로 2단계(초안/정제) 단일 추출을 수행합니다.

        처리 순서:
            1) EXTRACTION 프롬프트로 1차 구조화 추출
            2) 1차 JSON을 REFINE 프롬프트에 넣어 최종 정제

        Raises:
            AssertionError: instructor client가 초기화되지 않은 경우.
        """
        assert self.instructor_client is not None
        first_pass = self.instructor_client.chat.completions.create(
            model=self.model,
            response_model=ExtractionResult,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": EXTRACTION_HUMAN_PROMPT.format(text=text)},
            ],
        )
        final_pass = self.instructor_client.chat.completions.create(
            model=self.model,
            response_model=ExtractionResult,
            messages=[
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": REFINE_HUMAN_PROMPT.format(
                        text=text,
                        first_pass_json=first_pass.model_dump_json(indent=2, ensure_ascii=False),
                    ),
                },
            ],
        )
        return final_pass

    def _extract_by_verb_with_instructor(self, text: str) -> MultiActionExtractionResult:
        """Instructor 백엔드로 split-by-verb 다중 추출을 수행합니다.

        문장 내 복수 동작을 `actions[]` 스키마로 직접 생성합니다.

        Raises:
            AssertionError: instructor client가 초기화되지 않은 경우.
        """
        assert self.instructor_client is not None
        return self.instructor_client.chat.completions.create(
            model=self.model,
            response_model=MultiActionExtractionResult,
            messages=[
                {"role": "system", "content": MULTI_ACTION_SYSTEM_PROMPT},
                {"role": "user", "content": MULTI_ACTION_HUMAN_PROMPT.format(text=text)},
            ],
        )


def _read_rules_file(path: str) -> str:
    """추가 프롬프트 규칙 텍스트 파일을 UTF-8로 읽습니다.

    파일이 없으면 `FileNotFoundError`를 발생시켜 호출자에게 명확히 알립니다.

    예시:
        _read_rules_file("prompts/tuned_rules.txt")
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rules file not found: {path}")
    return p.read_text(encoding="utf-8")


def extract_one(
    text: str,
    model: str = "gpt-4.1",
    extra_rules_file: str = "",
    rules_config_file: str = "",
    provider: Literal["openai", "ollama", "litellm", "spacy"] = "openai",
    ollama_base_url: str = "http://localhost:11434",
    use_instructor: bool = False,
    use_guardrails: bool = False,
    use_spacy_postprocess: bool | None = None,
    spacy_model: str = "xx_ent_wiki_sm",
    litellm_api_base: str = "",
    litellm_api_key: str = "",
) -> dict:
    """단일 텍스트 추출용 편의 API입니다.

    내부에서 `MultilingualSVOExtractor`를 생성하고 `extract()` 결과를 dict로 반환합니다.
    CLI가 아닌 코드 경로에서 간단히 호출할 때 사용합니다.

    Args:
        text: 추출할 원문 문자열.
        model: 사용할 모델명.
        extra_rules_file: 추가 프롬프트 규칙 파일 경로.
        rules_config_file: 동적 규칙 JSON 파일 경로.
        provider: 추출 provider.
        ollama_base_url: Ollama 서버 주소.
        use_instructor: Instructor 모드 사용 여부.
        use_guardrails: guardrails 적용 여부.
        use_spacy_postprocess: spaCy 후처리 적용 여부.
        spacy_model: spaCy 모델명.
        litellm_api_base: LiteLLM API base URL.
        litellm_api_key: LiteLLM API key.

    Returns:
        단일 추출 결과 dict(language/subject/verb/object/conditions/confidence).

    Raises:
        ValueError: text가 비어 있을 때.
        FileNotFoundError: extra_rules_file 또는 rules_config_file 경로가 잘못되었을 때.

    예시:
        extract_one("Please cancel the subscription", provider="spacy")
    """
    if not text.strip():
        raise ValueError("text must not be empty")
    extra_rules = _read_rules_file(extra_rules_file) if extra_rules_file else ""
    extractor = MultilingualSVOExtractor(
        provider=provider,
        model=model,
        extra_rules=extra_rules,
        ollama_base_url=ollama_base_url,
        use_instructor=use_instructor,
        use_guardrails=use_guardrails,
        use_spacy_postprocess=use_spacy_postprocess,
        spacy_model=spacy_model,
        litellm_api_base=litellm_api_base,
        litellm_api_key=litellm_api_key,
        rules_config_file=rules_config_file,
    )
    result = extractor.extract(text)
    return result.model_dump()


def extract_many(
    texts: list[str],
    model: str = "gpt-4.1",
    extra_rules_file: str = "",
    rules_config_file: str = "",
    provider: Literal["openai", "ollama", "litellm", "spacy"] = "openai",
    ollama_base_url: str = "http://localhost:11434",
    use_instructor: bool = False,
    use_guardrails: bool = False,
    use_spacy_postprocess: bool | None = None,
    spacy_model: str = "xx_ent_wiki_sm",
    litellm_api_base: str = "",
    litellm_api_key: str = "",
) -> list[dict]:
    """여러 텍스트를 일괄 추출하는 편의 API입니다.

    extractor 인스턴스를 한 번만 생성하여 반복 호출하므로 개별 호출보다 효율적입니다.

    Args:
        texts: 추출할 문자열 목록.
        model/provider/...: `extract_one`과 동일.

    Returns:
        입력 순서를 유지한 결과 dict 목록.

    Raises:
        FileNotFoundError: extra_rules_file 또는 rules_config_file 경로가 잘못되었을 때.

    예시:
        extract_many(["요약해줘", "보내줘"], provider="spacy")
    """
    extra_rules = _read_rules_file(extra_rules_file) if extra_rules_file else ""
    extractor = MultilingualSVOExtractor(
        provider=provider,
        model=model,
        extra_rules=extra_rules,
        ollama_base_url=ollama_base_url,
        use_instructor=use_instructor,
        use_guardrails=use_guardrails,
        use_spacy_postprocess=use_spacy_postprocess,
        spacy_model=spacy_model,
        litellm_api_base=litellm_api_base,
        litellm_api_key=litellm_api_key,
        rules_config_file=rules_config_file,
    )
    outputs: list[dict] = []
    for text in texts:
        outputs.append(extractor.extract(text).model_dump())
    return outputs


def extract_by_verb(
    text: str,
    model: str = "gpt-4.1",
    extra_rules_file: str = "",
    rules_config_file: str = "",
    provider: Literal["openai", "ollama", "litellm", "spacy"] = "openai",
    ollama_base_url: str = "http://localhost:11434",
    use_instructor: bool = False,
    use_guardrails: bool = False,
    use_spacy_postprocess: bool | None = None,
    spacy_model: str = "xx_ent_wiki_sm",
    litellm_api_base: str = "",
    litellm_api_key: str = "",
) -> dict:
    """동사 분해(split-by-verb) 추출용 편의 API입니다.

    하나의 문장에서 복수 액션이 포함된 요청을 `actions[]` 형태로 반환합니다.

    Args:
        text: 추출할 원문 문자열.
        model/provider/...: `extract_one`과 동일.

    Returns:
        다중 액션 결과 dict(language/actions/confidence).

    Raises:
        ValueError: text가 비어 있을 때.
        FileNotFoundError: extra_rules_file 또는 rules_config_file 경로가 잘못되었을 때.

    예시:
        extract_by_verb("요약해서 보내줘", provider="spacy")
    """
    if not text.strip():
        raise ValueError("text must not be empty")
    extra_rules = _read_rules_file(extra_rules_file) if extra_rules_file else ""
    extractor = MultilingualSVOExtractor(
        provider=provider,
        model=model,
        extra_rules=extra_rules,
        ollama_base_url=ollama_base_url,
        use_instructor=use_instructor,
        use_guardrails=use_guardrails,
        use_spacy_postprocess=use_spacy_postprocess,
        spacy_model=spacy_model,
        litellm_api_base=litellm_api_base,
        litellm_api_key=litellm_api_key,
        rules_config_file=rules_config_file,
    )
    result = extractor.extract_by_verb(text)
    return result.model_dump()


def cli() -> None:
    """다국어 추출을 위한 CLI 진입점입니다.

    provider/spacy 후처리/guardrails/split-by-verb 등 실행 옵션을 받아
    추출 결과를 JSON으로 출력합니다.

    예시:
        python src/extractor.py --provider spacy --split-by-verb --text "정리해서 보내줘"
    """
    import argparse

    parser = argparse.ArgumentParser(description="Multilingual SVO + Condition Extractor")
    parser.add_argument("--text", required=True, help="Input sentence")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "ollama", "litellm", "spacy"],
        help="LLM provider",
    )
    parser.add_argument("--model", default="gpt-4.1", help="Model name (OpenAI or Ollama local model)")
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama server URL (used when --provider ollama)",
    )
    parser.add_argument(
        "--litellm-api-base",
        default="",
        help="LiteLLM API base URL (used when --provider litellm)",
    )
    parser.add_argument(
        "--litellm-api-key",
        default="",
        help="LiteLLM API key (used when --provider litellm)",
    )
    parser.add_argument(
        "--extra-rules-file",
        default="",
        help="Optional text file path with additional prompt rules",
    )
    parser.add_argument(
        "--rules-config-file",
        default=str(DEFAULT_RULES_CONFIG_PATH),
        help="JSON file path for dynamic verb/object/manner extraction rules",
    )
    parser.add_argument(
        "--split-by-verb",
        action="store_true",
        help="If set, returns verb-wise action list instead of single SVO",
    )
    parser.add_argument(
        "--use-instructor",
        action="store_true",
        help="Use Instructor backend (OpenAI provider only) for stronger schema-constrained output",
    )
    parser.add_argument(
        "--use-guardrails",
        action="store_true",
        help="Enable post-extraction quality guardrails (dedupe/normalize/sanity checks)",
    )
    parser.add_argument(
        "--use-spacy-postprocess",
        action="store_true",
        default=None,
        help="Enable spaCy-based rule postprocessing for person/department/action corrections",
    )
    parser.add_argument(
        "--no-spacy-postprocess",
        dest="use_spacy_postprocess",
        action="store_false",
        help="Disable spaCy-based postprocessing",
    )
    parser.add_argument(
        "--spacy-model",
        default="xx_ent_wiki_sm",
        help="spaCy model name used for postprocessing (falls back to blank model if unavailable)",
    )
    args = parser.parse_args()

    if args.split_by_verb:
        result = extract_by_verb(
            args.text,
            model=args.model,
            extra_rules_file=args.extra_rules_file,
            rules_config_file=args.rules_config_file,
            provider=args.provider,
            ollama_base_url=args.ollama_base_url,
            use_instructor=args.use_instructor,
            use_guardrails=args.use_guardrails,
            use_spacy_postprocess=args.use_spacy_postprocess,
            spacy_model=args.spacy_model,
            litellm_api_base=args.litellm_api_base,
            litellm_api_key=args.litellm_api_key,
        )
    else:
        result = extract_one(
            args.text,
            model=args.model,
            extra_rules_file=args.extra_rules_file,
            rules_config_file=args.rules_config_file,
            provider=args.provider,
            ollama_base_url=args.ollama_base_url,
            use_instructor=args.use_instructor,
            use_guardrails=args.use_guardrails,
            use_spacy_postprocess=args.use_spacy_postprocess,
            spacy_model=args.spacy_model,
            litellm_api_base=args.litellm_api_base,
            litellm_api_key=args.litellm_api_key,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
