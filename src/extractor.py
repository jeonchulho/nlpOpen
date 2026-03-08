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
    from extractor_defaults import ACTION_SCOPE_CLEANUP, ACTION_SCOPE_PATTERNS_BY_LANG, ACTION_SCOPE_VERB_HINTS, ADDITIONAL_CONDITION_PATTERNS_BY_LANG, DEFAULT_OBJECT_BY_LANG, DEFAULT_VERB_BY_LANG, DEPARTMENT_SUFFIXES, GENERIC_PERSON_STOPWORDS, KOREAN_PERSON_STOPWORDS, MANNER_TOKENS_BY_LANG, NON_PERSON_TOKENS_BY_LANG, OBJECT_ACTION_MODIFIER_ACTION_TOKENS_BY_LANG, OBJECT_ACTION_MODIFIER_ATTRIBUTE_SUFFIXES_BY_LANG, OBJECT_ACTION_MODIFIER_PATTERNS_BY_LANG, OBJECT_ACTION_MODIFIER_SKIP_TOKENS_BY_LANG, OBJECT_DETAIL_RULES_BY_LANG, OBJECT_PHRASE_FALLBACK_BY_LANG, OBJECT_RULES_BY_LANG, SPACY_ONLY_CONFIDENCE, SPACY_ONLY_MULTI_CONFIDENCE, SPACY_ONLY_SUBJECT_BY_LANG, SPACY_REFINE_ACTION_EXCLUDED_TYPES, SPACY_SIGNAL_DEPARTMENT_SUFFIX_PATTERN_JA, SPACY_SIGNAL_DEPT_PATTERN, SPACY_SIGNAL_JA_PERSON_PREFIXES, SPACY_SIGNAL_LATIN_TIME_SUFFIX_PATTERN, SPACY_SIGNAL_PATTERNS_BY_LANG, SPACY_SIGNAL_PERSON_TIME_SUFFIX_PATTERN, SUBJECT_PICKER_RULES_BY_LANG, SUMMARIZE_HINT_KEYWORDS, SUMMARIZE_OBJECT_RULES_BY_LANG, SUMMARIZE_VERB_BY_LANG, SUPPORTED_LANGUAGES, TIME_CASE_INSENSITIVE_LANGS, TIME_PATTERNS_BY_LANG, TIME_PATTERNS_COMMON, VERB_PATTERNS_BY_LANG
except ImportError:  # pragma: no cover
    from src.extractor_llm_backend import build_llm as llm_build_backend, extract_multi as llm_extract_multi_backend, extract_single as llm_extract_single_backend
    from src.extractor_spacy_backend import build_spacy_multi_payload, build_spacy_nlp as spacy_build_backend, build_spacy_single_payload, extract_action_scoped_entities as spacy_extract_action_scoped_entities_backend, extract_multi as spacy_extract_multi_backend, extract_object_action_modifiers as spacy_extract_object_action_modifiers_backend, extract_single as spacy_extract_single_backend, extract_spacy_signals as spacy_extract_signals_backend, spacy_refine_conditions as spacy_refine_conditions_backend
    from src.extractor_common import detect_language as common_detect_language, pick_lang_value as common_pick_lang_value, pick_lang_value_by_text as common_pick_lang_value_by_text, pick_main_verb as common_pick_main_verb
    from src.extractor_defaults import ACTION_SCOPE_CLEANUP, ACTION_SCOPE_PATTERNS_BY_LANG, ACTION_SCOPE_VERB_HINTS, ADDITIONAL_CONDITION_PATTERNS_BY_LANG, DEFAULT_OBJECT_BY_LANG, DEFAULT_VERB_BY_LANG, DEPARTMENT_SUFFIXES, GENERIC_PERSON_STOPWORDS, KOREAN_PERSON_STOPWORDS, MANNER_TOKENS_BY_LANG, NON_PERSON_TOKENS_BY_LANG, OBJECT_ACTION_MODIFIER_ACTION_TOKENS_BY_LANG, OBJECT_ACTION_MODIFIER_ATTRIBUTE_SUFFIXES_BY_LANG, OBJECT_ACTION_MODIFIER_PATTERNS_BY_LANG, OBJECT_ACTION_MODIFIER_SKIP_TOKENS_BY_LANG, OBJECT_DETAIL_RULES_BY_LANG, OBJECT_PHRASE_FALLBACK_BY_LANG, OBJECT_RULES_BY_LANG, SPACY_ONLY_CONFIDENCE, SPACY_ONLY_MULTI_CONFIDENCE, SPACY_ONLY_SUBJECT_BY_LANG, SPACY_REFINE_ACTION_EXCLUDED_TYPES, SPACY_SIGNAL_DEPARTMENT_SUFFIX_PATTERN_JA, SPACY_SIGNAL_DEPT_PATTERN, SPACY_SIGNAL_JA_PERSON_PREFIXES, SPACY_SIGNAL_LATIN_TIME_SUFFIX_PATTERN, SPACY_SIGNAL_PATTERNS_BY_LANG, SPACY_SIGNAL_PERSON_TIME_SUFFIX_PATTERN, SUBJECT_PICKER_RULES_BY_LANG, SUMMARIZE_HINT_KEYWORDS, SUMMARIZE_OBJECT_RULES_BY_LANG, SUMMARIZE_VERB_BY_LANG, SUPPORTED_LANGUAGES, TIME_CASE_INSENSITIVE_LANGS, TIME_PATTERNS_BY_LANG, TIME_PATTERNS_COMMON, VERB_PATTERNS_BY_LANG

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
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
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
4) 사람명, 조직명(팀/부/본부 등), 보조 행위(정리/검토/승인 등)는 조건에 각각 person/department/action으로 분류한다.
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
    -> subject="고객", verb="보내줘", object="쪽지 정리본", conditions=[(type=person, text=전철호), (type=person, text=이선정), (type=department, text=영업팀), (type=action, text=정리해서)]

[입력 문장]
{text}
""".strip()


REFINE_SYSTEM_PROMPT = """
너는 정보추출 결과를 교정하는 검수기다.

규칙:
1) 원문과 1차 추출 결과를 비교해 누락/오추출을 교정한다.
2) subject/verb/object의 역할이 틀리면 바로잡는다.
3) 조건정보를 가능한 한 세분화해서 보완한다.
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
4) 원문에 없는 동작을 생성하지 않는다.
5) 출력은 반드시 지정된 JSON 스키마를 따른다.
""".strip()


MULTI_ACTION_HUMAN_PROMPT = """
[입력 문장]
{text}
""".strip()


class MultilingualSVOExtractor:
    """Two-pass extractor for high precision SVO + condition extraction."""

    def __init__(
        self,
        provider: Literal["openai", "ollama", "litellm", "spacy"] = "openai",
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        extra_rules: str = "",
        ollama_base_url: str = "http://localhost:11434",
        use_instructor: bool = False,
        use_guardrails: bool = False,
        use_spacy_postprocess: bool = False,
        spacy_model: str = "xx_ent_wiki_sm",
        litellm_api_base: str = "",
        litellm_api_key: str = "",
        rules_config_file: str = "",
    ) -> None:
        load_dotenv()
        self.provider = provider
        self.model = model
        self.use_instructor = use_instructor
        self.use_guardrails = use_guardrails
        self.use_spacy_postprocess = use_spacy_postprocess
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
        if self.provider == "spacy":
            return spacy_extract_single_backend(self, text)
        return llm_extract_single_backend(self, text)

    def extract_by_verb(self, text: str) -> MultiActionExtractionResult:
        if self.provider == "spacy":
            return spacy_extract_multi_backend(self, text)
        return llm_extract_multi_backend(self, text)

    def _extract_spacy_only(self, text: str) -> ExtractionResult:
        payload = build_spacy_single_payload(self, text)
        return ExtractionResult(**payload)

    def _extract_by_verb_spacy_only(self, text: str) -> MultiActionExtractionResult:
        payload = build_spacy_multi_payload(self, text)
        actions = [VerbAction(**a) for a in payload.get("actions", [])]
        return MultiActionExtractionResult(
            language=str(payload.get("language", "en")),
            actions=actions,
            confidence=float(payload.get("confidence", self.spacy_only_multi_confidence)),
        )

    def _pick_summarize_object(self, text: str, lang: str) -> str:
        detailed = self._pick_detailed_object(text, lang)
        if detailed:
            return detailed
        for pattern, label in self.summarize_object_rules_by_lang.get(lang, []):
            flags = re.I if re.search(r"[A-Za-z]", pattern) else 0
            if re.search(pattern, text, flags=flags):
                return label
        return self._pick_object(text, lang)

    def _get_summarize_hint_keywords(self, lang: str) -> list[str]:
        merged: list[str] = []
        for key in ("default", "all", "*", lang):
            for kw in self.summarize_hint_keywords_by_lang.get(key, []):
                if kw not in merged:
                    merged.append(kw)
        return merged

    @staticmethod
    def _detect_language(text: str) -> str:
        return common_detect_language(text)

    def _pick_main_verb(self, text: str, lang: str = "ko") -> str:
        return common_pick_main_verb(
            text=text,
            lang=lang,
            verb_patterns_by_lang=self.verb_patterns_by_lang,
            default_verb_candidates_by_lang=self.default_verb_candidates_by_lang,
        )

    def _pick_object(self, text: str, lang: str = "ko") -> str:
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

        return self._pick_lang_value(self.default_object_candidates_by_lang, lang, "en", "request target")

    def _pick_detailed_object(self, text: str, lang: str) -> str:
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
        fallback = self._pick_lang_value(self.subject_by_lang, lang, "en", "customer")
        rules = self.subject_picker_rules_by_lang.get(lang)
        if not isinstance(rules, dict):
            return fallback

        topic_pattern = str(rules.get("topic_pattern", "")).strip()
        topic_group = int(rules.get("topic_group", 1) or 1)
        topic_exclude = {str(x).strip().casefold() for x in rules.get("topic_exclude", []) if str(x).strip()}
        predicate_verbs = [str(x).strip() for x in rules.get("predicate_verb_keywords", []) if str(x).strip()]
        predicate_patterns = [str(x).strip() for x in rules.get("predicate_verb_patterns", []) if str(x).strip()]
        object_skip_values = {
            str(x).strip().casefold() for x in rules.get("object_skip_values", []) if str(x).strip()
        }
        suffix_overrides_raw = rules.get("object_suffix_subject_overrides", {})
        suffix_overrides = dict(suffix_overrides_raw) if isinstance(suffix_overrides_raw, dict) else {}
        allow_subject_from_object = _as_bool(rules.get("allow_subject_from_object", True), True)

        if topic_pattern:
            try:
                topic_matches = list(re.finditer(topic_pattern, text))
                if topic_matches:
                    matched = topic_matches[-1]
                    if matched.lastindex and matched.lastindex >= topic_group:
                        cand = matched.group(topic_group).strip()
                    else:
                        cand = matched.group(0).strip()
                    if cand and cand.casefold() not in topic_exclude:
                        return cand
            except re.error:
                # Ignore broken custom regex and keep fallback path alive.
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
        conds: list[dict] = []
        patterns = [*self.time_patterns_common, *self.time_patterns_by_lang.get(lang, [])]
        flags = re.I if lang in self.time_case_insensitive_langs else 0
        for p in patterns:
            for m in re.finditer(p, text, flags=flags):
                conds.append({"type": "time", "text": m.group(0).strip()})

        # Keep longer time spans and drop shorter tokens subsumed by them
        # (e.g., keep "월,화요일" and remove "월", "화요일").
        deduped = self._dedupe_conditions(conds)
        texts = [str(c.get("text", "")).strip() for c in deduped]
        ordered = sorted(texts, key=len, reverse=True)

        def _normalize_for_compare(value: str) -> str:
            # Normalize spacing/punctuation differences to collapse equivalent time ranges.
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
        conds: list[dict] = []
        tokens = self.manner_tokens_by_lang.get(lang, [])
        if lang != "en":
            tokens = [*tokens, *self.manner_tokens_by_lang.get("en", [])]
        for token in tokens:
            if token in text:
                conds.append({"type": "manner", "text": token})

        # Treat summarize intent as a manner-like condition for request decomposition.
        lower_text = text.lower()
        summarize_hints = self._get_summarize_hint_keywords(lang)
        for hint in summarize_hints:
            if hint and hint.lower() in lower_text:
                conds.append({"type": "manner", "text": hint})

        return conds

    def _extract_additional_conditions(self, text: str, lang: str = "ko") -> list[dict]:
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
        return spacy_build_backend(model_name=model_name, spacy_module=spacy)

    def _apply_spacy_postprocess_single(self, result: ExtractionResult, text: str) -> ExtractionResult:
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
        return spacy_extract_action_scoped_entities_backend(self, text, verb_text)

    def _extract_object_action_modifiers(self, object_text: str) -> list[str]:
        return spacy_extract_object_action_modifiers_backend(self, object_text)

    def _extract_spacy_signals(self, text: str) -> dict[str, set[str]]:
        return spacy_extract_signals_backend(
            extractor=self,
            text=text,
            korean_person_stopwords=KOREAN_PERSON_STOPWORDS,
            generic_person_stopwords=GENERIC_PERSON_STOPWORDS,
            department_suffixes=DEPARTMENT_SUFFIXES,
        )

    def _apply_guardrails_single(self, result: ExtractionResult) -> ExtractionResult:
        if not self.use_guardrails:
            return result

        data = result.model_dump()
        data["language"] = self._normalize_language(data.get("language", ""))
        data["subject"] = data.get("subject", "").strip()
        data["verb"] = data.get("verb", "").strip()
        data["object"] = data.get("object", "").strip()
        data["conditions"] = self._dedupe_conditions(data.get("conditions", []))
        data["confidence"] = self._clamp_confidence(data.get("confidence", 0.0))

        # Guardrail: if any core field becomes empty, keep original output instead of emitting invalid structure.
        if not data["subject"] or not data["verb"] or not data["object"]:
            return result

        return ExtractionResult(**data)

    def _apply_guardrails_multi(self, result: MultiActionExtractionResult) -> MultiActionExtractionResult:
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
        lang = (language or "").strip().lower()
        return lang if lang in SUPPORTED_LANGUAGES else "en"

    @staticmethod
    def _clamp_confidence(confidence: float) -> float:
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _dedupe_conditions(conditions: list[dict]) -> list[dict]:
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
    use_spacy_postprocess: bool = False,
    spacy_model: str = "xx_ent_wiki_sm",
    litellm_api_base: str = "",
    litellm_api_key: str = "",
) -> dict:
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
    use_spacy_postprocess: bool = False,
    spacy_model: str = "xx_ent_wiki_sm",
    litellm_api_base: str = "",
    litellm_api_key: str = "",
) -> list[dict]:
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
    use_spacy_postprocess: bool = False,
    spacy_model: str = "xx_ent_wiki_sm",
    litellm_api_base: str = "",
    litellm_api_key: str = "",
) -> dict:
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
        help="Enable spaCy-based rule postprocessing for person/department/action corrections",
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
