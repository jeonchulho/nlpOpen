from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    ChatOllama = None


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
        provider: Literal["openai", "ollama"] = "openai",
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        extra_rules: str = "",
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        load_dotenv()
        self.llm = self._build_llm(
            provider=provider,
            model=model,
            temperature=temperature,
            ollama_base_url=ollama_base_url,
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

        self.extract_chain = self.extract_prompt | self.llm.with_structured_output(ExtractionResult)
        self.refine_chain = self.refine_prompt | self.llm.with_structured_output(ExtractionResult)
        self.multi_action_chain = self.multi_action_prompt | self.llm.with_structured_output(
            MultiActionExtractionResult
        )

    @staticmethod
    def _build_llm(
        provider: Literal["openai", "ollama"],
        model: str,
        temperature: float,
        ollama_base_url: str,
    ):
        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError("OPENAI_API_KEY is not set.")
            return ChatOpenAI(model=model, temperature=temperature)

        if provider == "ollama":
            if ChatOllama is None:
                raise ImportError(
                    "langchain-ollama is not installed. Install dependencies with: pip install -r requirements.txt"
                )
            return ChatOllama(model=model, temperature=temperature, base_url=ollama_base_url)

        raise ValueError(f"Unsupported provider: {provider}")

    def extract(self, text: str) -> ExtractionResult:
        first_pass = self.extract_chain.invoke({"text": text})
        final_pass = self.refine_chain.invoke(
            {"text": text, "first_pass_json": first_pass.model_dump_json(indent=2, ensure_ascii=False)}
        )
        return final_pass

    def extract_by_verb(self, text: str) -> MultiActionExtractionResult:
        return self.multi_action_chain.invoke({"text": text})


def _read_rules_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rules file not found: {path}")
    return p.read_text(encoding="utf-8")


def extract_one(
    text: str,
    model: str = "gpt-4.1",
    extra_rules_file: str = "",
    provider: Literal["openai", "ollama"] = "openai",
    ollama_base_url: str = "http://localhost:11434",
) -> dict:
    if not text.strip():
        raise ValueError("text must not be empty")
    extra_rules = _read_rules_file(extra_rules_file) if extra_rules_file else ""
    extractor = MultilingualSVOExtractor(
        provider=provider,
        model=model,
        extra_rules=extra_rules,
        ollama_base_url=ollama_base_url,
    )
    result = extractor.extract(text)
    return result.model_dump()


def extract_many(
    texts: list[str],
    model: str = "gpt-4.1",
    extra_rules_file: str = "",
    provider: Literal["openai", "ollama"] = "openai",
    ollama_base_url: str = "http://localhost:11434",
) -> list[dict]:
    extra_rules = _read_rules_file(extra_rules_file) if extra_rules_file else ""
    extractor = MultilingualSVOExtractor(
        provider=provider,
        model=model,
        extra_rules=extra_rules,
        ollama_base_url=ollama_base_url,
    )
    outputs: list[dict] = []
    for text in texts:
        outputs.append(extractor.extract(text).model_dump())
    return outputs


def extract_by_verb(
    text: str,
    model: str = "gpt-4.1",
    extra_rules_file: str = "",
    provider: Literal["openai", "ollama"] = "openai",
    ollama_base_url: str = "http://localhost:11434",
) -> dict:
    if not text.strip():
        raise ValueError("text must not be empty")
    extra_rules = _read_rules_file(extra_rules_file) if extra_rules_file else ""
    extractor = MultilingualSVOExtractor(
        provider=provider,
        model=model,
        extra_rules=extra_rules,
        ollama_base_url=ollama_base_url,
    )
    result = extractor.extract_by_verb(text)
    return result.model_dump()


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Multilingual SVO + Condition Extractor")
    parser.add_argument("--text", required=True, help="Input sentence")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"], help="LLM provider")
    parser.add_argument("--model", default="gpt-4.1", help="Model name (OpenAI or Ollama local model)")
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama server URL (used when --provider ollama)",
    )
    parser.add_argument(
        "--extra-rules-file",
        default="",
        help="Optional text file path with additional prompt rules",
    )
    parser.add_argument(
        "--split-by-verb",
        action="store_true",
        help="If set, returns verb-wise action list instead of single SVO",
    )
    args = parser.parse_args()

    if args.split_by_verb:
        result = extract_by_verb(
            args.text,
            model=args.model,
            extra_rules_file=args.extra_rules_file,
            provider=args.provider,
            ollama_base_url=args.ollama_base_url,
        )
    else:
        result = extract_one(
            args.text,
            model=args.model,
            extra_rules_file=args.extra_rules_file,
            provider=args.provider,
            ollama_base_url=args.ollama_base_url,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
