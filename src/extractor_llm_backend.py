from __future__ import annotations

import os
from typing import Any


def build_llm(
    provider: str,
    model: str,
    temperature: float,
    ollama_base_url: str,
    litellm_api_base: str,
    litellm_api_key: str,
    chat_openai_cls: Any,
    chat_ollama_cls: Any,
    chat_litellm_cls: Any,
):
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        return chat_openai_cls(model=model, temperature=temperature)

    if provider == "ollama":
        if chat_ollama_cls is None:
            raise ImportError(
                "langchain-ollama is not installed. Install dependencies with: pip install -r requirements.txt"
            )
        return chat_ollama_cls(model=model, temperature=temperature, base_url=ollama_base_url)

    if provider == "litellm":
        if chat_litellm_cls is None:
            raise ImportError(
                "langchain-community/litellm is not installed. Install dependencies with: pip install -r requirements.txt"
            )
        kwargs = {"model": model, "temperature": temperature}
        if litellm_api_base:
            kwargs["api_base"] = litellm_api_base
        if litellm_api_key:
            kwargs["api_key"] = litellm_api_key
        return chat_litellm_cls(**kwargs)

    if provider == "spacy":
        return None

    raise ValueError(f"Unsupported provider: {provider}")


def extract_single(extractor: Any, text: str):
    if extractor.use_instructor:
        result = extractor._extract_with_instructor(text)
        result = extractor._apply_spacy_postprocess_single(result, text)
        return extractor._apply_guardrails_single(result)

    assert extractor.extract_chain is not None and extractor.refine_chain is not None
    first_pass = extractor.extract_chain.invoke({"text": text})
    final_pass = extractor.refine_chain.invoke(
        {"text": text, "first_pass_json": first_pass.model_dump_json(indent=2, ensure_ascii=False)}
    )
    final_pass = extractor._apply_spacy_postprocess_single(final_pass, text)
    return extractor._apply_guardrails_single(final_pass)


def extract_multi(extractor: Any, text: str):
    if extractor.use_instructor:
        result = extractor._extract_by_verb_with_instructor(text)
        result = extractor._apply_spacy_postprocess_multi(result, text)
        return extractor._apply_guardrails_multi(result)

    assert extractor.multi_action_chain is not None
    result = extractor.multi_action_chain.invoke({"text": text})
    result = extractor._apply_spacy_postprocess_multi(result, text)
    return extractor._apply_guardrails_multi(result)
