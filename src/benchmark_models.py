from __future__ import annotations

import json
from pathlib import Path

try:
    from evaluate import evaluate, evaluate_split_by_verb
except ImportError:  # pragma: no cover
    from src.evaluate import evaluate, evaluate_split_by_verb


def run_benchmark(
    models: list[str],
    provider: str,
    golden: str,
    split_by_verb: bool,
    extra_rules_file: str,
    ollama_base_url: str,
    use_instructor: bool,
    use_guardrails: bool,
    use_spacy_postprocess: bool,
    spacy_model: str,
    litellm_api_base: str,
    litellm_api_key: str,
) -> list[dict]:
    rows: list[dict] = []
    for model in models:
        if split_by_verb:
            report = evaluate_split_by_verb(
                golden_path=golden,
                model=model,
                extra_rules_file=extra_rules_file,
                provider=provider,
                ollama_base_url=ollama_base_url,
                use_instructor=use_instructor,
                use_guardrails=use_guardrails,
                use_spacy_postprocess=use_spacy_postprocess,
                spacy_model=spacy_model,
                litellm_api_base=litellm_api_base,
                litellm_api_key=litellm_api_key,
            )
        else:
            report = evaluate(
                golden_path=golden,
                model=model,
                extra_rules_file=extra_rules_file,
                provider=provider,
                ollama_base_url=ollama_base_url,
                use_instructor=use_instructor,
                use_guardrails=use_guardrails,
                use_spacy_postprocess=use_spacy_postprocess,
                spacy_model=spacy_model,
                litellm_api_base=litellm_api_base,
                litellm_api_key=litellm_api_key,
            )

        rows.append(
            {
                "provider": provider,
                "model": model,
                "use_instructor": use_instructor,
                "use_guardrails": use_guardrails,
                "use_spacy_postprocess": use_spacy_postprocess,
                "golden": golden,
                "split_by_verb": split_by_verb,
                "svo_macro_exact": report.get("svo_macro_exact"),
                "condition_recall": report.get("condition_recall"),
                "action_f1": report.get("action_f1"),
                "action_count_exact": report.get("action_count_exact"),
                "raw": report,
            }
        )
    return rows


def _to_markdown(rows: list[dict]) -> str:
    lines = []
    lines.append("# Model Benchmark")
    lines.append("")
    lines.append("| provider | model | split_by_verb | svo_macro_exact | condition_recall | action_f1 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {provider} | {model} | {split} | {svo} | {cond} | {f1} |".format(
                provider=r["provider"],
                model=r["model"],
                split=str(r["split_by_verb"]),
                svo=str(r.get("svo_macro_exact", "")),
                cond=str(r.get("condition_recall", "")),
                f1=str(r.get("action_f1", "")),
            )
        )
    return "\n".join(lines)


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark multiple models on the same golden set")
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["openai", "ollama", "litellm"],
        help="LLM provider",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model names, e.g. qwen2.5:14b,qwen2.5:32b",
    )
    parser.add_argument("--golden", default="data/golden_set_verb_12.jsonl", help="Path to golden JSONL")
    parser.add_argument("--split-by-verb", action="store_true", help="Use split-by-verb evaluation")
    parser.add_argument("--extra-rules-file", default="", help="Optional tuned rules file path")
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama server URL (used when --provider ollama)",
    )
    parser.add_argument("--litellm-api-base", default="", help="LiteLLM API base URL")
    parser.add_argument("--litellm-api-key", default="", help="LiteLLM API key")
    parser.add_argument(
        "--use-instructor",
        action="store_true",
        help="Use Instructor backend (OpenAI provider only)",
    )
    parser.add_argument("--use-guardrails", action="store_true", help="Enable post-extraction guardrails")
    parser.add_argument("--use-spacy-postprocess", action="store_true", help="Enable spaCy rule postprocess")
    parser.add_argument("--spacy-model", default="xx_ent_wiki_sm", help="spaCy model for postprocess")
    parser.add_argument(
        "--out-json",
        default="reports/benchmark_models.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--out-md",
        default="reports/benchmark_models.md",
        help="Output markdown summary path",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("--models must include at least one model")

    rows = run_benchmark(
        models=models,
        provider=args.provider,
        golden=args.golden,
        split_by_verb=args.split_by_verb,
        extra_rules_file=args.extra_rules_file,
        ollama_base_url=args.ollama_base_url,
        use_instructor=args.use_instructor,
        use_guardrails=args.use_guardrails,
        use_spacy_postprocess=args.use_spacy_postprocess,
        spacy_model=args.spacy_model,
        litellm_api_base=args.litellm_api_base,
        litellm_api_key=args.litellm_api_key,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(rows), encoding="utf-8")

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    cli()
