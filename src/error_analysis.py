from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

try:
    from extractor import extract_one
except ImportError:  # pragma: no cover
    from src.extractor import extract_one


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _condition_text_set(conditions: list[dict]) -> set[str]:
    return {_norm(c.get("text", "")) for c in conditions if c.get("text", "").strip()}


def analyze_errors(
    golden_path: str,
    model: str = "gpt-4.1",
    provider: str = "openai",
    ollama_base_url: str = "http://localhost:11434",
    use_instructor: bool = False,
) -> dict:
    rows = []
    with Path(golden_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    mismatch_counter: Counter[str] = Counter()
    condition_type_miss_counter: Counter[str] = Counter()
    condition_type_false_positive_counter: Counter[str] = Counter()
    failures: list[dict] = []

    for row in rows:
        text = row["text"]
        gold = row["gold"]
        pred = extract_one(
            text,
            model=model,
            provider=provider,
            ollama_base_url=ollama_base_url,
            use_instructor=use_instructor,
        )

        sample_errors: list[str] = []

        if _norm(pred["subject"]) != _norm(gold["subject"]):
            mismatch_counter["subject_mismatch"] += 1
            sample_errors.append("subject_mismatch")

        if _norm(pred["verb"]) != _norm(gold["verb"]):
            mismatch_counter["verb_mismatch"] += 1
            sample_errors.append("verb_mismatch")

        if _norm(pred["object"]) != _norm(gold["object"]):
            mismatch_counter["object_mismatch"] += 1
            sample_errors.append("object_mismatch")

        pred_conditions = pred.get("conditions", [])
        gold_conditions = gold.get("conditions", [])

        pred_texts = _condition_text_set(pred_conditions)
        gold_texts = _condition_text_set(gold_conditions)

        missed_texts = sorted(gold_texts - pred_texts)
        extra_texts = sorted(pred_texts - gold_texts)

        if missed_texts:
            mismatch_counter["condition_missing"] += 1
            sample_errors.append("condition_missing")
        if extra_texts:
            mismatch_counter["condition_false_positive"] += 1
            sample_errors.append("condition_false_positive")

        gold_type_by_text = {_norm(c.get("text", "")): c.get("type", "other") for c in gold_conditions}
        pred_type_by_text = {_norm(c.get("text", "")): c.get("type", "other") for c in pred_conditions}

        for t in missed_texts:
            condition_type_miss_counter[gold_type_by_text.get(t, "other")] += 1
        for t in extra_texts:
            condition_type_false_positive_counter[pred_type_by_text.get(t, "other")] += 1

        shared = pred_texts.intersection(gold_texts)
        for t in shared:
            if pred_type_by_text.get(t, "other") != gold_type_by_text.get(t, "other"):
                mismatch_counter["condition_type_mismatch"] += 1
                sample_errors.append("condition_type_mismatch")
                break

        if sample_errors:
            failures.append(
                {
                    "id": row.get("id"),
                    "text": text,
                    "errors": sorted(set(sample_errors)),
                    "gold": gold,
                    "pred": pred,
                    "missed_condition_texts": missed_texts,
                    "extra_condition_texts": extra_texts,
                }
            )

    total = len(rows)
    summary = {
        "count": total,
        "failed_count": len(failures),
        "pass_count": total - len(failures),
        "failure_rate": round(len(failures) / max(1, total), 4),
        "error_type_counts": dict(mismatch_counter),
        "condition_missing_by_type": dict(condition_type_miss_counter),
        "condition_false_positive_by_type": dict(condition_type_false_positive_counter),
    }

    return {"summary": summary, "failures": failures}


def _to_markdown(report: dict) -> str:
    s = report["summary"]
    lines = []
    lines.append("# Error Analysis Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- total: {s['count']}")
    lines.append(f"- failed: {s['failed_count']}")
    lines.append(f"- passed: {s['pass_count']}")
    lines.append(f"- failure_rate: {s['failure_rate']}")
    lines.append("")

    lines.append("## Error Type Counts")
    if s["error_type_counts"]:
        for k, v in sorted(s["error_type_counts"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Condition Missing By Type")
    if s["condition_missing_by_type"]:
        for k, v in sorted(s["condition_missing_by_type"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Condition False Positive By Type")
    if s["condition_false_positive_by_type"]:
        for k, v in sorted(
            s["condition_false_positive_by_type"].items(), key=lambda x: (-x[1], x[0])
        ):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Top Failures")
    if not report["failures"]:
        lines.append("- no failures")
    else:
        top = report["failures"][:10]
        for item in top:
            lines.append(f"- id={item.get('id')}, errors={','.join(item['errors'])}, text={item['text']}")

    return "\n".join(lines)


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Error analysis for SVO extraction")
    parser.add_argument("--golden", default="data/golden_set_50.jsonl", help="Path to JSONL golden set")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"], help="LLM provider")
    parser.add_argument("--model", default="gpt-4.1", help="Model name (OpenAI or Ollama local model)")
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama server URL (used when --provider ollama)",
    )
    parser.add_argument(
        "--out-json",
        default="reports/error_analysis.json",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--out-md",
        default="reports/error_analysis.md",
        help="Output path for Markdown report",
    )
    parser.add_argument(
        "--use-instructor",
        action="store_true",
        help="Use Instructor backend (OpenAI provider only)",
    )
    args = parser.parse_args()

    report = analyze_errors(
        args.golden,
        model=args.model,
        provider=args.provider,
        ollama_base_url=args.ollama_base_url,
        use_instructor=args.use_instructor,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(report), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    cli()
