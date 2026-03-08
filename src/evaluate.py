from __future__ import annotations

import json
from pathlib import Path

try:
    from extractor import extract_by_verb, extract_one
except ImportError:  # pragma: no cover
    from src.extractor import extract_by_verb, extract_one


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _field_exact(pred: str, gold: str) -> int:
    return int(_norm(pred) == _norm(gold))


def _condition_recall(pred_conditions: list[dict], gold_conditions: list[dict]) -> float:
    if not gold_conditions:
        return 1.0
    pred_texts = {_norm(c.get("text", "")) for c in pred_conditions}
    gold_texts = {_norm(c.get("text", "")) for c in gold_conditions}
    hit = len(pred_texts.intersection(gold_texts))
    return hit / max(1, len(gold_texts))


def _action_match_score(pred: dict, gold: dict) -> int:
    score = 0
    score += _field_exact(pred.get("subject", ""), gold.get("subject", ""))
    score += _field_exact(pred.get("verb", ""), gold.get("verb", ""))
    score += _field_exact(pred.get("object", ""), gold.get("object", ""))
    return score


def _to_gold_actions(row: dict) -> list[dict]:
    if "gold_actions" in row and isinstance(row["gold_actions"], list):
        return row["gold_actions"]
    return [row["gold"]]


def evaluate(
    golden_path: str,
    model: str = "gpt-4.1",
    extra_rules_file: str = "",
    provider: str = "openai",
    ollama_base_url: str = "http://localhost:11434",
) -> dict:
    rows = []
    with Path(golden_path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    subject_hits = 0
    verb_hits = 0
    object_hits = 0
    cond_scores = []

    for row in rows:
        pred = extract_one(
            row["text"],
            model=model,
            extra_rules_file=extra_rules_file,
            provider=provider,
            ollama_base_url=ollama_base_url,
        )
        gold = row["gold"]

        subject_hits += _field_exact(pred["subject"], gold["subject"])
        verb_hits += _field_exact(pred["verb"], gold["verb"])
        object_hits += _field_exact(pred["object"], gold["object"])
        cond_scores.append(_condition_recall(pred.get("conditions", []), gold.get("conditions", [])))

    n = len(rows)
    result = {
        "count": n,
        "subject_exact": round(subject_hits / n, 4),
        "verb_exact": round(verb_hits / n, 4),
        "object_exact": round(object_hits / n, 4),
        "condition_recall": round(sum(cond_scores) / n, 4),
    }
    result["svo_macro_exact"] = round(
        (result["subject_exact"] + result["verb_exact"] + result["object_exact"]) / 3,
        4,
    )
    return result


def evaluate_split_by_verb(
    golden_path: str,
    model: str = "gpt-4.1",
    extra_rules_file: str = "",
    provider: str = "openai",
    ollama_base_url: str = "http://localhost:11434",
) -> dict:
    rows = []
    with Path(golden_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    subject_hits = 0
    verb_hits = 0
    object_hits = 0
    cond_scores = []

    matched_gold_actions = 0
    total_gold_actions = 0
    total_pred_actions = 0
    action_count_exact_hits = 0

    for row in rows:
        pred_multi = extract_by_verb(
            row["text"],
            model=model,
            extra_rules_file=extra_rules_file,
            provider=provider,
            ollama_base_url=ollama_base_url,
        )
        pred_actions = pred_multi.get("actions", [])
        gold_actions = _to_gold_actions(row)

        total_gold_actions += len(gold_actions)
        total_pred_actions += len(pred_actions)
        action_count_exact_hits += int(len(pred_actions) == len(gold_actions))

        used_pred = set()
        for gold in gold_actions:
            best_idx = -1
            best_score = -1
            for i, pred in enumerate(pred_actions):
                if i in used_pred:
                    continue
                score = _action_match_score(pred, gold)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1:
                continue

            used_pred.add(best_idx)
            pred = pred_actions[best_idx]

            s_hit = _field_exact(pred.get("subject", ""), gold.get("subject", ""))
            v_hit = _field_exact(pred.get("verb", ""), gold.get("verb", ""))
            o_hit = _field_exact(pred.get("object", ""), gold.get("object", ""))

            subject_hits += s_hit
            verb_hits += v_hit
            object_hits += o_hit
            cond_scores.append(_condition_recall(pred.get("conditions", []), gold.get("conditions", [])))

            if s_hit and v_hit and o_hit:
                matched_gold_actions += 1

    n_actions = max(1, total_gold_actions)
    action_precision = matched_gold_actions / max(1, total_pred_actions)
    action_recall = matched_gold_actions / max(1, total_gold_actions)
    action_f1 = 0.0
    if action_precision + action_recall > 0:
        action_f1 = 2 * action_precision * action_recall / (action_precision + action_recall)

    result = {
        "count": len(rows),
        "mode": "split_by_verb",
        "total_gold_actions": total_gold_actions,
        "total_pred_actions": total_pred_actions,
        "action_count_exact": round(action_count_exact_hits / max(1, len(rows)), 4),
        "subject_exact": round(subject_hits / n_actions, 4),
        "verb_exact": round(verb_hits / n_actions, 4),
        "object_exact": round(object_hits / n_actions, 4),
        "condition_recall": round((sum(cond_scores) / max(1, len(cond_scores))), 4),
        "action_precision": round(action_precision, 4),
        "action_recall": round(action_recall, 4),
        "action_f1": round(action_f1, 4),
    }
    result["svo_macro_exact"] = round(
        (result["subject_exact"] + result["verb_exact"] + result["object_exact"]) / 3,
        4,
    )
    return result


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate extraction on golden set")
    parser.add_argument(
        "--golden",
        default="data/golden_set_50.jsonl",
        help="Path to JSONL golden set",
    )
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
        help="Evaluate using verb-wise action splitting output",
    )
    args = parser.parse_args()

    if args.split_by_verb:
        report = evaluate_split_by_verb(
            args.golden,
            model=args.model,
            extra_rules_file=args.extra_rules_file,
            provider=args.provider,
            ollama_base_url=args.ollama_base_url,
        )
    else:
        report = evaluate(
            args.golden,
            model=args.model,
            extra_rules_file=args.extra_rules_file,
            provider=args.provider,
            ollama_base_url=args.ollama_base_url,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
