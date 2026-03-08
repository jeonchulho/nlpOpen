from __future__ import annotations

import json
from pathlib import Path

BASE_RULES = [
    "주어가 명시되지 않으면 문맥상 요청 주체를 보수적으로 추정한다. 요청/희망 표현은 기본적으로 고객을 우선 검토한다.",
    "동사는 문장 핵심 사건을 나타내는 술어를 우선 추출하고, 보조 용언/완곡 표현보다 핵심 행위를 우선한다.",
    "목적어는 동사의 직접 대상(무엇을/무엇에)을 우선하며, 시간/장소/방법 표현은 목적어에 포함하지 않는다.",
    "조건은 시간/장소/방법/이유/제약으로 분리하고, 접속어(그리고/또는/및/when/if 등)로 연결된 조건은 개별 항목으로 나눈다.",
    "원문에 없는 조건을 생성하지 말고, 근거 없는 확장은 금지한다.",
]

ERROR_RULE_MAP = {
    "subject_mismatch": "주어 역할 판별 시 발화 주체(고객/상담원/시스템/대상 객체)를 구분하고, 질문형 문장에서도 행위 주체를 우선 보존한다.",
    "verb_mismatch": "동사 추출 시 시제/높임/가능 표현보다 핵심 행위 의미를 유지하고, 불필요한 주변 어절을 제거한다.",
    "object_mismatch": "목적어는 행위의 직접 대상 명사구로 한정하고, 도구/경로/시간 표현은 조건으로 이동한다.",
    "condition_missing": "조건 누락 방지를 위해 '~면', '~때', '~까지', '~이후', '~로만' 같은 조건 패턴을 모두 탐지한다.",
    "condition_false_positive": "조건 추출은 문장에 명시된 구절만 사용하고, 암묵 조건 추론은 하지 않는다.",
    "condition_type_mismatch": "조건 타입 매핑을 엄격히 적용한다: 기간/시점=time, 장소=location, 수단/형식=manner, 원인=reason, 제한/자격=constraint.",
}

TYPE_RULE_MAP = {
    "time": "시간 조건은 시점/기간/기한 표현(예: 오늘, 내일, 3시까지, 이후)을 빠짐없이 분리한다.",
    "location": "장소 조건은 지역/채널/플랫폼(앱/웹/매장 등) 표현을 location으로 분류한다.",
    "manner": "방법 조건은 수단/형식/전달 방식(문자, 이메일, PDF 등)을 manner로 분류한다.",
    "reason": "이유 조건은 '~때문에', '~라서' 등 인과 표현을 reason으로 분류한다.",
    "constraint": "제약 조건은 only/만/한해서/이상/이내/자격 조건을 constraint로 분류한다.",
    "other": "분류가 애매한 조건만 other로 두고, 가능한 기존 타입으로 우선 매핑한다.",
}


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"analysis report not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _top_keys(counter: dict, threshold: int) -> list[str]:
    picked = []
    for key, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if count >= threshold:
            picked.append(key)
    return picked


def build_tuned_rules(report: dict, min_count: int = 2) -> list[str]:
    summary = report.get("summary", {})
    error_counts = summary.get("error_type_counts", {})
    miss_by_type = summary.get("condition_missing_by_type", {})
    fp_by_type = summary.get("condition_false_positive_by_type", {})

    rules = list(BASE_RULES)

    for error_key in _top_keys(error_counts, min_count):
        rule = ERROR_RULE_MAP.get(error_key)
        if rule:
            rules.append(rule)

    for cond_type in _top_keys(miss_by_type, min_count):
        rule = TYPE_RULE_MAP.get(cond_type)
        if rule:
            rules.append(rule)

    for cond_type in _top_keys(fp_by_type, min_count):
        rule = TYPE_RULE_MAP.get(cond_type)
        if rule and rule not in rules:
            rules.append(rule)

    seen = set()
    deduped = []
    for r in rules:
        if r not in seen:
            deduped.append(r)
            seen.add(r)
    return deduped


def _to_markdown(report: dict, rules: list[str], min_count: int) -> str:
    summary = report.get("summary", {})
    lines = []
    lines.append("# Prompt Tuning Report")
    lines.append("")
    lines.append("## Source Summary")
    lines.append(f"- total: {summary.get('count', 0)}")
    lines.append(f"- failed: {summary.get('failed_count', 0)}")
    lines.append(f"- failure_rate: {summary.get('failure_rate', 0)}")
    lines.append(f"- min_count: {min_count}")
    lines.append("")

    lines.append("## Selected Error Drivers")
    error_counts = summary.get("error_type_counts", {})
    if error_counts:
        for key, value in sorted(error_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Tuned Rules")
    for idx, rule in enumerate(rules, start=1):
        lines.append(f"{idx}. {rule}")

    return "\n".join(lines)


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate tuned prompt rules from error analysis")
    parser.add_argument(
        "--analysis-json",
        default="reports/error_analysis.json",
        help="Path to error analysis JSON report",
    )
    parser.add_argument(
        "--out-rules",
        default="prompts/tuned_rules.txt",
        help="Output rules text file",
    )
    parser.add_argument(
        "--out-md",
        default="reports/prompt_tuning.md",
        help="Output markdown tuning report",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum error count to include targeted tuning rules",
    )
    args = parser.parse_args()

    report = _load_json(args.analysis_json)
    rules = build_tuned_rules(report, min_count=args.min_count)

    out_rules = Path(args.out_rules)
    out_rules.parent.mkdir(parents=True, exist_ok=True)
    out_rules.write_text("\n".join(f"- {r}" for r in rules) + "\n", encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(report, rules, args.min_count), encoding="utf-8")

    print(f"Saved: {out_rules}")
    print(f"Saved: {out_md}")
    print(json.dumps({"rules_count": len(rules), "min_count": args.min_count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
