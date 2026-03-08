# nlpOpen

LangChain `ChatOpenAI` 기반 다국어 정보추출 프로젝트입니다.

요구사항 반영:
- 정보추출: `주어(subject)`, `동사(verb)`, `목적어(object)`, `조건정보(conditions)`
- 언어 지원: 한국어/영어/일어/중국어/아랍어/독어/프랑스어
- 입력 중심: 한국어 고객문의 텍스트
- 정답 데이터(골든셋): 50건 포함

조건 타입:
- `time`, `location`, `manner`, `reason`, `constraint`
- `action`(보조 행위), `person`(사람), `department`(부서), `other`

## 1) 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 환경변수

`OPENAI_API_KEY`를 설정하세요.

```bash
export OPENAI_API_KEY="your_api_key"
```

로컬 LLM(Ollama) 사용 시에는 OpenAI 키가 필요 없습니다.

## 3) 단건 추출

```bash
python src/extractor.py --text "내일 오후 3시까지 주문 취소해 주세요." --model gpt-4.1
```

OpenAI + Instructor 추출(스키마 강제 강화):

```bash
python src/extractor.py \
	--provider openai \
	--model gpt-4.1 \
	--use-instructor \
	--use-guardrails \
	--text "내일 오후 3시까지 주문 취소해 주세요."
```

LiteLLM 추출(멀티 모델 운영):

```bash
python src/extractor.py \
	--provider litellm \
	--model openai/gpt-4o-mini \
	--litellm-api-base "https://your-litellm-gateway" \
	--litellm-api-key "YOUR_LITELLM_KEY" \
	--use-guardrails \
	--text "내일 오후 3시까지 주문 취소해 주세요."
```

로컬 LLM(Ollama) 단건 추출 예시:

```bash
python src/extractor.py \
	--provider ollama \
	--model qwen2.5:14b-instruct \
	--ollama-base-url http://localhost:11434 \
	--text "내일 오후 3시까지 주문 취소해 주세요."
```

동사별 분리 추출(복합 지시문 분해):

```bash
python src/extractor.py \
	--text "오늘,내일,모레 전철호가 보낸 쪽지 정리해서 이선정,영업팀에게 쪽지로 보내줘" \
	--model gpt-4.1 \
	--split-by-verb
```

예상 출력(JSON):

```json
{
	"language": "ko",
	"subject": "고객",
	"verb": "취소해 주세요",
	"object": "주문",
	"conditions": [
		{
			"type": "time",
			"text": "내일 오후 3시까지"
		}
	],
	"confidence": 0.95
}
```

## 4) 골든셋 평가

골든셋 파일: `data/golden_set_50.jsonl`

```bash
python src/evaluate.py --golden data/golden_set_50.jsonl --model gpt-4.1
```

OpenAI + Instructor 평가:

```bash
python src/evaluate.py \
	--provider openai \
	--model gpt-4.1 \
	--use-instructor \
	--use-guardrails \
	--golden data/golden_set_50.jsonl
```

로컬 LLM(Ollama) 평가 예시:

```bash
python src/evaluate.py \
	--provider ollama \
	--model qwen2.5:14b-instruct \
	--golden data/golden_set_50.jsonl
```

동사별 분리 평가:

```bash
python src/evaluate.py \
	--golden data/golden_set_50.jsonl \
	--model gpt-4.1 \
	--split-by-verb
```

동사 분해 전용 골든셋(복합 지시문 포함 12건) 평가:

```bash
python src/evaluate.py \
	--golden data/golden_set_verb_12.jsonl \
	--model gpt-4.1 \
	--split-by-verb
```

튜닝 규칙까지 반영한 동사별 분리 평가:

```bash
python src/evaluate.py \
	--golden data/golden_set_50.jsonl \
	--model gpt-4.1 \
	--split-by-verb \
	--extra-rules-file prompts/tuned_rules.txt
```

평가 지표:
- `subject_exact`
- `verb_exact`
- `object_exact`
- `condition_recall`
- `svo_macro_exact`

동사별 분리 모드 추가 지표:
- `action_count_exact`
- `action_precision`
- `action_recall`
- `action_f1`

## 5) 오탐/누락 에러 분석 리포트

아래 명령으로 유형별 에러 리포트를 생성합니다.

```bash
python src/error_analysis.py \
	--golden data/golden_set_50.jsonl \
	--model gpt-4.1 \
	--out-json reports/error_analysis.json \
	--out-md reports/error_analysis.md
```

로컬 LLM(Ollama) 에러 분석 예시:

```bash
python src/error_analysis.py \
	--provider ollama \
	--model qwen2.5:14b-instruct \
	--golden data/golden_set_50.jsonl \
	--out-json reports/error_analysis.json \
	--out-md reports/error_analysis.md
```

생성 결과:
- `reports/error_analysis.json`: 샘플별 상세 오답 정보
- `reports/error_analysis.md`: 요약 리포트(오탐/누락 유형 카운트)

## 6) 정확도 높이는 핵심 설계

- `temperature=0`으로 결정성 강화
- `with_structured_output(Pydantic)`로 JSON 스키마 강제
- 1차 추출 후 2차 검수(교정) 체인으로 누락/오추출 감소
- 다국어 few-shot 예시를 프롬프트에 포함

## 7) 파일 구조

```text
.
├── README.md
├── requirements.txt
├── data
│   └── golden_set_50.jsonl
├── reports
│   ├── error_analysis.json
│   └── error_analysis.md
└── src
	├── error_analysis.py
		├── evaluate.py
		└── extractor.py
```

## 8) 모델 권장

- 최고 정확도 우선: `gpt-4.1`
- 비용/속도 균형: `gpt-4.1-mini`

## 9) 프롬프트 자동 튜닝

`error_analysis.json`을 바탕으로 추가 규칙 파일을 생성합니다.

```bash
python src/prompt_tuning.py \
	--analysis-json reports/error_analysis.json \
	--out-rules prompts/tuned_rules.txt \
	--out-md reports/prompt_tuning.md \
	--min-count 2
```

생성 결과:
- `prompts/tuned_rules.txt`: 추출 프롬프트에 붙일 추가 규칙
- `reports/prompt_tuning.md`: 어떤 에러 유형을 기준으로 규칙이 선택됐는지 요약

튜닝 규칙 적용 단건 추출:

```bash
python src/extractor.py \
	--text "오늘 중으로 처리되지 않으면 주문을 전부 취소하겠습니다." \
	--model gpt-4.1 \
	--extra-rules-file prompts/tuned_rules.txt
```

## 10) 로컬 모델 벤치마크 (14B vs 32B)

동일 데이터셋에서 모델별 정확도를 비교합니다.

```bash
python src/benchmark_models.py \
	--provider ollama \
	--models qwen2.5:14b,qwen2.5:32b \
	--golden data/golden_set_verb_12.jsonl \
	--split-by-verb \
	--use-guardrails \
	--extra-rules-file prompts/tuned_rules.txt \
	--out-json reports/benchmark_models.json \
	--out-md reports/benchmark_models.md
```

출력:
- `reports/benchmark_models.json`: 모델별 원본 지표
- `reports/benchmark_models.md`: 비교표 요약