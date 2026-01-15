# SFT 학습 데이터 생성 가이드

## 개요

`generate_sft_data.py`는 vLLM을 활용하여 수학 수능 문제에 대한 풀이를 생성하고, SFT(Supervised Fine-Tuning) 학습 데이터를 만드는 스크립트입니다.

## 코드 흐름

### 1. 초기화 및 데이터 로드

```
시작
  ↓
명령행 인자 파싱 (data_dir, output_dir, n, worker, format 등)
  ↓
sentences_ask_boxed.jsonl 로드 (boxed 요청 문장들)
  ↓
수학 JSONL 파일 검색 (2022_math.jsonl, 2023_math.jsonl, ...)
```

### 2. 문제 처리 루프

각 수학 파일에 대해:

```
파일 로드 (예: 2025_math.jsonl)
  ↓
각 문제마다 n번 반복 (기본 10번)
  ↓
  ├─ 문제 전처리 (clean_problem_text)
  │   ├─ 번호 제거: "1. " → ""
  │   └─ 점수 제거: "[2점]" → ""
  │
  ├─ 프롬프트 생성 (get_prompt)
  │   ├─ 해시 기반으로 boxed 문장 선택
  │   └─ 4가지 배치 변형 (문제 앞/뒤, 줄바꿈 1/2개)
  │
  ├─ vLLM API 호출 (send_msg)
  │   ├─ reasoning_effort='high'
  │   ├─ temperature=1.0
  │   └─ 재시도 로직 (최대 10회)
  │
  └─ 결과 저장
      ├─ 개별 파일: {problem_idx}_{gen_idx}.jsonl
      └─ 형식 변환 (simple/sharegpt/alpaca)
```

### 3. 결과 병합

```
모든 개별 결과 파일 수집
  ↓
형식에 맞게 변환
  ↓
단일 JSONL 파일로 병합
  ↓
sft_output/merged/sft_math_all_{format}.jsonl 저장
```

## 주요 함수 설명

### `clean_problem_text(problem_text: str) -> str`
- 문제 텍스트에서 번호와 점수를 제거
- LaTeX 형식은 유지
- 예: `"1. $x^2 + 1$ [2점]"` → `"$x^2 + 1$"`

### `get_prompt(problem_text, request_sentences, generation_id) -> str`
- 전처리된 문제와 boxed 요청 문장을 조합
- `generation_id`를 활용하여 다양한 프롬프트 변형 생성
- 4가지 배치 방식:
  1. `문제\n문장`
  2. `문제\n\n문장`
  3. `문장\n문제`
  4. `문장\n\n문제`

### `format_output(problem, solution, answer, source, generation_id, format_type) -> dict`
- 결과를 지정된 형식으로 변환
- **simple**: `{problem, solution, answer, source, generation_id}`
- **sharegpt**: `{conversations: [{from, value}...], ...}`
- **alpaca**: `{instruction, input, output, ...}`

### `process_item(idx, problems, request_sentences, ...)`
- 단일 문제-생성 쌍 처리
- 재시도 로직 포함 (최대 10회, 지수 백오프)
- 개별 결과 파일 저장

### `run_generation(...)`
- ThreadPoolExecutor를 사용한 병렬 처리
- 모든 문제에 대해 n번씩 풀이 생성

### `merge_results(input_dirs, output_path)`
- 여러 디렉토리의 결과를 하나로 병합
- 최종 SFT 학습 데이터 생성

## 사용 예시

### 기본 사용 (모든 수학 데이터)

```bash
python generate_sft_data.py \
    --data_dir ./data \
    --output_dir ./sft_output \
    --n 10 \
    --worker 200 \
    --format simple
```

### 특정 파일만 처리

```bash
python generate_sft_data.py \
    --input_file ./data/2025_math.jsonl \
    --n 10 \
    --worker 200
```

### ShareGPT 형식으로 출력

```bash
python generate_sft_data.py \
    --format sharegpt \
    --n 10
```

### 기존 결과만 병합

```bash
python generate_sft_data.py --merge_only
```

## 출력 구조

```
sft_output/
├── res_each_query/          # 개별 결과 (중간 저장)
│   ├── 2022_math/
│   │   ├── 0_0.jsonl        # 문제 0, 생성 0
│   │   ├── 0_1.jsonl        # 문제 0, 생성 1
│   │   └── ...
│   ├── 2023_math/
│   └── ...
└── merged/                  # 병합된 최종 데이터
    └── sft_math_all_simple.jsonl
```

## 출력 형식 예시

### Simple 형식

```json
{
  "problem": "$\\sqrt[3]{5} \\times 25^{\\frac{1}{3}}$ 의 값은?",
  "solution": "계산 과정... \\boxed{5}",
  "answer": 5,
  "source": "2025_math",
  "generation_id": 3
}
```

### ShareGPT 형식

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "$\\sqrt[3]{5} \\times 25^{\\frac{1}{3}}$ 의 값은?\n\nReturn your final answer within \\boxed{}."
    },
    {
      "from": "gpt",
      "value": "계산 과정... \\boxed{5}"
    }
  ],
  "source": "2025_math",
  "answer": 5,
  "generation_id": 3
}
```

### Alpaca 형식

```json
{
  "instruction": "다음 수학 문제를 풀고, 최종 답을 \\boxed{} 안에 넣어주세요.",
  "input": "$\\sqrt[3]{5} \\times 25^{\\frac{1}{3}}$ 의 값은?",
  "output": "계산 과정... \\boxed{5}",
  "answer": 5,
  "source": "2025_math",
  "generation_id": 3
}
```

## 주요 특징

1. **병렬 처리**: ThreadPoolExecutor로 대량 요청 처리
2. **재시도 로직**: 네트워크 오류 시 자동 재시도 (최대 10회)
3. **확장 가능한 형식**: simple/sharegpt/alpaca 형식 지원
4. **중복 방지**: 이미 생성된 결과는 스킵
5. **문제 전처리**: 번호/점수 자동 제거로 깔끔한 학습 데이터 생성

## 의존성

- `openai`: vLLM API 호출
- `aiohttp`: 비동기 HTTP (재시도 처리용)
- 표준 라이브러리: `os`, `json`, `re`, `time`, `glob`, `concurrent.futures`
