# CSAT Eval

한국 수능 문제에 대한 LLM 평가 도구

## 기능

- OpenAI API를 통한 평가
- HuggingFace Transformers 모델 로컬 평가
- gpt-oss/Qwen 모델 로컬 평가 (reasoning_effort/enable_thinking 지원)
- vLLM 서버를 통한 평가

## 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
- openai
- transformers
- torch
- datasets
- tqdm
- pdfplumber
- PyMuPDF

## 사용 방법

### 1. OpenAI API 평가

```bash
python main.py \
    --mode openai \
    --model gpt-5.1 \
    --split 2026_math \
    --max_tokens 512 \
    --temperature 0.0
```

### 2. HuggingFace Transformers 모델 평가

```bash
python main.py \
    --mode transformers \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --split 2026_math \
    --max_tokens 512 \
    --temperature 0.0
```

로컬 경로 사용:
```bash
python main.py \
    --mode transformers \
    --model /path/to/model \
    --split 2026_math
```

### 3. gpt-oss/Qwen 모델 평가

```bash
python main.py \
    --mode gpt-oss \
    --model /path/to/gpt-oss-model \
    --split 2026_math \
    --reasoning_effort high \
    --max_tokens 512 \
    --temperature 0.0
```

Qwen 모델:
```bash
python main.py \
    --mode gpt-oss \
    --model Qwen/Qwen2.5-72B-Instruct \
    --split 2026_math \
    --reasoning_effort high
```

### 4. vLLM 서버를 통한 평가

로컬 vLLM 서버:
```bash
python main.py \
    --mode vllm \
    --vllm_base_url http://localhost:8000/v1 \
    --vllm_model_id meta-llama/Llama-3.3-70B-Instruct \
    --split 2026_math \
    --max_tokens 512 \
    --temperature 0.0
```

원격 vLLM 서버:
```bash
python main.py \
    --mode vllm \
    --vllm_base_url http://192.168.1.100:8000/v1 \
    --vllm_model_id qwen/Qwen2.5-72B-Instruct \
    --split 2026_english \
    --max_tokens 512 \
    --temperature 0.0
```

### 5. 자동 모드 판별

`--mode`를 지정하지 않으면 모델 이름으로 자동 판별됩니다:

```bash
# OpenAI 모델 (gpt-*, o1, o3로 시작)
python main.py --model gpt-5.1 --split 2026_math

# gpt-oss/Qwen 모델 (gpt-oss, qwen 포함)
python main.py --model /path/to/gpt-oss-120b --split 2026_math

# HuggingFace 모델 (경로 또는 llama 포함)
python main.py --model meta-llama/Llama-3.3-70B-Instruct --split 2026_math
```

## 데이터셋 생성

### 자동 생성

평가 실행 시 데이터셋이 없으면 자동으로 생성됩니다:

```bash
python main.py --model gpt-5.1 --split 2026_math
```

### 수동 생성

```bash
python -m data_builder.main \
    --split 2026_math \
    --data_dir ./data \
    --pdf_dir ./pdf_csat_data
```

## 일괄 평가

`run_eval_all.sh`를 사용하여 여러 모델과 데이터셋에 대해 일괄 평가:

```bash
bash run_eval_all.sh
```

스크립트에서 모델 리스트와 데이터셋 리스트를 수정할 수 있습니다.

## 결과 확인

평가 결과는 `./results/{year}/{subject}/{model_name}.jsonl`에 저장됩니다.

CSV 결과는 각 평가 완료 후 자동으로 생성되며, `{split}_수능 LLM 풀이 결과.csv` 형식으로 저장됩니다.

- `(o)`: 정답
- `(x)`: 오답
- `(포기)`: 답안 없음

## 주요 옵션

- `--mode`: 평가 모드 (openai, transformers, gpt-oss, vllm)
- `--model`: 모델 이름 또는 경로
- `--split`: 데이터셋 split (예: 2026_math, 2025_english)
- `--max_tokens`: 최대 생성 토큰 수 (기본값: 512)
- `--temperature`: 생성 온도 (기본값: 0.0)
- `--reasoning_effort`: gpt-oss 모델의 reasoning effort (none, low, medium, high, 기본값: high)
- `--vllm_base_url`: vLLM 서버 base URL
- `--vllm_model_id`: vLLM 서버에서 사용할 모델 ID
- `--max_samples`: 평가할 최대 샘플 수 (0이면 전체)
