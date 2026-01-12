#!/bin/bash
# vLLM 서버를 통한 단일 모델 평가 스크립트

# ========================================
# 사용자 입력 받기
# ========================================
echo "=========================================="
echo "vLLM 평가 설정"
echo "=========================================="

# vLLM 서버 주소 입력
read -p "vLLM 서버 주소를 입력하세요 (예: http://localhost:8000/v1): " VLLM_BASE_URL
if [ -z "$VLLM_BASE_URL" ]; then
    echo "오류: vLLM 서버 주소가 입력되지 않았습니다."
    exit 1
fi

# 모델 ID 입력
read -p "모델 ID를 입력하세요 (예: meta-llama/Llama-3.3-70B-Instruct): " VLLM_MODEL_ID
if [ -z "$VLLM_MODEL_ID" ]; then
    echo "오류: 모델 ID가 입력되지 않았습니다."
    exit 1
fi

# 데이터셋 리스트 입력
echo ""
echo "평가할 데이터셋을 입력하세요 (공백으로 구분, 예: 2026_math 2026_english 2025_math):"
echo "또는 'all'을 입력하면 모든 데이터셋에 대해 평가합니다."
read -p "데이터셋: " DATASETS_INPUT

if [ -z "$DATASETS_INPUT" ]; then
    echo "오류: 데이터셋이 입력되지 않았습니다."
    exit 1
fi

# "all" 입력 시 모든 데이터셋 자동 탐지
if [ "$DATASETS_INPUT" = "all" ] || [ "$DATASETS_INPUT" = "ALL" ]; then
    echo "모든 데이터셋을 자동으로 탐지합니다..."
    DATASETS=()
    for jsonl_file in ./data/*.jsonl; do
        if [ -f "$jsonl_file" ]; then
            filename=$(basename "$jsonl_file")
            dataset_name="${filename%.jsonl}"
            DATASETS+=("$dataset_name")
        fi
    done
    if [ ${#DATASETS[@]} -eq 0 ]; then
        echo "오류: 데이터셋을 찾을 수 없습니다."
        exit 1
    fi
    echo "발견된 데이터셋: ${DATASETS[@]}"
else
    # 공백으로 구분된 입력을 배열로 변환
    DATASETS=($DATASETS_INPUT)
fi

# 최대 샘플 수 (0이면 전체)
read -p "최대 샘플 수 (0이면 전체, Enter=0): " MAX_SAMPLES
MAX_SAMPLES=${MAX_SAMPLES:-0}

# 최대 토큰 수
read -p "최대 토큰 수 (Enter=512): " MAX_TOKENS
MAX_TOKENS=${MAX_TOKENS:-512}

# Temperature
read -p "Temperature (Enter=0.0): " TEMPERATURE
TEMPERATURE=${TEMPERATURE:-0.0}

echo ""
echo "=========================================="
echo "설정 확인"
echo "=========================================="
echo "vLLM 서버: $VLLM_BASE_URL"
echo "모델 ID: $VLLM_MODEL_ID"
echo "데이터셋: ${DATASETS[@]}"
echo "최대 샘플 수: $MAX_SAMPLES"
echo "최대 토큰 수: $MAX_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "=========================================="
echo ""

read -p "계속하시겠습니까? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "취소되었습니다."
    exit 0
fi

# ========================================
# 1단계: 데이터셋 자동 생성 (없는 경우만)
# ========================================
echo ""
echo "=========================================="
echo "데이터셋 확인 및 생성 중..."
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    if [ ! -f "./data/${dataset}.jsonl" ]; then
        echo "데이터셋 생성: $dataset"
        python -m data_builder.main --split "$dataset" --data_dir ./data --pdf_dir ./pdf_csat_data
        if [ $? -ne 0 ]; then
            echo "[WARNING] 데이터셋 생성 실패: $dataset (평가에서 제외됨)"
        fi
    else
        echo "데이터셋 존재: $dataset"
    fi
done

echo ""

# ========================================
# 2단계: 각 데이터셋에 대해 평가 실행
# ========================================
for dataset in "${DATASETS[@]}"; do
    # 데이터셋이 없으면 스킵
    if [ ! -f "./data/${dataset}.jsonl" ]; then
        echo "[SKIP] 데이터셋 없음: $dataset"
        continue
    fi
    
    echo "=========================================="
    echo "평가 시작: 모델=$VLLM_MODEL_ID, 데이터셋=$dataset"
    echo "=========================================="
    
    python main.py \
        --mode vllm \
        --vllm_base_url "$VLLM_BASE_URL" \
        --vllm_model_id "$VLLM_MODEL_ID" \
        --split "$dataset" \
        --max_samples "$MAX_SAMPLES" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE"
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] 완료: $VLLM_MODEL_ID on $dataset"
    else
        echo "[FAILED] 실패: $VLLM_MODEL_ID on $dataset"
    fi
    
    echo ""
done

# ========================================
# 3단계: 모든 평가 완료 후 각 데이터셋별로 최종 CSV 생성
# ========================================
echo "=========================================="
echo "최종 CSV 생성 중..."
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    if [ -f "./data/${dataset}.jsonl" ]; then
        echo "CSV 생성: $dataset"
        python -c "from csv_results import build_split_csv; build_split_csv('$dataset', results_root='./results', out_dir='.'); print('완료: $dataset')"
    fi
done

echo "=========================================="
echo "모든 평가 완료!"
echo "=========================================="
