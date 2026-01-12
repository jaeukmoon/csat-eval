#!/bin/bash
# vLLM 서버를 통한 단일 모델 평가 스크립트

# ========================================
# 설정 (여기에 값을 입력하세요)
# ========================================

# vLLM 서버 주소 (예: http://localhost:8000/v1)
VLLM_BASE_URL="http://localhost:8000/v1"

# vLLM 서버에 로드된 모델 ID (예: meta-llama/Llama-3.3-70B-Instruct)
VLLM_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"

# 평가할 데이터셋 (공백으로 구분, 또는 "all" 입력 시 모든 데이터셋)
# 예: DATASETS_INPUT="2026_math 2026_english 2025_math"
# 예: DATASETS_INPUT="all"
DATASETS_INPUT="all"

# 최대 샘플 수 (0이면 전체)
MAX_SAMPLES=0

# 최대 토큰 수
MAX_TOKENS=512

# Temperature (빈 문자열이면 모델 기본값 사용)
TEMPERATURE=""

# 동시 요청 수 (비동기 병렬 처리용)
CONCURRENCY=20

# ========================================
# 설정 확인 및 데이터셋 처리
# ========================================
echo "=========================================="
echo "vLLM 평가 설정"
echo "=========================================="
echo "vLLM 서버: $VLLM_BASE_URL"
echo "모델 ID: $VLLM_MODEL_ID"
echo "최대 샘플 수: $MAX_SAMPLES"
echo "최대 토큰 수: $MAX_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "동시 요청 수: $CONCURRENCY"
echo "=========================================="
echo ""

if [ -z "$VLLM_BASE_URL" ] || [ -z "$VLLM_MODEL_ID" ]; then
    echo "오류: VLLM_BASE_URL 또는 VLLM_MODEL_ID가 설정되지 않았습니다."
    echo "스크립트 상단의 설정 부분을 확인하세요."
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

echo "평가할 데이터셋: ${DATASETS[@]}"
echo ""

# ========================================
# 1단계: 데이터셋 자동 생성 (없는 경우만)
# ========================================
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
    
    # temperature가 설정되어 있으면 옵션 추가
    TEMP_OPT=""
    if [ -n "$TEMPERATURE" ]; then
        TEMP_OPT="--temperature $TEMPERATURE"
    fi
    
    python main.py \
        --mode vllm \
        --vllm_base_url "$VLLM_BASE_URL" \
        --vllm_model_id "$VLLM_MODEL_ID" \
        --split "$dataset" \
        --max_samples "$MAX_SAMPLES" \
        --max_tokens "$MAX_TOKENS" \
        --concurrency "$CONCURRENCY" \
        $TEMP_OPT
    
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

# CSV 결과 저장 디렉토리 생성
mkdir -p ./results/csv

for dataset in "${DATASETS[@]}"; do
    if [ -f "./data/${dataset}.jsonl" ]; then
        echo "CSV 생성: $dataset"
        python -c "from csv_results import build_split_csv; build_split_csv('$dataset', results_root='./results', out_dir='./results/csv'); print('완료: $dataset')"
    fi
done

echo ""
echo "=========================================="
echo "모든 평가 완료!"
echo "=========================================="
echo "결과 저장 위치:"
echo "  - JSONL 결과: ./results/{년도}/{과목}/"
echo "  - CSV 결과: ./results/csv/"
echo "=========================================="
