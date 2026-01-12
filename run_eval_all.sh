#!/bin/bash
# 여러 모델과 데이터셋에 대해 평가를 실행하는 스크립트

# 모델 리스트 (OpenAI 모델과 HuggingFace 모델 혼합 가능)
MODELS=(
    "gpt-4o-mini"
    "gpt-4o"
    "gpt-5.1"
    "gpt-5.2"
    "meta-llama/Llama-3.3-70B-Instruct"
    # "/group-volume/models/meta-llama/Llama-3.3-70B-Instruct"  # 로컬 경로 예시
    # "/folder/openai/gpt-oss-120b"  # gpt-oss 모델 예시
)

# 데이터셋 리스트 (split 이름)
DATASETS=(
    "2025_math"
    "2025_english"
    "2024_math"
    "2023_math"
    "2022_math"
    "2026_math"
    "2026_english"
)

# 최대 샘플 수 (0이면 전체, 테스트용으로는 작은 숫자 사용)
MAX_SAMPLES=0

# reasoning_effort 설정 (gpt-oss/Qwen 모델용: none, low, medium, high)
REASONING_EFFORT="high"

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
# 2단계: 각 모델과 데이터셋 조합에 대해 평가 실행
# ========================================
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        # 데이터셋이 없으면 스킵
        if [ ! -f "./data/${dataset}.jsonl" ]; then
            echo "[SKIP] 데이터셋 없음: $dataset"
            continue
        fi
        
        echo "=========================================="
        echo "평가 시작: 모델=$model, 데이터셋=$dataset"
        echo "=========================================="
        
        python main.py \
            --split "$dataset" \
            --model "$model" \
            --max_samples "$MAX_SAMPLES" \
            --reasoning_effort "$REASONING_EFFORT"
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] 완료: $model on $dataset"
        else
            echo "[FAILED] 실패: $model on $dataset"
        fi
        
        echo ""
    done
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
