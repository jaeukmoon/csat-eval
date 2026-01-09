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
)

# 데이터셋 리스트 (split 이름)
DATASETS=(
    "2025_math"
    "2025_english"
    "2024_math"
    "2023_math"
    "2022_math"
    "2026_math"
)

# 최대 샘플 수 (0이면 전체, 테스트용으로는 작은 숫자 사용)
MAX_SAMPLES=0

# 각 모델과 데이터셋 조합에 대해 평가 실행
# mode는 모델 이름으로 자동 판별됨 (--mode 옵션 제거)
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "=========================================="
        echo "평가 시작: 모델=$model, 데이터셋=$dataset"
        echo "=========================================="
        
        # CSV 생성을 건너뛰기 위해 --no-csv 플래그 추가 (또는 별도 스크립트로 분리)
        # 일단은 기존대로 실행 (각 실행마다 CSV 업데이트)
        python main.py \
            --split "$dataset" \
            --model "$model" \
            --max_samples "$MAX_SAMPLES"
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] 완료: $model on $dataset"
        else
            echo "[FAILED] 실패: $model on $dataset"
        fi
        
        echo ""
    done
done

# 모든 평가 완료 후 각 데이터셋별로 최종 CSV 생성
echo "=========================================="
echo "최종 CSV 생성 중..."
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    echo "CSV 생성: $dataset"
    python -c "from csv_results import build_split_csv; build_split_csv('$dataset', results_root='./results', out_dir='.'); print('완료: $dataset')"
done

echo "=========================================="
echo "모든 평가 완료!"
echo "=========================================="
