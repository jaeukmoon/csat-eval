#!/bin/bash
# 테스트용: 소량 샘플로 빠르게 테스트하는 스크립트

# API 키 확인 및 입력
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY가 설정되지 않았습니다."
    read -sp "OpenAI API 키를 입력하세요: " OPENAI_API_KEY
    echo ""
    export OPENAI_API_KEY
fi

# 테스트할 모델 (1개만)
MODEL="gpt-5.1"

# 테스트할 데이터셋 (1개만)
DATASET="2025_english"

# 테스트 샘플 수
MAX_SAMPLES=3

echo "=========================================="
echo "테스트 평가: 모델=$MODEL, 데이터셋=$DATASET"
echo "샘플 수: $MAX_SAMPLES"
echo "=========================================="

python main.py \
    --split "$DATASET" \
    --model "$MODEL" \
    --max_samples "$MAX_SAMPLES"

if [ $? -eq 0 ]; then
    echo "✅ 테스트 완료!"
else
    echo "❌ 테스트 실패!"
fi
