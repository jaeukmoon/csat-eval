#!/bin/bash
# SFT 데이터 생성 및 검증 파이프라인
# 수학 문제 풀이 생성 -> 정답 검증 -> 필터링된 데이터 저장

set -e  # 오류 발생 시 중단

# ============================================================================
# 설정 (필요에 따라 수정하세요)
# ============================================================================

# 데이터 디렉토리
DATA_DIR="./data"

# 출력 디렉토리
OUTPUT_DIR="./sft_output"

# 문제당 생성 횟수
N=10

# 동시 워커 수
WORKER=20

# 출력 형식: simple, sharegpt, alpaca
FORMAT="sharegpt"

# vLLM 서버 설정
BASE_URL="http://10.0.74.208:8000/v1"
MODEL="glm-4.7"

# 특정 파일만 처리 (비워두면 모든 수학 파일 처리)
INPUT_FILE=""

# 병합만 수행 (true/false)
MERGE_ONLY=false

# ============================================================================
# 도움말
# ============================================================================

show_help() {
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  --data_dir DIR       데이터 디렉토리 (기본: ./data)"
    echo "  --output_dir DIR     출력 디렉토리 (기본: ./sft_output)"
    echo "  --n NUM              문제당 생성 횟수 (기본: 10)"
    echo "  --worker NUM         동시 워커 수 (기본: 20)"
    echo "  --format FORMAT      출력 형식: simple, sharegpt, alpaca (기본: sharegpt)"
    echo "  --base_url URL       vLLM 서버 URL"
    echo "  --model NAME         모델 이름"
    echo "  --input_file FILE    특정 파일만 처리"
    echo "  --merge_only         기존 결과 병합만 수행"
    echo "  --generate_only      생성만 수행 (검증 스킵)"
    echo "  --validate_only      검증만 수행 (생성 스킵)"
    echo "  -h, --help           도움말 출력"
    echo ""
    echo "예시:"
    echo "  $0                                    # 기본 설정으로 전체 파이프라인 실행"
    echo "  $0 --n 5 --worker 10                  # 생성 횟수와 워커 수 조정"
    echo "  $0 --input_file ./data/2025_math.jsonl  # 특정 파일만 처리"
    echo "  $0 --validate_only                    # 검증만 수행"
}

# ============================================================================
# 인자 파싱
# ============================================================================

GENERATE_ONLY=false
VALIDATE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --worker)
            WORKER="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --base_url)
            BASE_URL="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --merge_only)
            MERGE_ONLY=true
            shift
            ;;
        --generate_only)
            GENERATE_ONLY=true
            shift
            ;;
        --validate_only)
            VALIDATE_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# 설정 출력
# ============================================================================

echo "========================================"
echo "SFT 데이터 파이프라인"
echo "========================================"
echo "데이터 디렉토리: $DATA_DIR"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "생성 횟수 (n): $N"
echo "워커 수: $WORKER"
echo "출력 형식: $FORMAT"
echo "vLLM 서버: $BASE_URL"
echo "모델: $MODEL"
if [ -n "$INPUT_FILE" ]; then
    echo "입력 파일: $INPUT_FILE"
fi
echo "========================================"
echo ""

# ============================================================================
# 1단계: SFT 데이터 생성
# ============================================================================

if [ "$VALIDATE_ONLY" = false ]; then
    echo ""
    echo "========================================"
    echo "1단계: SFT 데이터 생성"
    echo "========================================"
    
    GENERATE_CMD="python generate_sft_data.py \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --n $N \
        --worker $WORKER \
        --format $FORMAT \
        --base_url $BASE_URL \
        --model $MODEL"
    
    if [ -n "$INPUT_FILE" ]; then
        GENERATE_CMD="$GENERATE_CMD --input_file $INPUT_FILE"
    fi
    
    if [ "$MERGE_ONLY" = true ]; then
        GENERATE_CMD="$GENERATE_CMD --merge_only"
    fi
    
    echo "실행: $GENERATE_CMD"
    echo ""
    
    eval $GENERATE_CMD
    
    echo ""
    echo "1단계 완료!"
fi

# ============================================================================
# 2단계: 정답 검증 및 필터링
# ============================================================================

if [ "$GENERATE_ONLY" = false ]; then
    echo ""
    echo "========================================"
    echo "2단계: 정답 검증 및 필터링"
    echo "========================================"
    
    # 병합된 파일 경로
    MERGED_FILE="$OUTPUT_DIR/merged/sft_math_all_${FORMAT}.jsonl"
    VALIDATED_DIR="$OUTPUT_DIR/validated"
    VALIDATED_FILE="$VALIDATED_DIR/sft_math_validated_${FORMAT}.jsonl"
    
    if [ -f "$MERGED_FILE" ]; then
        echo "검증 대상: $MERGED_FILE"
        
        VALIDATE_CMD="python validate_sft_data.py \
            --input $MERGED_FILE \
            --output $VALIDATED_FILE"
        
        echo "실행: $VALIDATE_CMD"
        echo ""
        
        eval $VALIDATE_CMD
        
        echo ""
        echo "2단계 완료!"
    else
        echo "경고: 병합된 파일을 찾을 수 없음: $MERGED_FILE"
        echo "생성 단계를 먼저 실행하세요."
    fi
fi

# ============================================================================
# 완료
# ============================================================================

echo ""
echo "========================================"
echo "파이프라인 완료!"
echo "========================================"
echo ""
echo "출력 파일:"
echo "  - 생성된 데이터: $OUTPUT_DIR/merged/sft_math_all_${FORMAT}.jsonl"
echo "  - 검증된 데이터: $OUTPUT_DIR/validated/sft_math_validated_${FORMAT}.jsonl"
echo ""
