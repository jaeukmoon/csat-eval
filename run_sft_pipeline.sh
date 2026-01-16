#!/bin/bash
# SFT 데이터 생성 및 실시간 검증 파이프라인
# tmux를 사용하여 생성과 검증을 병렬로 실행합니다.
#
# 저장 구조:
#   sft_output/
#     2022_math/
#       subjectives/           <- 생성된 주관식 버전
#       multiples/             <- 생성된 객관식 버전
#       subjectives_validated/ <- 검증된 주관식 (정답만)
#       multiples_validated/   <- 검증된 객관식 (정답만)
#     2025_math/
#       ...

set -e  # 오류 발생 시 중단

# ============================================================================
# 설정 (여기서 한 번만 수정하면 됩니다)
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
# 예: INPUT_FILE="./data/2022_math.jsonl"
INPUT_FILE=""

# 병합만 수행 (true/false)
MERGE_ONLY=false

# tmux 세션 이름
TMUX_SESSION="sft_pipeline"

# ============================================================================
# 도움말
# ============================================================================

show_help() {
    echo "사용법: $0 [옵션]"
    echo ""
    echo "tmux를 사용하여 데이터 생성과 검증을 병렬로 실행합니다."
    echo "스크립트 상단의 설정값을 수정한 후 실행하세요."
    echo ""
    echo "저장 구조:"
    echo "  sft_output/과목/subjectives/           <- 생성된 원본"
    echo "  sft_output/과목/multiples/             <- 생성된 원본"
    echo "  sft_output/과목/subjectives_validated/ <- 검증된 정답"
    echo "  sft_output/과목/multiples_validated/   <- 검증된 정답"
    echo ""
    echo "옵션:"
    echo "  --no-tmux            tmux 없이 순차 실행 (기존 방식)"
    echo "  --generate_only      생성만 수행 (검증 스킵)"
    echo "  --validate_only      검증만 수행 (생성 스킵)"
    echo "  --merge_only         기존 결과 병합만 수행"
    echo "  -h, --help           도움말 출력"
    echo ""
    echo "예시:"
    echo "  $0                   # tmux 병렬 실행 (권장)"
    echo "  $0 --no-tmux         # 순차 실행"
    echo ""
    echo "tmux 세션 관리:"
    echo "  tmux attach -t $TMUX_SESSION   # 세션 접속"
    echo "  tmux kill-session -t $TMUX_SESSION  # 세션 종료"
}

# ============================================================================
# 인자 파싱
# ============================================================================

USE_TMUX=true
GENERATE_ONLY=false
VALIDATE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-tmux)
            USE_TMUX=false
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
        --merge_only)
            MERGE_ONLY=true
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
echo "tmux 사용: $USE_TMUX"
if [ -n "$INPUT_FILE" ]; then
    echo "입력 파일: $INPUT_FILE"
fi
echo "========================================"
echo ""

# ============================================================================
# 디렉토리 준비
# ============================================================================

mkdir -p "$OUTPUT_DIR"

# 종료 신호 파일 경로
STOP_FILE="$OUTPUT_DIR/.watcher_stop"
rm -f "$STOP_FILE"  # 기존 종료 파일 삭제

# ============================================================================
# 명령어 생성
# ============================================================================

# 생성 명령어
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

# Watcher 명령어 (OUTPUT_DIR 전체를 모니터링, _validated 폴더는 자동 제외됨)
WATCHER_CMD="python validate_watcher.py \
    --watch_dirs $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --interval 2.0 \
    --stop_file $STOP_FILE"

# ============================================================================
# 실행
# ============================================================================

if [ "$USE_TMUX" = true ]; then
    # tmux 병렬 실행 모드
    echo "tmux 병렬 실행 모드"
    echo ""
    
    # 기존 세션 확인 및 종료
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "기존 세션 종료: $TMUX_SESSION"
        tmux kill-session -t "$TMUX_SESSION"
    fi
    
    if [ "$VALIDATE_ONLY" = true ]; then
        # 검증만 수행
        echo "검증만 수행 모드"
        echo "실행: $WATCHER_CMD"
        eval $WATCHER_CMD
    else
        # tmux 세션 생성 및 병렬 실행
        echo "tmux 세션 생성: $TMUX_SESSION"
        echo ""
        
        # 세션 생성 (첫 번째 pane: watcher)
        tmux new-session -d -s "$TMUX_SESSION" -n "pipeline"
        
        # 첫 번째 pane: Watcher 실행
        tmux send-keys -t "$TMUX_SESSION:0" "echo '=== Watcher (실시간 검증) ===' && $WATCHER_CMD" C-m
        
        # 화면 분할 (두 번째 pane: generator)
        tmux split-window -h -t "$TMUX_SESSION:0"
        
        # 두 번째 pane: Generator 실행 (완료 후 종료 신호 생성)
        GENERATE_WITH_SIGNAL="echo '=== Generator (데이터 생성) ===' && $GENERATE_CMD && echo '생성 완료. Watcher 종료 중...' && sleep 5 && touch $STOP_FILE"
        tmux send-keys -t "$TMUX_SESSION:0.1" "$GENERATE_WITH_SIGNAL" C-m
        
        echo "========================================"
        echo "파이프라인이 백그라운드에서 실행 중입니다."
        echo "========================================"
        echo ""
        echo "세션 접속: tmux attach -t $TMUX_SESSION"
        echo "세션 종료: tmux kill-session -t $TMUX_SESSION"
        echo ""
        echo "왼쪽 pane: 실시간 검증 (Watcher)"
        echo "오른쪽 pane: 데이터 생성 (Generator)"
        echo ""
        echo "저장 구조:"
        echo "  - 생성된 원본: $OUTPUT_DIR/과목/subjectives/, multiples/"
        echo "  - 검증된 정답: $OUTPUT_DIR/과목/subjectives_validated/, multiples_validated/"
        echo ""
        
        # 자동으로 세션에 attach
        echo "세션에 접속합니다... (Ctrl+B, D로 detach)"
        sleep 1
        tmux attach -t "$TMUX_SESSION"
    fi
else
    # 순차 실행 모드 (기존 방식)
    echo "순차 실행 모드"
    echo ""
    
    if [ "$VALIDATE_ONLY" = false ]; then
        echo "========================================"
        echo "1단계: SFT 데이터 생성"
        echo "========================================"
        echo "실행: $GENERATE_CMD"
        echo ""
        eval $GENERATE_CMD
        echo ""
        echo "1단계 완료!"
    fi
    
    if [ "$GENERATE_ONLY" = false ]; then
        echo ""
        echo "========================================"
        echo "2단계: 정답 검증 및 필터링"
        echo "========================================"
        
        # 병합된 파일 경로
        MERGED_FILE="$OUTPUT_DIR/merged/sft_math_all_${FORMAT}.jsonl"
        VALIDATED_FILE="$OUTPUT_DIR/merged/sft_math_validated_${FORMAT}.jsonl"
        
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
            echo "Watcher로 실시간 검증 수행..."
            eval $WATCHER_CMD
        fi
    fi
    
    echo ""
    echo "========================================"
    echo "파이프라인 완료!"
    echo "========================================"
    echo ""
    echo "저장 구조:"
    echo "  - 생성된 원본: $OUTPUT_DIR/과목/subjectives/, multiples/"
    echo "  - 검증된 정답: $OUTPUT_DIR/과목/subjectives_validated/, multiples_validated/"
    echo "  - 병합된 파일: $OUTPUT_DIR/merged/"
    echo ""
fi
