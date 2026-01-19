#!/bin/bash
# SFT 데이터 생성 및 실시간 검증 파이프라인
# tmux를 사용하여 생성과 검증을 병렬로 실행합니다.
# 정답이 없는 문제는 자동으로 재생성합니다 (최대 3회).
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

# 재생성 최대 횟수
MAX_RETRY=3

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
    echo "  --validate_and_retry 기존 데이터 재검증 + 누락 문제 자동 재생성"
    echo "  --merge_only         기존 결과 병합만 수행"
    echo "  --no-retry           재생성 비활성화"
    echo "  -h, --help           도움말 출력"
    echo ""
    echo "예시:"
    echo "  $0                   # tmux 병렬 실행 + 자동 재생성 (권장)"
    echo "  $0 --no-tmux         # 순차 실행"
    echo "  $0 --validate_and_retry  # 기존 데이터 재검증 + 누락 재생성"
    echo "  $0 --no-retry        # 재생성 비활성화"
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
VALIDATE_AND_RETRY=false
ENABLE_RETRY=true

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
        --validate_and_retry)
            VALIDATE_AND_RETRY=true
            shift
            ;;
        --merge_only)
            MERGE_ONLY=true
            shift
            ;;
        --no-retry)
            ENABLE_RETRY=false
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
echo "자동 재생성: $ENABLE_RETRY (최대 ${MAX_RETRY}회)"
if [ "$VALIDATE_AND_RETRY" = true ]; then
    echo "모드: 재검증 + 누락 재생성"
fi
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
RETRY_QUEUE="$OUTPUT_DIR/.retry_queue.jsonl"
STATUS_FILE="$OUTPUT_DIR/.status.json"

# 기존 파일 삭제
rm -f "$STOP_FILE"
rm -f "$RETRY_QUEUE"
rm -f "$STATUS_FILE"

# ============================================================================
# 명령어 생성 함수
# ============================================================================

build_generate_cmd() {
    local RETRY_FILE="$1"
    
    local CMD="python generate_sft_data.py \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --n $N \
        --worker $WORKER \
        --format $FORMAT \
        --base_url $BASE_URL \
        --model $MODEL"
    
    if [ -n "$INPUT_FILE" ]; then
        CMD="$CMD --input_file $INPUT_FILE"
    fi
    
    if [ "$MERGE_ONLY" = true ]; then
        CMD="$CMD --merge_only"
    fi
    
    if [ -n "$RETRY_FILE" ] && [ -f "$RETRY_FILE" ]; then
        CMD="$CMD --retry_file $RETRY_FILE"
    fi
    
    echo "$CMD"
}

build_watcher_cmd() {
    local EXTRA_ARGS=""
    # validate_and_retry 모드에서는 기존 validated를 신뢰하지 않고 전체 재검증/재집계
    if [ "$VALIDATE_AND_RETRY" = true ]; then
        EXTRA_ARGS="--rescan"
    fi
    echo "python validate_watcher.py \
        --watch_dirs $OUTPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --interval 2.0 \
        --stop_file $STOP_FILE \
        --retry_queue $RETRY_QUEUE \
        --expected_n $N \
        --dashboard_interval 10 \
        --status_file $STATUS_FILE \
        $EXTRA_ARGS"
}

# ============================================================================
# 순차 실행 모드 (재생성 루프 포함)
# ============================================================================

run_sequential() {
    local RETRY_COUNT=0
    local RETRY_FILE=""
    local SKIP_GENERATE=false
    
    # validate_and_retry 모드: 첫 번째 반복에서는 생성 스킵
    if [ "$VALIDATE_AND_RETRY" = true ]; then
        SKIP_GENERATE=true
        echo ""
        echo "========================================"
        echo "재검증 모드: 기존 데이터 검증 후 누락 문제 재생성"
        echo "========================================"
    fi
    
    while true; do
        # 종료 파일 삭제
        rm -f "$STOP_FILE"
        
        # 생성 단계 (SKIP_GENERATE가 false일 때만 실행)
        if [ "$SKIP_GENERATE" = false ] && [ "$GENERATE_ONLY" = false ] || [ "$SKIP_GENERATE" = false ] && [ $RETRY_COUNT -gt 0 ]; then
            echo ""
            echo "========================================"
            if [ $RETRY_COUNT -eq 0 ]; then
                echo "1단계: SFT 데이터 생성"
            else
                echo "재생성 시도 ${RETRY_COUNT}/${MAX_RETRY}"
            fi
            echo "========================================"
            
            GENERATE_CMD=$(build_generate_cmd "$RETRY_FILE")
            echo "실행: $GENERATE_CMD"
            echo ""
            eval $GENERATE_CMD
            echo ""
            echo "생성 완료!"
        fi
        
        if [ "$GENERATE_ONLY" = false ]; then
            echo ""
            echo "========================================"
            if [ "$VALIDATE_AND_RETRY" = true ] && [ $RETRY_COUNT -eq 0 ]; then
                echo "기존 데이터 재검증 (rescan 모드)"
            else
                echo "2단계: 정답 검증 및 필터링"
            fi
            echo "========================================"
            
            # Watcher로 검증 (재생성 큐 생성)
            WATCHER_CMD=$(build_watcher_cmd)
            echo "실행: $WATCHER_CMD"
            echo ""
            
            # Watcher 실행 (즉시 종료 - 기존 파일 처리 후 종료)
            touch "$STOP_FILE"  # 즉시 종료 신호
            eval $WATCHER_CMD
            
            echo ""
            echo "검증 완료!"
        fi
        
        # 재생성 필요 여부 확인
        if [ "$ENABLE_RETRY" = false ]; then
            echo "재생성 비활성화됨"
            break
        fi
        
        if [ ! -f "$RETRY_QUEUE" ]; then
            echo "모든 문제에 정답 있음. 완료!"
            break
        fi
        
        # validate_and_retry 모드에서 누락 발견 시 재생성 루프로 전환
        if [ "$VALIDATE_AND_RETRY" = true ] && [ "$SKIP_GENERATE" = true ]; then
            echo ""
            echo "========================================"
            echo "⚠️  누락된 문제 발견! 재생성을 시작합니다."
            echo "========================================"
            cat "$RETRY_QUEUE"
            echo ""
            SKIP_GENERATE=false  # 이후부터는 생성 허용
            RETRY_FILE="$RETRY_QUEUE"
            RETRY_COUNT=$((RETRY_COUNT + 1))
            continue
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        
        if [ $RETRY_COUNT -gt $MAX_RETRY ]; then
            echo ""
            echo "⚠️  최대 재생성 횟수(${MAX_RETRY})에 도달했습니다."
            echo "남은 재생성 필요 문제: $RETRY_QUEUE"
            break
        fi
        
        echo ""
        echo "========================================"
        echo "⚠️  재생성 필요 문제 발견!"
        echo "========================================"
        cat "$RETRY_QUEUE"
        echo ""
        
        RETRY_FILE="$RETRY_QUEUE"
    done
    
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
}

# ============================================================================
# tmux 병렬 실행 모드 (재생성 루프 포함)
# ============================================================================

run_with_tmux() {
    # 기존 세션 확인 및 종료
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "기존 세션 종료: $TMUX_SESSION"
        tmux kill-session -t "$TMUX_SESSION"
    fi
    
    # validate_and_retry 모드: 먼저 검증 실행 후 누락 있으면 재생성 루프로 진입
    if [ "$VALIDATE_AND_RETRY" = true ]; then
        echo ""
        echo "========================================"
        echo "재검증 모드: 기존 데이터 검증 후 누락 문제 재생성"
        echo "========================================"
        
        # 먼저 검증 실행 (rescan 모드)
        echo "1단계: 기존 데이터 재검증..."
        WATCHER_CMD=$(build_watcher_cmd)
        echo "실행: $WATCHER_CMD"
        touch "$STOP_FILE"  # 즉시 종료 신호
        eval $WATCHER_CMD
        
        # retry_queue가 없으면 완료
        if [ ! -f "$RETRY_QUEUE" ]; then
            echo ""
            echo "모든 문제에 정답 있음. 완료!"
            return
        fi
        
        echo ""
        echo "========================================"
        echo "⚠️  누락된 문제 발견! 재생성을 시작합니다."
        echo "========================================"
        cat "$RETRY_QUEUE"
        echo ""
        
        # validate_and_retry 비활성화하고 일반 모드로 진입
        VALIDATE_AND_RETRY=false
        # 아래 일반 tmux 루프로 계속 진행
    fi
    
    # 재생성 스크립트 생성 (tmux에서 실행)
    RETRY_SCRIPT="$OUTPUT_DIR/.retry_loop.sh"
    cat > "$RETRY_SCRIPT" << 'RETRY_SCRIPT_EOF'
#!/bin/bash
OUTPUT_DIR="$1"
DATA_DIR="$2"
N="$3"
WORKER="$4"
FORMAT="$5"
BASE_URL="$6"
MODEL="$7"
INPUT_FILE="$8"
MAX_RETRY="$9"
STOP_FILE="${10}"
RETRY_QUEUE="${11}"

RETRY_COUNT=0
RETRY_FILE=""

while true; do
    rm -f "$STOP_FILE"
    
    echo ""
    echo "========================================"
    if [ $RETRY_COUNT -eq 0 ]; then
        echo "=== Generator (데이터 생성) ==="
    else
        echo "=== 재생성 시도 ${RETRY_COUNT}/${MAX_RETRY} ==="
    fi
    echo "========================================"
    
    CMD="python generate_sft_data.py \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --n $N \
        --worker $WORKER \
        --format $FORMAT \
        --base_url $BASE_URL \
        --model $MODEL"
    
    if [ -n "$INPUT_FILE" ]; then
        CMD="$CMD --input_file $INPUT_FILE"
    fi
    
    if [ -n "$RETRY_FILE" ] && [ -f "$RETRY_FILE" ]; then
        CMD="$CMD --retry_file $RETRY_FILE"
    fi
    
    echo "실행: $CMD"
    eval $CMD
    
    echo ""
    echo "생성 완료. Watcher 종료 대기 중..."
    sleep 5
    touch "$STOP_FILE"
    
    # Watcher 종료 대기
    sleep 10
    
    # 재생성 필요 여부 확인
    if [ ! -f "$RETRY_QUEUE" ]; then
        echo "모든 문제에 정답 있음. 완료!"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    if [ $RETRY_COUNT -gt $MAX_RETRY ]; then
        echo ""
        echo "⚠️  최대 재생성 횟수(${MAX_RETRY})에 도달했습니다."
        break
    fi
    
    echo ""
    echo "⚠️  재생성 필요 문제 발견! 재시작..."
    cat "$RETRY_QUEUE"
    
    RETRY_FILE="$RETRY_QUEUE"
    rm -f "$STOP_FILE"
    
    sleep 3
done

echo ""
echo "========================================"
echo "Generator 종료"
echo "========================================"
RETRY_SCRIPT_EOF
    chmod +x "$RETRY_SCRIPT"
    
    # Watcher 재시작 스크립트
    WATCHER_SCRIPT="$OUTPUT_DIR/.watcher_loop.sh"
    cat > "$WATCHER_SCRIPT" << 'WATCHER_SCRIPT_EOF'
#!/bin/bash
OUTPUT_DIR="$1"
N="$2"
STOP_FILE="$3"
RETRY_QUEUE="$4"
STATUS_FILE="$5"

echo "=== Watcher (실시간 검증) ==="
echo "재생성 루프 지원 모드"
echo "상태 파일: $STATUS_FILE"
echo ""

while true; do
    python validate_watcher.py \
        --watch_dirs "$OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --interval 2.0 \
        --stop_file "$STOP_FILE" \
        --retry_queue "$RETRY_QUEUE" \
        --expected_n "$N" \
        --dashboard_interval 10 \
        --status_file "$STATUS_FILE"
    
    # 재생성 큐가 있고 종료 파일이 있으면 재시작 대기
    if [ -f "$RETRY_QUEUE" ]; then
        echo ""
        echo "재생성 대기 중... (3초 후 재시작)"
        rm -f "$STOP_FILE"
        sleep 3
    else
        echo "완료!"
        break
    fi
done
WATCHER_SCRIPT_EOF
    chmod +x "$WATCHER_SCRIPT"
    
    # tmux 세션 생성
    echo "tmux 세션 생성: $TMUX_SESSION"
    echo ""
    
    tmux new-session -d -s "$TMUX_SESSION" -n "pipeline"
    
    # 왼쪽 pane: Watcher
    tmux send-keys -t "$TMUX_SESSION:0" "$WATCHER_SCRIPT '$OUTPUT_DIR' '$N' '$STOP_FILE' '$RETRY_QUEUE' '$STATUS_FILE'" C-m
    
    # 오른쪽 pane: Generator (재생성 루프)
    tmux split-window -h -t "$TMUX_SESSION:0"
    tmux send-keys -t "$TMUX_SESSION:0.1" "$RETRY_SCRIPT '$OUTPUT_DIR' '$DATA_DIR' '$N' '$WORKER' '$FORMAT' '$BASE_URL' '$MODEL' '$INPUT_FILE' '$MAX_RETRY' '$STOP_FILE' '$RETRY_QUEUE'" C-m
    
    echo "========================================"
    echo "파이프라인이 백그라운드에서 실행 중입니다."
    echo "========================================"
    echo ""
    echo "세션 접속: tmux attach -t $TMUX_SESSION"
    echo "세션 종료: tmux kill-session -t $TMUX_SESSION"
    echo ""
    echo "왼쪽 pane: 실시간 검증 (Watcher)"
    echo "오른쪽 pane: 데이터 생성 (Generator) + 자동 재생성"
    echo ""
    echo "재생성: 정답이 없는 문제 발견 시 자동으로 최대 ${MAX_RETRY}회 재생성"
    echo ""
    echo "저장 구조:"
    echo "  - 생성된 원본: $OUTPUT_DIR/과목/subjectives/, multiples/"
    echo "  - 검증된 정답: $OUTPUT_DIR/과목/subjectives_validated/, multiples_validated/"
    echo ""
    echo "실시간 상태 확인:"
    echo "  cat $STATUS_FILE | python -m json.tool"
    echo ""
    
    echo "세션에 접속합니다... (Ctrl+B, D로 detach)"
    sleep 1
    tmux attach -t "$TMUX_SESSION"
}

# ============================================================================
# 실행
# ============================================================================

if [ "$USE_TMUX" = true ]; then
    run_with_tmux
else
    run_sequential
fi
