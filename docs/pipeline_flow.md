# SFT 데이터 생성 및 검증 파이프라인 흐름

## 개요

이 문서는 `run_sft_pipeline.sh`에서 `validate_sft_data.py`와 `validate_watcher.py`가 어떻게 tmux를 사용하여 병렬로 실행되는지 설명합니다.

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│              run_sft_pipeline.sh (메인 스크립트)              │
│                                                               │
│  1. 설정값 로드 (상단에 정의된 변수들)                        │
│  2. tmux 세션 생성: "sft_pipeline"                           │
│  3. 두 개의 pane으로 분할                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ├─────────────────────────┐
                            │                         │
                            ▼                         ▼
        ┌───────────────────────────┐   ┌───────────────────────────┐
        │   Pane 0 (왼쪽)           │   │   Pane 1 (오른쪽)          │
        │   validate_watcher.py     │   │   generate_sft_data.py     │
        │                           │   │                           │
        │   - 실시간 파일 감시       │   │   - 문제 데이터 생성       │
        │   - 새 파일 검증           │   │   - sft_output/에 저장    │
        │   - 정답만 저장            │   │                           │
        └───────────────────────────┘   └───────────────────────────┘
                    │                                 │
                    │                                 │
                    │         파일 생성 감지          │
                    │◄────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │   validate_sft_data.py     │
        │   (함수들만 import)         │
        │                             │
        │   - extract_boxed_answer()  │
        │   - normalize_latex()       │
        │   - check_answer()          │
        │   - validate_item()         │
        └───────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │   검증된 파일 저장          │
        │   *_validated/ 폴더에      │
        └───────────────────────────┘
```

## 상세 실행 흐름

### 1단계: 스크립트 시작

```bash
./run_sft_pipeline.sh
```

- 스크립트 상단의 설정값을 읽어옴 (입력 불필요)
- tmux 세션 `sft_pipeline`이 이미 있으면 종료
- 새 tmux 세션 생성

### 2단계: tmux 세션 생성 및 분할

```bash
tmux new-session -d -s "sft_pipeline" -n "pipeline"
tmux split-window -h -t "$TMUX_SESSION:0"
```

**결과:**
- 세션 이름: `sft_pipeline`
- Pane 0 (왼쪽): Watcher 실행
- Pane 1 (오른쪽): Generator 실행

### 3단계: Pane 0 - Watcher 시작

```bash
python validate_watcher.py \
    --watch_dirs sft_output \
    --output_dir sft_output \
    --interval 2.0 \
    --stop_file sft_output/.watcher_stop
```

**Watcher의 역할:**
1. `sft_output/` 디렉토리 전체를 모니터링
2. `_validated` 폴더는 제외 (재귀 방지)
3. 새 `.jsonl` 파일이 생성되면 즉시 감지
4. `validate_sft_data.py`의 함수들을 import하여 검증 수행
5. 정답만 `*_validated/` 폴더에 저장

**저장 경로 변환:**
```
sft_output/2022_math/subjectives/0_0.jsonl
  → sft_output/2022_math/subjectives_validated/0_0.jsonl

sft_output/2022_math/multiples/0_0.jsonl
  → sft_output/2022_math/multiples_validated/0_0.jsonl
```

### 4단계: Pane 1 - Generator 시작

```bash
python generate_sft_data.py \
    --data_dir ./data \
    --output_dir sft_output \
    --n 10 \
    --worker 20 \
    --format sharegpt \
    --base_url http://10.0.74.208:8000/v1 \
    --model glm-4.7
```

**Generator의 역할:**
1. 수학 문제 파일들을 읽어옴 (`data/*_math.jsonl`)
2. 각 문제에 대해 주관식/객관식 버전 생성
3. vLLM 서버에 요청하여 풀이 생성
4. `sft_output/과목/subjectives/` 또는 `multiples/`에 저장
5. 파일이 생성될 때마다 Watcher가 자동으로 감지

### 5단계: 실시간 검증 루프

```
Generator가 파일 생성
    ↓
Watcher가 파일 감지 (2초마다 체크)
    ↓
validate_sft_data.py의 함수들 호출
    ↓
정답 추출 및 비교
    ↓
정답만 *_validated/ 폴더에 저장
```

**검증 과정:**
1. `extract_boxed_answer()`: `\boxed{}` 안의 답 추출
2. `normalize_latex()`: LaTeX 표현 정규화 (`\frac{1}{2}` → `1/2`)
3. `normalize_answer()`: 답 정규화 및 약분
4. `check_answer()`: 추출된 답과 정답 비교
5. 정답이면 저장, 오답이면 스킵

### 6단계: 종료

Generator가 모든 작업 완료 후:
```bash
touch sft_output/.watcher_stop
```

Watcher가 종료 파일을 감지하면:
- 현재 처리 중인 파일 마무리
- 통계 출력
- 종료

## 파일 간 관계

### validate_sft_data.py
- **역할**: 검증 로직의 "라이브러리"
- **함수들**:
  - `extract_boxed_answer()`: boxed 답 추출
  - `normalize_latex()`: LaTeX 정규화
  - `normalize_answer()`: 답 정규화
  - `check_answer()`: 답 비교
  - `validate_item()`: 단일 항목 검증
- **사용처**:
  - `validate_watcher.py`에서 import하여 사용
  - 순차 실행 모드에서도 직접 호출 가능

### validate_watcher.py
- **역할**: 실시간 파일 감시 및 검증 오케스트레이션
- **기능**:
  - 디렉토리 모니터링
  - 새 파일 감지
  - `validate_sft_data.py`의 함수들 호출
  - 검증된 파일 저장
- **의존성**: `validate_sft_data.py`의 함수들 import

### run_sft_pipeline.sh
- **역할**: 전체 파이프라인 오케스트레이션
- **기능**:
  - tmux 세션 관리
  - 두 프로세스 병렬 실행
  - 종료 신호 생성

## 실행 모드

### 1. tmux 병렬 실행 (기본, 권장)
```bash
./run_sft_pipeline.sh
```
- Generator와 Watcher가 동시에 실행
- 실시간으로 검증 진행
- 가장 효율적

### 2. 순차 실행
```bash
./run_sft_pipeline.sh --no-tmux
```
- Generator 완료 후 Watcher 실행
- tmux가 없는 환경에서 사용

### 3. 생성만 수행
```bash
./run_sft_pipeline.sh --generate_only
```
- Watcher 실행 안 함
- 나중에 수동으로 검증 가능

### 4. 검증만 수행
```bash
./run_sft_pipeline.sh --validate_only
```
- 기존 생성된 파일들만 검증
- Generator 실행 안 함

## 디렉토리 구조

```
sft_output/
├── 2022_math/
│   ├── subjectives/              # Generator가 생성
│   │   ├── 0_0.jsonl
│   │   ├── 0_1.jsonl
│   │   └── ...
│   ├── multiples/                # Generator가 생성
│   │   ├── 0_0.jsonl
│   │   └── ...
│   ├── subjectives_validated/     # Watcher가 생성 (정답만)
│   │   ├── 0_0.jsonl
│   │   └── ...
│   └── multiples_validated/       # Watcher가 생성 (정답만)
│       └── ...
├── 2025_math/
│   └── ...
├── merged/                        # generate_sft_data.py가 생성
│   └── sft_math_all_sharegpt.jsonl
└── .watcher_stop                  # Generator가 생성 (종료 신호)
```

## tmux 세션 관리

### 세션 접속
```bash
tmux attach -t sft_pipeline
```

### 세션에서 나가기 (백그라운드 유지)
```
Ctrl+B, D
```

### Pane 전환
```
Ctrl+B, ← (왼쪽 pane)
Ctrl+B, → (오른쪽 pane)
```

### 세션 종료
```bash
tmux kill-session -t sft_pipeline
```

## 주의사항

1. **재귀 방지**: Watcher는 `_validated` 폴더를 모니터링하지 않음
2. **파일 쓰기 완료 대기**: 새 파일 감지 후 0.1초 대기하여 쓰기 완료 보장
3. **중복 처리 방지**: `processed_files` Set으로 이미 처리한 파일 추적
4. **종료 신호**: Generator 완료 후 5초 대기 후 종료 파일 생성 (마지막 파일 처리 시간 확보)

## 성능 최적화

- **병렬 실행**: 생성과 검증이 동시에 진행되어 전체 시간 단축
- **실시간 검증**: 생성 완료를 기다리지 않고 즉시 검증
- **증분 처리**: 새로 생성된 파일만 처리하여 효율적
