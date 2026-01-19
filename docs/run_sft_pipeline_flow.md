# run_sft_pipeline.sh 코드 흐름 (현재 구현 기준)

이 문서는 `run_sft_pipeline.sh`를 실행했을 때의 **실제 코드 흐름**을 정리합니다.  
특히 **기본 모드**와 **재검증+재생성 모드(`--validate_and_retry`)**가 어떻게 다른지, 그리고 tmux에서 **Watcher/Generator가 어떻게 상호작용**하는지 설명합니다.

---

### 핵심 산출물/상태 파일

- **원본 생성 결과**: `sft_output/{source}/subjectives/`, `sft_output/{source}/multiples/`
- **검증 통과(정답+풀이) 결과**: `sft_output/{source}/subjectives_validated/`, `sft_output/{source}/multiples_validated/`
- **재생성 대상 큐**: `sft_output/.retry_queue.jsonl`
- **Watcher 종료 신호 파일**: `sft_output/.watcher_stop`
- **상태 파일(대시보드용)**: `sft_output/.status.json`

> `{source}`는 예: `2022_math`, `2026_math` 등입니다.

---

### 큰 그림 (모드/분기)

```mermaid
flowchart TD
  Start[run_sft_pipeline.sh 실행] --> ParseArgs[옵션 파싱]
  ParseArgs --> Init[.watcher_stop/.retry_queue/.status 초기화]
  Init --> Mode{USE_TMUX?}
  Mode -->|true| Tmux[run_with_tmux()]
  Mode -->|false| Seq[run_sequential()]

  Seq --> SeqFlag{VALIDATE_AND_RETRY?}
  SeqFlag -->|true| SeqDash[check_status.py로 현황/큐 생성]
  SeqFlag -->|false| SeqLoop[생성+검증+재생성 루프]
  SeqDash --> SeqQueue{.retry_queue.jsonl 존재?}
  SeqQueue -->|없음| Done[종료]
  SeqQueue -->|있음| SeqLoop

  Tmux --> TmuxFlag{VALIDATE_AND_RETRY?}
  TmuxFlag -->|true| TmuxDash[check_status.py로 현황/큐 생성]
  TmuxFlag -->|false| TmuxStart[tmux 세션 생성]
  TmuxDash --> TmuxQueue{.retry_queue.jsonl 존재?}
  TmuxQueue -->|없음| Done
  TmuxQueue -->|있음| TmuxStart

  TmuxStart --> WatcherPane[왼쪽 pane: watcher_loop.sh]
  TmuxStart --> GeneratorPane[오른쪽 pane: retry_loop.sh]
```

---

### 0) 실행 직후 공통 단계

1. **옵션 파싱**
   - `--no-tmux`, `--generate_only`, `--validate_and_retry`, `--merge_only`, `--no-retry` 등이 상단 설정을 덮어씁니다.
2. **초기화**
   - `sft_output/.watcher_stop`, `sft_output/.retry_queue.jsonl`, `sft_output/.status.json`을 삭제합니다.

---

### 1) 기본 모드 (`VALIDATE_AND_RETRY=false`)

목표: **전체 문제에 대해 N개씩 생성**하고, **정답+풀이가 1개 이상 확보될 때까지** 최대 `MAX_RETRY`만큼 자동 재생성.

#### 순차 모드 (`--no-tmux`)

- 루프 구조:
  - **(A) 생성**: `generate_sft_data.py` 실행 (처음에는 전체 생성, 이후에는 `--retry_file .retry_queue.jsonl`로 재생성 대상만)
  - **(B) 검증**: `validate_watcher.py`를 “기존 파일 처리 후 종료” 형태로 실행 (stop_file을 즉시 만들어서 한 바퀴 돌고 끝나게 함)
  - **(C) 재생성 판단**: `.retry_queue.jsonl`이 없으면 종료, 있으면 `MAX_RETRY`까지 반복

#### tmux 모드 (기본, 권장)

- 왼쪽 pane: `watcher_loop.sh`
  - `validate_watcher.py`를 반복 실행
  - `.retry_queue.jsonl`이 존재하면 재시작하며 대시보드/상태를 갱신
- 오른쪽 pane: `retry_loop.sh`
  - 처음에는 전체 생성
  - 이후 `.retry_queue.jsonl`이 있으면 `generate_sft_data.py --retry_file`로 재생성 반복

> tmux에서는 생성과 검증이 **동시에** 진행되므로 전체 속도가 빠릅니다.

---

### 2) 재검증+재생성 모드 (`--validate_and_retry`, `VALIDATE_AND_RETRY=true`)

목표: **기존에 생성된 데이터 상태를 먼저 스캔해서** “정답+풀이가 1개도 없는 문제”만 골라서 `MAX_RETRY`만큼 재생성.

#### Step 1: 대시보드 출력 + 재생성 큐 생성

`run_sft_pipeline.sh`는 이 모드에서 가장 먼저:

- `python check_status.py --output_dir "$OUTPUT_DIR" --expected_n "$N" --save_retry`

를 실행해서:

- 연도별로 **validated 개수/누락 문제**를 출력하고,
- 누락 문제 목록을 `sft_output/.retry_queue.jsonl`에 저장합니다.

`.retry_queue.jsonl`이 없으면 “모든 문제에 정답 있음. 완료!”로 종료합니다.

#### Step 2: 재생성 루프 시작 (tmux/순차 공통)

`.retry_queue.jsonl`이 있으면:

- tmux 모드: tmux 세션을 띄우고 오른쪽 pane에서 `generate_sft_data.py --retry_file`로 재생성 시작
- 순차 모드: 루프에서 `generate_sft_data.py --retry_file` → `validate_watcher.py` → 다시 큐 확인을 반복

---

### 3) 구성 요소별 역할

#### `generate_sft_data.py`

- 입력: `data/*_math.jsonl`
- 출력: `sft_output/{source}/subjectives/` + `.../multiples/`
- 재생성 모드: `--retry_file sft_output/.retry_queue.jsonl`
  - 파일에 적힌 `(source, question_type, problem_idx)`만 대상으로 생성

#### `validate_watcher.py`

- 입력: `sft_output/` 아래 원본 생성 폴더(단, `_validated`는 제외)
- 출력: `*_validated/`에 **정답+풀이가 있는 항목만** 저장
- `.retry_queue.jsonl` 생성/갱신:
  - **validated 폴더에 해당 problem_idx 파일이 하나도 없으면** 재생성 대상으로 기록

#### `check_status.py`

- 입력: `sft_output/`를 직접 스캔
- 출력: 연도별로 생성/검증 카운트와 누락 문제 출력
- `--save_retry` 시: `sft_output/.retry_queue.jsonl` 생성

---

### 4) “tmux에서 대시보드만 보이고 재생성이 안 되는 경우” 체크 포인트

오른쪽 pane(Generator)에 아래가 보여야 정상입니다:

- `=== Generator 시작 ===` (디버그)
- `실행: python generate_sft_data.py ... --retry_file sft_output/.retry_queue.jsonl`

만약 오른쪽 pane이 조용하면, 보통 아래 중 하나입니다:

- `.retry_queue.jsonl`이 실제로 생성되지 않음 (대시보드는 뜨지만 큐 파일이 없음)
- tmux 오른쪽 pane이 실행 중 예외/종료 (파이썬/경로/환경)
- `generate_sft_data.py`가 retry 모드에서 대상이 비어 있음(큐 내용 문제)

