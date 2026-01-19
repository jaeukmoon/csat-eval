"""
SFT 데이터 실시간 검증 Watcher
새로 생성되는 파일을 감지하여 즉시 검증하고 정답만 저장합니다.

저장 구조:
  - 과목/subjectives/ → 과목/subjectives_validated/
  - 과목/multiples/   → 과목/multiples_validated/

재생성 기능:
  - 문제별 정답 개수 추적
  - 정답이 0개인 문제는 retry_queue.jsonl에 기록

대시보드:
  - 연도별/문제별 진행 상황 실시간 표시
  - 상태 파일(.status.json)에 현재 상태 저장
"""
import os
import re
import sys
import time
import json
import signal
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, Any, Optional, Tuple, List

# validate_sft_data.py에서 함수들 import
from validate_sft_data import (
    open_jsonl, 
    validate_item, 
    check_answer,
    extract_boxed_answer,
    get_assistant_content
)


class ValidateWatcher:
    """디렉토리를 모니터링하고 새 파일을 검증하는 Watcher"""
    
    def __init__(self, watch_dirs: list, base_output_dir: str, 
                 poll_interval: float = 1.0, stop_file: str = None,
                 retry_queue_file: str = None, expected_n: int = 10,
                 dashboard_interval: int = 10, status_file: str = None):
        """
        Args:
            watch_dirs: 모니터링할 디렉토리 목록
            base_output_dir: 기본 출력 디렉토리
            poll_interval: 파일 체크 간격 (초)
            stop_file: 이 파일이 생성되면 watcher 종료
            retry_queue_file: 재생성 필요 문제 저장 파일 경로
            expected_n: 문제당 예상 생성 횟수
            dashboard_interval: 대시보드 갱신 간격 (초)
            status_file: 상태 저장 파일 경로
        """
        self.watch_dirs = [os.path.abspath(d) for d in watch_dirs]
        self.base_output_dir = os.path.abspath(base_output_dir)
        self.poll_interval = poll_interval
        self.stop_file = stop_file
        self.retry_queue_file = retry_queue_file
        self.expected_n = expected_n
        self.dashboard_interval = dashboard_interval
        self.status_file = status_file
        
        # 처리된 파일 추적
        self.processed_files: Set[str] = set()
        
        # 문제별 정답 개수 추적
        # 키: (source, question_type, problem_idx)
        # 값: {"correct": int, "total": int, "generated": int}
        self.problem_stats: Dict[Tuple[str, str, int], Dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0, "generated": 0}
        )
        
        # 통계
        self.stats = {
            "total_files": 0,
            "total_items": 0,
            "correct": 0,
            "incorrect": 0
        }
        
        # 대시보드 관련
        self.last_dashboard_time = 0
        self.start_time = time.time()
        
        # 종료 플래그
        self.running = True
        
        # 종료 신호 핸들러
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """종료 신호 처리"""
        print(f"\n종료 신호 수신 (signal {signum})")
        self.running = False
    
    def _is_in_validated_dir(self, file_path: str) -> bool:
        """파일이 _validated 폴더에 있는지 확인"""
        return "_validated" in file_path
    
    def _parse_file_path(self, file_path: str) -> Optional[Tuple[str, str, int, int]]:
        """파일 경로에서 source, question_type, problem_idx, gen_idx 추출"""
        try:
            path_parts = file_path.replace("\\", "/").split("/")
            fname = os.path.basename(file_path)
            
            match = re.match(r"(\d+)_(\d+)\.jsonl$", fname)
            if not match:
                return None
            problem_idx = int(match.group(1))
            gen_idx = int(match.group(2))
            
            question_type = None
            source = None
            for i, part in enumerate(path_parts):
                if part in ("subjectives", "multiples"):
                    question_type = part
                    if i > 0:
                        source = path_parts[i - 1]
                    break
            
            if source and question_type:
                return (source, question_type, problem_idx, gen_idx)
            return None
        except Exception:
            return None
    
    def _get_existing_files(self) -> Set[str]:
        """현재 존재하는 모든 .jsonl 파일 목록 반환"""
        files = set()
        for watch_dir in self.watch_dirs:
            if not os.path.exists(watch_dir):
                continue
            if self._is_in_validated_dir(watch_dir):
                continue
            for root, dirs, filenames in os.walk(watch_dir):
                dirs[:] = [d for d in dirs if "_validated" not in d]
                for fname in filenames:
                    if fname.endswith('.jsonl'):
                        file_path = os.path.join(root, fname)
                        if not self._is_in_validated_dir(file_path):
                            files.add(file_path)
        return files
    
    def _get_validated_output_path(self, file_path: str) -> str:
        """원본 파일 경로에서 검증된 파일 저장 경로 계산"""
        abs_path = os.path.abspath(file_path)
        path_parts = abs_path.split(os.sep)
        
        for i, part in enumerate(path_parts):
            if part in ("subjectives", "multiples"):
                path_parts[i] = part + "_validated"
                break
        
        return os.sep.join(path_parts)
    
    def _get_validated_dir(self, source: str, question_type: str) -> str:
        """source와 question_type에 해당하는 validated 디렉토리 경로 반환"""
        return os.path.join(self.base_output_dir, source, f"{question_type}_validated")
    
    def _check_validated_exists(self, source: str, question_type: str, problem_idx: int) -> bool:
        """해당 문제에 대한 검증된 파일이 하나라도 있는지 확인 (validated 폴더 기준)"""
        validated_dir = self._get_validated_dir(source, question_type)
        if not os.path.exists(validated_dir):
            return False
        
        # 해당 문제 번호로 시작하는 파일이 있는지 확인
        for fname in os.listdir(validated_dir):
            if fname.startswith(f"{problem_idx}_") and fname.endswith(".jsonl"):
                return True
        return False
    
    def _count_validated_files(self, source: str, question_type: str, problem_idx: int) -> int:
        """해당 문제에 대한 검증된 파일 개수 반환 (validated 폴더 기준)"""
        validated_dir = self._get_validated_dir(source, question_type)
        if not os.path.exists(validated_dir):
            return 0
        
        count = 0
        for fname in os.listdir(validated_dir):
            if fname.startswith(f"{problem_idx}_") and fname.endswith(".jsonl"):
                count += 1
        return count
    
    def _scan_validated_folders(self):
        """기존 validated 폴더를 스캔하여 통계 초기화"""
        print("기존 검증 완료 폴더 스캔 중...")
        validated_count = 0
        
        # base_output_dir 내의 모든 source 디렉토리 탐색
        if not os.path.exists(self.base_output_dir):
            return
        
        for source_name in os.listdir(self.base_output_dir):
            source_path = os.path.join(self.base_output_dir, source_name)
            if not os.path.isdir(source_path):
                continue
            
            # subjectives_validated, multiples_validated 폴더 확인
            for qtype in ["subjectives", "multiples"]:
                validated_dir = os.path.join(source_path, f"{qtype}_validated")
                if not os.path.exists(validated_dir):
                    continue
                
                # 해당 폴더의 모든 jsonl 파일 스캔
                for fname in os.listdir(validated_dir):
                    if not fname.endswith(".jsonl"):
                        continue
                    
                    match = re.match(r"(\d+)_(\d+)\.jsonl$", fname)
                    if not match:
                        continue
                    
                    problem_idx = int(match.group(1))
                    gen_idx = int(match.group(2))
                    
                    key = (source_name, qtype, problem_idx)
                    
                    # 파일 내용을 읽어서 정답 개수 카운트
                    file_path = os.path.join(validated_dir, fname)
                    try:
                        data = open_jsonl(file_path)
                        correct_count = len(data)  # validated 폴더의 파일은 모두 정답
                        
                        self.problem_stats[key]["correct"] += correct_count
                        self.problem_stats[key]["total"] += correct_count
                        self.problem_stats[key]["generated"] += 1
                        
                        # 이미 검증된 원본 파일 경로를 processed_files에 추가
                        # (원본 폴더 경로로 변환)
                        original_dir = os.path.join(source_path, qtype)
                        original_file = os.path.join(original_dir, fname)
                        self.processed_files.add(original_file)
                        
                        validated_count += 1
                        
                    except Exception as e:
                        print(f"  오류: {file_path} - {e}")
        
        print(f"  → 기존 검증 완료 파일: {validated_count}개")
    
    def _validate_and_save(self, file_path: str) -> Dict[str, Any]:
        """파일을 검증하고 정답만 저장"""
        result = {
            "file": file_path,
            "total": 0,
            "correct": 0,
            "incorrect": 0
        }
        
        try:
            data = open_jsonl(file_path)
            if not data:
                return result
            
            result["total"] = len(data)
            correct_items = []
            
            for item in data:
                validation = validate_item(item)
                if validation["is_correct"]:
                    result["correct"] += 1
                    correct_items.append(item)
                else:
                    result["incorrect"] += 1
            
            # 문제별 통계 업데이트
            parsed = self._parse_file_path(file_path)
            if parsed:
                source, question_type, problem_idx, gen_idx = parsed
                key = (source, question_type, problem_idx)
                self.problem_stats[key]["correct"] += result["correct"]
                self.problem_stats[key]["total"] += result["total"]
                self.problem_stats[key]["generated"] += 1
            
            # 정답이 있으면 저장
            if correct_items:
                output_path = self._get_validated_output_path(file_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in correct_items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            return result
            
        except Exception as e:
            print(f"  오류: {file_path} - {e}")
            return result
    
    def _get_grouped_stats(self) -> Dict[str, Dict[str, List[Dict]]]:
        """통계를 source별, question_type별로 그룹화"""
        grouped = defaultdict(lambda: defaultdict(list))
        
        for (source, question_type, problem_idx), stats in self.problem_stats.items():
            grouped[source][question_type].append({
                "problem_idx": problem_idx,
                "generated": stats["generated"],
                "correct": stats["correct"],
                "total": stats["total"]
            })
        
        # 문제 번호로 정렬
        for source in grouped:
            for qtype in grouped[source]:
                grouped[source][qtype].sort(key=lambda x: x["problem_idx"])
        
        return grouped
    
    def _print_dashboard(self, clear_screen: bool = True):
        """대시보드 출력"""
        if clear_screen:
            # 터미널 화면 지우기
            print("\033[2J\033[H", end="")
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60)}분 {int(elapsed % 60)}초"
        
        print("╔" + "═" * 68 + "╗")
        print(f"║  SFT 데이터 생성 현황 (실시간)              {now}  ║")
        print(f"║  경과 시간: {elapsed_str:<15}                                    ║")
        print("╠" + "═" * 68 + "╣")
        
        grouped = self._get_grouped_stats()
        
        total_problems = 0
        total_with_correct = 0
        
        for source in sorted(grouped.keys()):
            for qtype in ["subjectives", "multiples"]:
                if qtype not in grouped[source]:
                    continue
                
                problems = grouped[source][qtype]
                qtype_display = "주관식" if qtype == "subjectives" else "객관식"
                
                print(f"║  {source} ({qtype_display})" + " " * (68 - len(source) - len(qtype_display) - 7) + "║")
                print("║  ┌────┬────────┬────────┬────────────────────────────────────────┐  ║")
                print("║  │ #  │ 생성   │ 정답   │ 진행률                                 │  ║")
                print("║  ├────┼────────┼────────┼────────────────────────────────────────┤  ║")
                
                for p in problems:
                    idx = p["problem_idx"] + 1  # 1-based 표시
                    gen = p["generated"]
                    correct = p["correct"]
                    
                    progress = (gen / self.expected_n * 100) if self.expected_n > 0 else 0
                    bar_filled = int(progress / 5)
                    bar = "█" * bar_filled + "░" * (20 - bar_filled)
                    
                    # 상태 표시
                    if gen >= self.expected_n:
                        if correct > 0:
                            status = "✓"
                        else:
                            status = "⚠"
                    else:
                        status = " "
                    
                    total_problems += 1
                    if correct > 0:
                        total_with_correct += 1
                    
                    print(f"║  │{idx:3d} │ {gen:2d}/{self.expected_n:<2d}  │  {correct:3d}   │ {bar} {progress:3.0f}% {status}│  ║")
                
                print("║  └────┴────────┴────────┴────────────────────────────────────────┘  ║")
                print("║" + " " * 68 + "║")
        
        # 전체 통계
        accuracy = (self.stats["correct"] / self.stats["total_items"] * 100 
                    if self.stats["total_items"] > 0 else 0)
        coverage = (total_with_correct / total_problems * 100 
                    if total_problems > 0 else 0)
        
        print("╠" + "═" * 68 + "╣")
        print(f"║  전체 통계                                                          ║")
        print(f"║    - 처리된 파일: {self.stats['total_files']:5d}                                         ║")
        print(f"║    - 총 생성: {self.stats['total_items']:5d} | 정답: {self.stats['correct']:5d} ({accuracy:5.1f}%)                   ║")
        print(f"║    - 문제 커버리지: {total_with_correct}/{total_problems} ({coverage:.1f}%) 문제에 최소 1개 정답         ║")
        
        # 재생성 필요 문제
        retry_problems = self._find_problems_needing_retry()
        if retry_problems:
            print(f"║    - ⚠️  재생성 필요: {len(retry_problems)}개 문제                                    ║")
        
        print("╚" + "═" * 68 + "╝")
        print("\n(Ctrl+C로 종료)")
    
    def _save_status_file(self):
        """현재 상태를 JSON 파일로 저장"""
        if not self.status_file:
            return
        
        grouped = self._get_grouped_stats()
        retry_problems = self._find_problems_needing_retry()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - self.start_time,
            "stats": self.stats.copy(),
            "problem_stats": {},
            "retry_needed": len(retry_problems),
            "sources": {}
        }
        
        # 문제별 통계
        for (source, qtype, prob_idx), pstats in self.problem_stats.items():
            key = f"{source}/{qtype}/{prob_idx}"
            status["problem_stats"][key] = pstats.copy()
        
        # source별 요약
        for source in grouped:
            status["sources"][source] = {}
            for qtype in grouped[source]:
                problems = grouped[source][qtype]
                total_gen = sum(p["generated"] for p in problems)
                total_correct = sum(p["correct"] for p in problems)
                with_correct = sum(1 for p in problems if p["correct"] > 0)
                status["sources"][source][qtype] = {
                    "problems": len(problems),
                    "total_generated": total_gen,
                    "total_correct": total_correct,
                    "problems_with_correct": with_correct
                }
        
        try:
            os.makedirs(os.path.dirname(self.status_file) if os.path.dirname(self.status_file) else ".", exist_ok=True)
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"상태 파일 저장 오류: {e}")
    
    def _print_progress(self, result: Dict[str, Any]):
        """진행 상황 출력 (대시보드 모드가 아닐 때)"""
        status = "✓" if result["correct"] > 0 else "✗"
        parent = os.path.basename(os.path.dirname(result['file']))
        fname = os.path.basename(result['file'])
        print(f"  {status} {parent}/{fname}: "
              f"{result['correct']}/{result['total']} 정답")
    
    def _print_stats(self):
        """전체 통계 출력"""
        accuracy = (self.stats["correct"] / self.stats["total_items"] * 100 
                    if self.stats["total_items"] > 0 else 0)
        print(f"\n{'='*50}")
        print(f"실시간 검증 통계:")
        print(f"  - 처리된 파일: {self.stats['total_files']}")
        print(f"  - 총 항목: {self.stats['total_items']}")
        print(f"  - 정답: {self.stats['correct']} ({accuracy:.1f}%)")
        print(f"  - 오답: {self.stats['incorrect']}")
        print(f"{'='*50}")
    
    def _find_problems_needing_retry(self) -> list:
        """정답이 0개인 문제 목록 반환 (validated 폴더 기준으로 실제 확인)"""
        problems_needing_retry = []
        
        for (source, question_type, problem_idx), stats in self.problem_stats.items():
            # 생성이 완료된 문제인지 확인
            if stats["generated"] < self.expected_n:
                continue
            
            # validated 폴더에 실제로 파일이 있는지 확인
            has_validated = self._check_validated_exists(source, question_type, problem_idx)
            
            if not has_validated:
                problems_needing_retry.append({
                    "source": source,
                    "problem_idx": problem_idx,
                    "question_type": question_type,
                    "total_generated": stats["generated"]
                })
        
        return problems_needing_retry
    
    def _save_retry_queue(self):
        """재생성 필요 문제를 파일에 저장"""
        if not self.retry_queue_file:
            return
        
        problems = self._find_problems_needing_retry()
        
        if problems:
            os.makedirs(os.path.dirname(self.retry_queue_file) if os.path.dirname(self.retry_queue_file) else ".", exist_ok=True)
            with open(self.retry_queue_file, 'w', encoding='utf-8') as f:
                for p in problems:
                    f.write(json.dumps(p, ensure_ascii=False) + '\n')
            print(f"\n재생성 필요 문제 {len(problems)}개 → {self.retry_queue_file}")
        else:
            if os.path.exists(self.retry_queue_file):
                os.remove(self.retry_queue_file)
            print(f"\n재생성 필요 문제: 없음 (모든 문제에 정답 있음)")
    
    def _print_retry_summary(self):
        """재생성 필요 문제 요약 출력"""
        problems = self._find_problems_needing_retry()
        
        if problems:
            print(f"\n{'='*50}")
            print(f"⚠️  재생성 필요 문제: {len(problems)}개")
            print(f"{'='*50}")
            
            by_source = defaultdict(list)
            for p in problems:
                by_source[p["source"]].append(p)
            
            for source, items in sorted(by_source.items()):
                subj = [p for p in items if p["question_type"] == "subjectives"]
                mult = [p for p in items if p["question_type"] == "multiples"]
                print(f"  {source}:")
                if subj:
                    indices = [p["problem_idx"] for p in subj]
                    print(f"    - subjectives: {len(subj)}개 (문제 {indices})")
                if mult:
                    indices = [p["problem_idx"] for p in mult]
                    print(f"    - multiples: {len(mult)}개 (문제 {indices})")
            print(f"{'='*50}")
    
    def run(self):
        """Watcher 실행"""
        print(f"{'='*50}")
        print(f"실시간 검증 Watcher 시작")
        print(f"{'='*50}")
        print(f"모니터링 디렉토리:")
        for d in self.watch_dirs:
            print(f"  - {d}")
        print(f"저장 방식: subjectives → subjectives_validated")
        print(f"          multiples → multiples_validated")
        print(f"체크 간격: {self.poll_interval}초")
        print(f"대시보드 갱신: {self.dashboard_interval}초")
        if self.stop_file:
            print(f"종료 파일: {self.stop_file}")
        if self.retry_queue_file:
            print(f"재생성 큐: {self.retry_queue_file}")
        if self.status_file:
            print(f"상태 파일: {self.status_file}")
        print(f"{'='*50}\n")
        
        # 기존 validated 폴더 스캔 (재시작 시 상태 복원)
        self._scan_validated_folders()
        
        # 기존 파일 처리
        existing_files = self._get_existing_files()
        print(f"기존 파일 {len(existing_files)}개 발견, 처리 시작...\n")
        
        for file_path in sorted(existing_files):
            if not self.running:
                break
            if file_path not in self.processed_files:
                result = self._validate_and_save(file_path)
                self.processed_files.add(file_path)
                self.stats["total_files"] += 1
                self.stats["total_items"] += result["total"]
                self.stats["correct"] += result["correct"]
                self.stats["incorrect"] += result["incorrect"]
        
        # 초기 대시보드 출력
        if existing_files:
            self._print_dashboard()
            self._save_status_file()
        
        # 새 파일 모니터링 루프
        while self.running:
            # 종료 파일 체크
            if self.stop_file and os.path.exists(self.stop_file):
                print(f"\n종료 파일 감지: {self.stop_file}")
                break
            
            # 새 파일 탐색
            current_files = self._get_existing_files()
            new_files = current_files - self.processed_files
            
            if new_files:
                for file_path in sorted(new_files):
                    if not self.running:
                        break
                    
                    time.sleep(0.1)
                    
                    result = self._validate_and_save(file_path)
                    self.processed_files.add(file_path)
                    self.stats["total_files"] += 1
                    self.stats["total_items"] += result["total"]
                    self.stats["correct"] += result["correct"]
                    self.stats["incorrect"] += result["incorrect"]
            
            # 대시보드 갱신
            current_time = time.time()
            if current_time - self.last_dashboard_time >= self.dashboard_interval:
                self._print_dashboard()
                self._save_status_file()
                self.last_dashboard_time = current_time
            
            time.sleep(self.poll_interval)
        
        # 최종 통계 및 재생성 큐 저장
        print("\n" + "="*50)
        print("Watcher 종료")
        self._print_stats()
        self._print_retry_summary()
        self._save_retry_queue()
        self._save_status_file()


def main():
    parser = argparse.ArgumentParser(description="SFT 데이터 실시간 검증 Watcher")
    parser.add_argument("--watch_dirs", nargs='+', required=True,
                        help="모니터링할 디렉토리")
    parser.add_argument("--output_dir", required=True,
                        help="기본 출력 디렉토리")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="파일 체크 간격 (초, 기본: 1.0)")
    parser.add_argument("--stop_file", type=str, default=None,
                        help="이 파일이 생성되면 watcher 종료")
    parser.add_argument("--retry_queue", type=str, default=None,
                        help="재생성 필요 문제 저장 파일 경로")
    parser.add_argument("--expected_n", type=int, default=10,
                        help="문제당 예상 생성 횟수 (기본: 10)")
    parser.add_argument("--dashboard_interval", type=int, default=10,
                        help="대시보드 갱신 간격 (초, 기본: 10)")
    parser.add_argument("--status_file", type=str, default=None,
                        help="상태 저장 파일 경로 (.status.json)")
    
    args = parser.parse_args()
    
    watcher = ValidateWatcher(
        watch_dirs=args.watch_dirs,
        base_output_dir=args.output_dir,
        poll_interval=args.interval,
        stop_file=args.stop_file,
        retry_queue_file=args.retry_queue,
        expected_n=args.expected_n,
        dashboard_interval=args.dashboard_interval,
        status_file=args.status_file
    )
    
    watcher.run()


if __name__ == "__main__":
    main()
