"""
SFT 데이터 실시간 검증 Watcher
새로 생성되는 파일을 감지하여 즉시 검증하고 정답만 저장합니다.

저장 구조:
  - 과목/subjectives/ → 과목/subjectives_validated/
  - 과목/multiples/   → 과목/multiples_validated/
"""
import os
import sys
import time
import json
import signal
import argparse
from pathlib import Path
from typing import Set, Dict, Any, Optional

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
                 poll_interval: float = 1.0, stop_file: str = None):
        """
        Args:
            watch_dirs: 모니터링할 디렉토리 목록 (예: [sft_output/2022_math/subjectives, ...])
            base_output_dir: 기본 출력 디렉토리 (예: sft_output)
            poll_interval: 파일 체크 간격 (초)
            stop_file: 이 파일이 생성되면 watcher 종료
        """
        self.watch_dirs = [os.path.abspath(d) for d in watch_dirs]
        self.base_output_dir = os.path.abspath(base_output_dir)
        self.poll_interval = poll_interval
        self.stop_file = stop_file
        
        # 처리된 파일 추적
        self.processed_files: Set[str] = set()
        
        # 통계
        self.stats = {
            "total_files": 0,
            "total_items": 0,
            "correct": 0,
            "incorrect": 0
        }
        
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
        """파일이 _validated 폴더에 있는지 확인 (재귀 방지)"""
        return "_validated" in file_path
    
    def _get_existing_files(self) -> Set[str]:
        """현재 존재하는 모든 .jsonl 파일 목록 반환 (_validated 폴더 제외)"""
        files = set()
        for watch_dir in self.watch_dirs:
            if not os.path.exists(watch_dir):
                continue
            # _validated 폴더는 제외
            if self._is_in_validated_dir(watch_dir):
                continue
            for root, dirs, filenames in os.walk(watch_dir):
                # _validated 디렉토리는 탐색에서 제외
                dirs[:] = [d for d in dirs if "_validated" not in d]
                for fname in filenames:
                    if fname.endswith('.jsonl'):
                        file_path = os.path.join(root, fname)
                        if not self._is_in_validated_dir(file_path):
                            files.add(file_path)
        return files
    
    def _get_validated_output_path(self, file_path: str) -> str:
        """
        원본 파일 경로에서 검증된 파일 저장 경로를 계산합니다.
        
        예:
          sft_output/2022_math/subjectives/0_0.jsonl
          → sft_output/2022_math/subjectives_validated/0_0.jsonl
          
          sft_output/2022_math/multiples/0_0.jsonl
          → sft_output/2022_math/multiples_validated/0_0.jsonl
        """
        abs_path = os.path.abspath(file_path)
        
        # 경로에서 subjectives 또는 multiples를 찾아서 _validated 추가
        path_parts = abs_path.split(os.sep)
        
        for i, part in enumerate(path_parts):
            if part in ("subjectives", "multiples"):
                path_parts[i] = part + "_validated"
                break
        
        return os.sep.join(path_parts)
    
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
    
    def _print_progress(self, result: Dict[str, Any]):
        """진행 상황 출력"""
        status = "✓" if result["correct"] > 0 else "✗"
        # 파일명과 부모 폴더 표시
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
        if self.stop_file:
            print(f"종료 파일: {self.stop_file}")
        print(f"{'='*50}\n")
        
        # 기존 파일 목록 (이미 처리된 것으로 간주할지 여부)
        # 새로 시작하면 기존 파일도 처리
        existing_files = self._get_existing_files()
        print(f"기존 파일 {len(existing_files)}개 발견, 처리 시작...\n")
        
        # 기존 파일 처리
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
                self._print_progress(result)
        
        if existing_files:
            self._print_stats()
        
        print("\n새 파일 대기 중... (Ctrl+C로 종료)\n")
        
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
                    
                    # 파일 쓰기 완료 대기 (짧은 딜레이)
                    time.sleep(0.1)
                    
                    result = self._validate_and_save(file_path)
                    self.processed_files.add(file_path)
                    self.stats["total_files"] += 1
                    self.stats["total_items"] += result["total"]
                    self.stats["correct"] += result["correct"]
                    self.stats["incorrect"] += result["incorrect"]
                    self._print_progress(result)
            
            time.sleep(self.poll_interval)
        
        # 최종 통계
        print("\n" + "="*50)
        print("Watcher 종료")
        self._print_stats()


def main():
    parser = argparse.ArgumentParser(description="SFT 데이터 실시간 검증 Watcher")
    parser.add_argument("--watch_dirs", nargs='+', required=True,
                        help="모니터링할 디렉토리 (예: sft_output/2022_math/subjectives)")
    parser.add_argument("--output_dir", required=True,
                        help="기본 출력 디렉토리 (예: sft_output)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="파일 체크 간격 (초, 기본: 1.0)")
    parser.add_argument("--stop_file", type=str, default=None,
                        help="이 파일이 생성되면 watcher 종료")
    
    args = parser.parse_args()
    
    watcher = ValidateWatcher(
        watch_dirs=args.watch_dirs,
        base_output_dir=args.output_dir,
        poll_interval=args.interval,
        stop_file=args.stop_file
    )
    
    watcher.run()


if __name__ == "__main__":
    main()
