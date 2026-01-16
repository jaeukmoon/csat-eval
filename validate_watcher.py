"""
SFT 데이터 실시간 검증 Watcher
새로 생성되는 파일을 감지하여 즉시 검증하고 정답만 저장합니다.
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
    
    def __init__(self, watch_dirs: list, output_dir: str, 
                 poll_interval: float = 1.0, stop_file: str = None):
        """
        Args:
            watch_dirs: 모니터링할 디렉토리 목록
            output_dir: 검증된 데이터를 저장할 디렉토리
            poll_interval: 파일 체크 간격 (초)
            stop_file: 이 파일이 생성되면 watcher 종료
        """
        self.watch_dirs = watch_dirs
        self.output_dir = output_dir
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
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 종료 신호 핸들러
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """종료 신호 처리"""
        print(f"\n종료 신호 수신 (signal {signum})")
        self.running = False
    
    def _get_existing_files(self) -> Set[str]:
        """현재 존재하는 모든 .jsonl 파일 목록 반환"""
        files = set()
        for watch_dir in self.watch_dirs:
            if not os.path.exists(watch_dir):
                continue
            for root, _, filenames in os.walk(watch_dir):
                for fname in filenames:
                    if fname.endswith('.jsonl'):
                        files.add(os.path.join(root, fname))
        return files
    
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
                # 원본 경로 구조 유지
                rel_path = None
                for watch_dir in self.watch_dirs:
                    if file_path.startswith(watch_dir):
                        rel_path = os.path.relpath(file_path, watch_dir)
                        break
                
                if rel_path is None:
                    rel_path = os.path.basename(file_path)
                
                output_path = os.path.join(self.output_dir, rel_path)
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
        print(f"  {status} {os.path.basename(result['file'])}: "
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
        print(f"모니터링 디렉토리: {self.watch_dirs}")
        print(f"출력 디렉토리: {self.output_dir}")
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
                        help="모니터링할 디렉토리 (여러 개 지정 가능)")
    parser.add_argument("--output_dir", required=True,
                        help="검증된 데이터를 저장할 디렉토리")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="파일 체크 간격 (초, 기본: 1.0)")
    parser.add_argument("--stop_file", type=str, default=None,
                        help="이 파일이 생성되면 watcher 종료")
    
    args = parser.parse_args()
    
    watcher = ValidateWatcher(
        watch_dirs=args.watch_dirs,
        output_dir=args.output_dir,
        poll_interval=args.interval,
        stop_file=args.stop_file
    )
    
    watcher.run()


if __name__ == "__main__":
    main()
