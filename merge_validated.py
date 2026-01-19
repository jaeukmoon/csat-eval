#!/usr/bin/env python3
"""
검증된 SFT 데이터를 년도별로 병합하는 스크립트

사용법:
  python merge_validated.py --input_dir ./sft_output --output_dir ./sft_output/merged

저장 구조:
  sft_output/
    2022_math/
      subjectives_validated/  <- 검증된 주관식
      multiples_validated/    <- 검증된 객관식
    2023_math/
      ...
  
  → sft_output/merged/
      2022_math.jsonl  <- 주관식+객관식 통합 (라벨링 포함)
      2023_math.jsonl
      all_years.jsonl  <- 전체 년도 통합
"""
import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional


def open_jsonl(file_path: str) -> List[Dict]:
    """JSONL 파일 읽기"""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict], file_path: str):
    """JSONL 파일 저장"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def parse_filename(filename: str) -> Optional[Dict]:
    """파일명에서 problem_idx, gen_idx 추출
    
    예: 15_3.jsonl -> {"problem_idx": 15, "gen_idx": 3}
    """
    match = re.match(r"(\d+)_(\d+)\.jsonl$", filename)
    if match:
        return {
            "problem_idx": int(match.group(1)),
            "gen_idx": int(match.group(2))
        }
    return None


def find_sources(input_dir: str) -> List[str]:
    """input_dir에서 source (년도_math) 목록 찾기"""
    sources = []
    
    if not os.path.exists(input_dir):
        return sources
    
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # subjectives_validated 또는 multiples_validated가 있으면 source로 인정
            if (os.path.exists(os.path.join(item_path, "subjectives_validated")) or 
                os.path.exists(os.path.join(item_path, "multiples_validated"))):
                sources.append(item)
    
    return sorted(sources)


def collect_validated_items(source_dir: str, question_type: str) -> List[Dict]:
    """특정 question_type의 validated 폴더에서 모든 아이템 수집
    
    Args:
        source_dir: source 디렉토리 경로 (예: sft_output/2022_math)
        question_type: "subjectives" 또는 "multiples"
    
    Returns:
        라벨링된 아이템 리스트
    """
    validated_dir = os.path.join(source_dir, f"{question_type}_validated")
    
    if not os.path.exists(validated_dir):
        return []
    
    items = []
    
    for filename in sorted(os.listdir(validated_dir)):
        if not filename.endswith('.jsonl'):
            continue
        
        parsed = parse_filename(filename)
        if not parsed:
            continue
        
        file_path = os.path.join(validated_dir, filename)
        
        try:
            file_items = open_jsonl(file_path)
            
            for item in file_items:
                # 메타데이터 추가
                item["_meta"] = {
                    "question_type": question_type,
                    "question_type_kr": "주관식" if question_type == "subjectives" else "객관식",
                    "problem_idx": parsed["problem_idx"],
                    "problem_num": parsed["problem_idx"] + 1,  # 1-based 번호
                    "gen_idx": parsed["gen_idx"],
                    "source_file": filename
                }
                items.append(item)
        except Exception as e:
            print(f"  오류: {file_path} - {e}")
    
    return items


def merge_source(source_dir: str, source: str) -> List[Dict]:
    """하나의 source (년도)에 대해 주관식+객관식 병합
    
    Args:
        source_dir: source 디렉토리 경로
        source: source 이름 (예: 2022_math)
    
    Returns:
        병합된 아이템 리스트 (문제 번호순 정렬)
    """
    all_items = []
    
    # 주관식 수집
    subj_items = collect_validated_items(source_dir, "subjectives")
    print(f"    주관식: {len(subj_items)}개")
    all_items.extend(subj_items)
    
    # 객관식 수집
    mult_items = collect_validated_items(source_dir, "multiples")
    print(f"    객관식: {len(mult_items)}개")
    all_items.extend(mult_items)
    
    # source 정보 추가
    for item in all_items:
        item["_meta"]["source"] = source
    
    # 정렬: 문제 번호 → question_type → gen_idx
    all_items.sort(key=lambda x: (
        x["_meta"]["problem_idx"],
        x["_meta"]["question_type"],
        x["_meta"]["gen_idx"]
    ))
    
    return all_items


def print_summary(items: List[Dict], source: str):
    """병합 결과 요약 출력"""
    if not items:
        print(f"    데이터 없음")
        return
    
    # 통계 계산
    by_qtype = defaultdict(int)
    by_problem = defaultdict(lambda: defaultdict(int))
    
    for item in items:
        meta = item["_meta"]
        qtype = meta["question_type"]
        prob_idx = meta["problem_idx"]
        
        by_qtype[qtype] += 1
        by_problem[prob_idx][qtype] += 1
    
    # 문제별 커버리지
    problems_with_subj = sum(1 for p in by_problem.values() if p.get("subjectives", 0) > 0)
    problems_with_mult = sum(1 for p in by_problem.values() if p.get("multiples", 0) > 0)
    total_problems = len(by_problem)
    
    print(f"    총 {len(items)}개 항목")
    print(f"    - 주관식: {by_qtype.get('subjectives', 0)}개 ({problems_with_subj}문제)")
    print(f"    - 객관식: {by_qtype.get('multiples', 0)}개 ({problems_with_mult}문제)")
    print(f"    - 문제 범위: {min(by_problem.keys())+1}번 ~ {max(by_problem.keys())+1}번 ({total_problems}문제)")


def main():
    parser = argparse.ArgumentParser(
        description="검증된 SFT 데이터를 년도별로 병합",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python merge_validated.py
  python merge_validated.py --input_dir ./sft_output --output_dir ./sft_output/merged
  python merge_validated.py --include_all  # 전체 통합 파일도 생성
        """
    )
    parser.add_argument("--input_dir", type=str, default="./sft_output",
                        help="입력 디렉토리 (기본: ./sft_output)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="출력 디렉토리 (기본: {input_dir}/merged)")
    parser.add_argument("--include_all", action="store_true",
                        help="전체 년도 통합 파일(all_years.jsonl)도 생성")
    parser.add_argument("--flatten_meta", action="store_true",
                        help="_meta 필드를 최상위로 펼치기")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "merged")
    
    print("=" * 60)
    print("검증된 SFT 데이터 병합")
    print("=" * 60)
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print()
    
    # source 목록 찾기
    sources = find_sources(input_dir)
    
    if not sources:
        print("검증된 데이터가 없습니다.")
        print(f"  {input_dir}/ 에 *_validated 폴더가 있는지 확인하세요.")
        return
    
    print(f"발견된 연도: {len(sources)}개")
    for s in sources:
        print(f"  - {s}")
    print()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 년도별 병합
    all_items = []
    
    for source in sources:
        source_dir = os.path.join(input_dir, source)
        output_file = os.path.join(output_dir, f"{source}.jsonl")
        
        print(f"[{source}]")
        items = merge_source(source_dir, source)
        
        if items:
            # _meta 펼치기 옵션
            if args.flatten_meta:
                for item in items:
                    meta = item.pop("_meta", {})
                    for k, v in meta.items():
                        item[k] = v
            
            save_jsonl(items, output_file)
            print(f"    → {output_file}")
            print_summary(items, source)
            
            all_items.extend(items)
        else:
            print(f"    검증된 데이터 없음")
        print()
    
    # 전체 통합 파일
    if args.include_all and all_items:
        all_output_file = os.path.join(output_dir, "all_years.jsonl")
        save_jsonl(all_items, all_output_file)
        print(f"[전체 통합]")
        print(f"  → {all_output_file}")
        print(f"  총 {len(all_items)}개 항목")
    
    print("=" * 60)
    print("병합 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
