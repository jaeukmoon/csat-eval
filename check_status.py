#!/usr/bin/env python3
"""
SFT 데이터 상태 확인 대시보드

연도별로 validated 개수를 출력하고, 재생성이 필요한 문제를 표시합니다.

사용법:
  python check_status.py
  python check_status.py --output_dir ./sft_output
  python check_status.py --save_retry  # retry_queue.jsonl 저장
"""
import os
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def find_sources(output_dir: str) -> List[str]:
    """output_dir에서 source (연도_math) 목록 찾기"""
    sources = []
    
    if not os.path.exists(output_dir):
        return sources
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # subjectives 또는 multiples 폴더가 있으면 source로 인정
            if (os.path.exists(os.path.join(item_path, "subjectives")) or 
                os.path.exists(os.path.join(item_path, "multiples"))):
                sources.append(item)
    
    return sorted(sources)


def count_files(directory: str) -> Dict[int, int]:
    """디렉토리 내 파일을 문제별로 카운트
    
    Returns:
        {problem_idx: count}
    """
    counts = defaultdict(int)
    
    if not os.path.exists(directory):
        return counts
    
    for fname in os.listdir(directory):
        match = re.match(r"(\d+)_\d+\.jsonl$", fname)
        if match:
            problem_idx = int(match.group(1))
            counts[problem_idx] += 1
    
    return counts


def get_problem_indices(directory: str) -> Set[int]:
    """디렉토리 내 모든 문제 인덱스 추출"""
    indices = set()
    
    if not os.path.exists(directory):
        return indices
    
    for fname in os.listdir(directory):
        match = re.match(r"(\d+)_\d+\.jsonl$", fname)
        if match:
            indices.add(int(match.group(1)))
    
    return indices


def analyze_source(output_dir: str, source: str, expected_n: int) -> Dict:
    """하나의 source 분석"""
    source_dir = os.path.join(output_dir, source)
    
    result = {
        "source": source,
        "subjectives": {
            "generated": 0,
            "validated": 0,
            "problems_generated": 0,
            "problems_validated": 0,
            "missing": []
        },
        "multiples": {
            "generated": 0,
            "validated": 0,
            "problems_generated": 0,
            "problems_validated": 0,
            "missing": []
        }
    }
    
    for qtype in ["subjectives", "multiples"]:
        gen_dir = os.path.join(source_dir, qtype)
        val_dir = os.path.join(source_dir, f"{qtype}_validated")
        
        gen_counts = count_files(gen_dir)
        val_counts = count_files(val_dir)
        
        # 생성된 문제 인덱스
        gen_indices = set(gen_counts.keys())
        val_indices = set(val_counts.keys())
        
        # 통계
        result[qtype]["generated"] = sum(gen_counts.values())
        result[qtype]["validated"] = sum(val_counts.values())
        result[qtype]["problems_generated"] = len(gen_indices)
        result[qtype]["problems_validated"] = len(val_indices)
        
        # 재생성 필요한 문제 찾기
        # 조건: 생성은 됐는데 validated가 0개인 문제
        for prob_idx in gen_indices:
            if prob_idx not in val_indices:
                result[qtype]["missing"].append({
                    "problem_idx": prob_idx,
                    "generated": gen_counts[prob_idx]
                })
        
        # 정렬
        result[qtype]["missing"].sort(key=lambda x: x["problem_idx"])
    
    return result


def print_dashboard(results: List[Dict], expected_n: int):
    """대시보드 출력"""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " SFT 데이터 상태 대시보드 ".center(78) + "║")
    print("╠" + "═" * 78 + "╣")
    
    total_gen = 0
    total_val = 0
    total_missing = 0
    
    for result in results:
        source = result["source"]
        
        subj = result["subjectives"]
        mult = result["multiples"]
        
        # 연도 헤더
        print("║" + f" 📅 {source} ".ljust(78) + "║")
        print("║" + "─" * 78 + "║")
        
        # 주관식
        subj_gen = subj["generated"]
        subj_val = subj["validated"]
        subj_prob_gen = subj["problems_generated"]
        subj_prob_val = subj["problems_validated"]
        subj_missing = len(subj["missing"])
        
        print(f"║   주관식: 생성 {subj_gen:4d}개 ({subj_prob_gen}문제) │ 검증 {subj_val:4d}개 ({subj_prob_val}문제) │ ⚠ 누락 {subj_missing:2d}문제".ljust(78) + "║")
        
        # 객관식
        mult_gen = mult["generated"]
        mult_val = mult["validated"]
        mult_prob_gen = mult["problems_generated"]
        mult_prob_val = mult["problems_validated"]
        mult_missing = len(mult["missing"])
        
        print(f"║   객관식: 생성 {mult_gen:4d}개 ({mult_prob_gen}문제) │ 검증 {mult_val:4d}개 ({mult_prob_val}문제) │ ⚠ 누락 {mult_missing:2d}문제".ljust(78) + "║")
        
        # 누락 문제 상세
        if subj["missing"]:
            missing_nums = [str(m["problem_idx"] + 1) for m in subj["missing"][:10]]
            more = f" +{len(subj['missing']) - 10}" if len(subj["missing"]) > 10 else ""
            print(f"║      → 주관식 누락: {', '.join(missing_nums)}{more}번".ljust(78) + "║")
        
        if mult["missing"]:
            missing_nums = [str(m["problem_idx"] + 1) for m in mult["missing"][:10]]
            more = f" +{len(mult['missing']) - 10}" if len(mult["missing"]) > 10 else ""
            print(f"║      → 객관식 누락: {', '.join(missing_nums)}{more}번".ljust(78) + "║")
        
        print("║" + " " * 78 + "║")
        
        total_gen += subj_gen + mult_gen
        total_val += subj_val + mult_val
        total_missing += subj_missing + mult_missing
    
    # 전체 통계
    print("╠" + "═" * 78 + "╣")
    print("║" + " 📊 전체 통계 ".ljust(78) + "║")
    print("║" + "─" * 78 + "║")
    print(f"║   총 생성: {total_gen:6d}개 │ 총 검증: {total_val:6d}개 │ 총 누락 문제: {total_missing:3d}개".ljust(78) + "║")
    
    if total_gen > 0:
        val_rate = (total_val / total_gen) * 100
        print(f"║   검증률: {val_rate:.1f}%".ljust(78) + "║")
    
    print("╚" + "═" * 78 + "╝")
    print()


def get_retry_list(results: List[Dict]) -> List[Dict]:
    """재생성 필요 문제 목록 생성"""
    retry_list = []
    
    for result in results:
        source = result["source"]
        
        for qtype in ["subjectives", "multiples"]:
            for missing in result[qtype]["missing"]:
                retry_list.append({
                    "source": source,
                    "question_type": qtype,
                    "problem_idx": missing["problem_idx"],
                    "total_generated": missing["generated"]
                })
    
    return retry_list


def save_retry_queue(retry_list: List[Dict], output_path: str):
    """재생성 큐 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in retry_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"재생성 큐 저장: {output_path} ({len(retry_list)}개 문제)")


def main():
    parser = argparse.ArgumentParser(
        description="SFT 데이터 상태 확인 대시보드",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output_dir", type=str, default="./sft_output",
                        help="출력 디렉토리 (기본: ./sft_output)")
    parser.add_argument("--expected_n", type=int, default=10,
                        help="문제당 예상 생성 횟수 (기본: 10)")
    parser.add_argument("--save_retry", action="store_true",
                        help="재생성 큐 파일 저장 (.retry_queue.jsonl)")
    parser.add_argument("--json", action="store_true",
                        help="JSON 형식으로 출력")
    
    args = parser.parse_args()
    
    # source 찾기
    sources = find_sources(args.output_dir)
    
    if not sources:
        print(f"데이터가 없습니다: {args.output_dir}")
        return
    
    # 분석
    results = []
    for source in sources:
        result = analyze_source(args.output_dir, source, args.expected_n)
        results.append(result)
    
    # 출력
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_dashboard(results, args.expected_n)
    
    # 재생성 큐 저장
    retry_list = get_retry_list(results)
    
    if args.save_retry and retry_list:
        retry_path = os.path.join(args.output_dir, ".retry_queue.jsonl")
        save_retry_queue(retry_list, retry_path)
    elif args.save_retry:
        print("재생성 필요한 문제가 없습니다.")
    
    # 요약
    if retry_list and not args.json:
        print(f"💡 재생성 필요: {len(retry_list)}개 문제")
        print(f"   → python check_status.py --save_retry 로 retry_queue 생성")
        print(f"   → ./run_sft_pipeline.sh --validate_and_retry 로 재생성")


if __name__ == "__main__":
    main()
