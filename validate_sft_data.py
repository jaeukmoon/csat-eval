"""
SFT 데이터 검증 스크립트
생성된 SFT 데이터에서 정답 여부를 확인하고 필터링합니다.
"""
import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any


def open_jsonl(path: str) -> List[Dict]:
    """JSONL 파일을 읽어 리스트로 반환"""
    data = []
    with open(path, mode='r', encoding='utf8') as rf:
        for line in rf:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def to_jsonl(out_path: str, data: List[Dict]) -> None:
    """데이터를 JSONL 형식으로 저장"""
    with open(out_path, mode='w', encoding='utf8') as wf:
        for row in data:
            wf.write(json.dumps(row, ensure_ascii=False))
            wf.write('\n')


# 정규식 패턴들
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
BOXED_NESTED_RE = re.compile(r"\\boxed\{(.+)\}", re.DOTALL)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    텍스트에서 \boxed{} 안의 답을 추출합니다.
    여러 개가 있으면 마지막 것을 반환합니다.
    """
    if not text:
        return None
    
    # 간단한 패턴 먼저 시도
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    
    # 중첩된 괄호가 있는 경우
    matches = BOXED_NESTED_RE.findall(text)
    if matches:
        # 가장 마지막 매치에서 괄호 균형을 맞춰 추출
        return matches[-1].strip()
    
    return None


def normalize_latex(text: str) -> str:
    """
    LaTeX 표현을 정규화합니다.
    - \\frac{a}{b} → a/b
    - \\sqrt{x} → sqrt(x)
    - 기타 LaTeX 명령어 단순화
    """
    if not text:
        return ""
    
    result = text
    
    # \frac{a}{b} → a/b
    # 중첩되지 않은 간단한 분수 처리
    frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    while re.search(frac_pattern, result):
        result = re.sub(frac_pattern, r"(\1)/(\2)", result)
    
    # \sqrt{x} → sqrt(x)
    result = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", result)
    
    # \sqrt[n]{x} → x^(1/n)
    result = re.sub(r"\\sqrt\[([^\]]+)\]\{([^{}]+)\}", r"(\2)^(1/\1)", result)
    
    # \cdot, \times → *
    result = re.sub(r"\\cdot|\\times", "*", result)
    
    # \div → /
    result = re.sub(r"\\div", "/", result)
    
    # \pm → +-
    result = re.sub(r"\\pm", "+-", result)
    
    # \pi → pi
    result = re.sub(r"\\pi", "pi", result)
    
    # \log, \ln, \sin, \cos, \tan 등 함수 이름에서 \ 제거
    result = re.sub(r"\\(log|ln|sin|cos|tan|exp)", r"\1", result)
    
    # \left, \right 제거
    result = re.sub(r"\\left|\\right", "", result)
    
    # \{ \} → ( )
    result = re.sub(r"\\{|\\}", "", result)
    
    # $ 제거 (수식 구분자)
    result = result.replace("$", "")
    
    # 백슬래시 제거 (남은 LaTeX 명령어)
    result = re.sub(r"\\[a-zA-Z]+", "", result)
    
    # 공백 정리
    result = re.sub(r"\s+", "", result)
    
    return result.lower()


def normalize_answer(answer: str) -> str:
    """
    답을 정규화합니다 (비교를 위해).
    - LaTeX 표현 정규화 (\\frac{1}{2} → (1)/(2))
    - 공백 제거
    - 숫자만 추출 (가능한 경우)
    """
    if answer is None:
        return ""
    
    answer = str(answer).strip()
    
    # LaTeX 정규화 먼저 적용
    normalized = normalize_latex(answer)
    
    # 숫자만 있는 경우 정수로 변환
    try:
        # 정수 변환 시도
        return str(int(float(normalized)))
    except ValueError:
        pass
    
    # 분수 형태인지 확인하고 계산 시도 (예: (1)/(2) → 0.5)
    # 간단한 정수 분수만 처리
    frac_match = re.match(r"^\(?(\d+)\)?/\(?(\d+)\)?$", normalized)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            # 정수로 나누어 떨어지면 정수로
            if num % den == 0:
                return str(num // den)
            # 아니면 분수 형태 유지 (최대공약수로 약분)
            from math import gcd
            g = gcd(num, den)
            return f"{num//g}/{den//g}"
    
    # 그 외에는 정규화된 문자열 반환
    return normalized


def check_answer(extracted: Optional[str], ground_truth: Any) -> bool:
    """추출된 답과 정답을 비교합니다."""
    if extracted is None or ground_truth is None:
        return False
    
    norm_extracted = normalize_answer(extracted)
    norm_truth = normalize_answer(str(ground_truth))
    
    return norm_extracted == norm_truth


def get_assistant_content(item: Dict) -> Optional[str]:
    """
    다양한 형식에서 assistant 응답을 추출합니다.
    - sharegpt: messages[1]["content"]
    - simple: solution
    - alpaca: output
    """
    # sharegpt 형식
    if "messages" in item:
        messages = item["messages"]
        for msg in messages:
            if msg.get("role") == "assistant":
                return msg.get("content")
    
    # simple 형식
    if "solution" in item:
        return item["solution"]
    
    # alpaca 형식
    if "output" in item:
        return item["output"]
    
    return None


def validate_item(item: Dict) -> Dict:
    """
    단일 항목을 검증하고 결과를 반환합니다.
    """
    ground_truth = item.get("answer")
    assistant_content = get_assistant_content(item)
    
    extracted = extract_boxed_answer(assistant_content) if assistant_content else None
    is_correct = check_answer(extracted, ground_truth)
    
    return {
        "item": item,
        "extracted_answer": extracted,
        "ground_truth": ground_truth,
        "is_correct": is_correct
    }


def validate_file(input_path: str, output_path: str = None, 
                  correct_only: bool = True) -> Dict[str, Any]:
    """
    JSONL 파일을 검증하고 결과를 저장합니다.
    
    Args:
        input_path: 입력 JSONL 파일 경로
        output_path: 출력 JSONL 파일 경로 (None이면 저장 안함)
        correct_only: True면 정답만 저장, False면 모두 저장 (검증 결과 포함)
    
    Returns:
        통계 정보 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"검증 중: {input_path}")
    print(f"{'='*60}")
    
    data = open_jsonl(input_path)
    print(f"총 {len(data)}개 항목 로드됨")
    
    results = []
    correct_count = 0
    no_answer_count = 0
    no_boxed_count = 0
    
    for item in data:
        validation = validate_item(item)
        
        if validation["ground_truth"] is None:
            no_answer_count += 1
        elif validation["extracted_answer"] is None:
            no_boxed_count += 1
        
        if validation["is_correct"]:
            correct_count += 1
            if correct_only:
                results.append(item)
        else:
            if not correct_only:
                # 검증 결과를 포함하여 저장
                item_with_validation = item.copy()
                item_with_validation["_validation"] = {
                    "extracted_answer": validation["extracted_answer"],
                    "is_correct": False
                }
                results.append(item_with_validation)
    
    # 통계
    total = len(data)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    stats = {
        "input_file": input_path,
        "total": total,
        "correct": correct_count,
        "incorrect": total - correct_count,
        "no_answer": no_answer_count,
        "no_boxed": no_boxed_count,
        "accuracy": accuracy
    }
    
    print(f"\n결과:")
    print(f"  - 총 항목: {total}")
    print(f"  - 정답: {correct_count} ({accuracy:.2f}%)")
    print(f"  - 오답: {total - correct_count}")
    print(f"  - 정답 필드 없음: {no_answer_count}")
    print(f"  - boxed 없음: {no_boxed_count}")
    
    # 결과 저장
    if output_path and results:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        to_jsonl(output_path, results)
        print(f"\n저장됨: {output_path} ({len(results)}개)")
    
    return stats


def validate_directory(input_dir: str, output_dir: str = None,
                       correct_only: bool = True, pattern: str = "*.jsonl") -> List[Dict]:
    """
    디렉토리 내의 모든 JSONL 파일을 검증합니다.
    """
    import glob
    
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        print(f"파일을 찾을 수 없음: {input_dir}/{pattern}")
        return []
    
    print(f"총 {len(files)}개 파일 발견")
    
    all_stats = []
    
    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        
        if output_dir:
            output_path = os.path.join(output_dir, f"validated_{filename}")
        else:
            output_path = None
        
        stats = validate_file(file_path, output_path, correct_only)
        all_stats.append(stats)
    
    # 전체 통계 출력
    if all_stats:
        total_items = sum(s["total"] for s in all_stats)
        total_correct = sum(s["correct"] for s in all_stats)
        overall_accuracy = (total_correct / total_items * 100) if total_items > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"전체 통계:")
        print(f"{'='*60}")
        print(f"  - 총 파일: {len(all_stats)}")
        print(f"  - 총 항목: {total_items}")
        print(f"  - 총 정답: {total_correct}")
        print(f"  - 전체 정답률: {overall_accuracy:.2f}%")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="SFT 데이터 검증기")
    parser.add_argument("--input", required=True, type=str,
                        help="입력 JSONL 파일 또는 디렉토리")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 경로 (파일 또는 디렉토리)")
    parser.add_argument("--all", action="store_true",
                        help="정답/오답 모두 저장 (기본: 정답만)")
    parser.add_argument("--pattern", type=str, default="*.jsonl",
                        help="파일 패턴 (디렉토리 모드에서 사용)")
    
    args = parser.parse_args()
    
    correct_only = not args.all
    
    if os.path.isdir(args.input):
        validate_directory(args.input, args.output, correct_only, args.pattern)
    else:
        validate_file(args.input, args.output, correct_only)
    
    print("\n완료!")


if __name__ == "__main__":
    main()
