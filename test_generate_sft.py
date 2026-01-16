"""
SFT 데이터 생성 테스트 스크립트
2025_math.jsonl만 사용하여 각 문제당 1번만 생성하여 검증합니다.
"""
import os
import json
import re
from openai import OpenAI

# ============================================================================
# 유틸리티 함수 (generate_sft_data.py에서 복사)
# ============================================================================

def open_jsonl(path):
    """JSONL 파일을 읽어 리스트로 반환"""
    data = []
    with open(path, mode='r', encoding='utf8') as rf:
        for line in rf:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def is_multiple_choice(problem_text: str) -> bool:
    """객관식 문제인지 판별합니다."""
    # 선택지 패턴 확인: \item[1], \item[2], ... 등이 5개 이상 있는지
    choice_pattern = r"\\item\[[1-5]\]"
    matches = re.findall(choice_pattern, problem_text)
    return len(matches) >= 5


def extract_choice_value(problem_text: str, choice_num: int) -> str:
    """
    객관식 문제에서 특정 번호 선택지의 값을 추출합니다.
    
    예: \\item[2] \\frac{1}{2} → choice_num=2 → "\\frac{1}{2}"
    """
    pattern = rf"\\item\[{choice_num}\]\s*(.+?)(?=\\item\[|\\end\{{itemize\}}|$)"
    match = re.search(pattern, problem_text, re.DOTALL)
    if match:
        value = match.group(1).strip()
        value = re.sub(r'\s+', ' ', value)
        return value
    return str(choice_num)


def clean_problem_text(problem_text: str, remove_mc_choices: bool = False) -> str:
    """
    문제 텍스트에서 번호와 점수를 제거합니다.
    
    제거 패턴:
    - 문제 번호: "1. ", "12. " 등 (문자열 시작 부분)
    - 점수 표시: "[2점]", "[3점]", "[4점]" 등
    
    Args:
        remove_mc_choices: True면 객관식 선택지도 제거
    """
    text = problem_text.strip()
    
    # 문제 번호 제거 (시작 부분의 "숫자. " 패턴)
    text = re.sub(r'^(\d+)\.\s*', '', text)
    
    # 점수 표시 제거 ("[2점]", "[3점]" 등)
    text = re.sub(r'\s*\[\d+점\]\s*', ' ', text)
    
    # 객관식 선택지 제거 (옵션)
    if remove_mc_choices:
        # LaTeX itemize 환경 제거
        text = re.sub(r'\\begin\{itemize\}.*?\\end\{itemize\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\\\begin\{itemize\}.*?\\\\end\{itemize\}', '', text, flags=re.DOTALL)
    
    # 연속 공백 정리
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


def get_prompt(problem_text: str, request_sentences: list, generation_id: int = 0,
               as_subjective: bool = False) -> str:
    """
    문제와 boxed 요청 문장을 조합하여 프롬프트를 생성합니다.
    """
    cleaned_problem = clean_problem_text(problem_text, remove_mc_choices=as_subjective)
    
    hash_code = hash(cleaned_problem) + generation_id
    sent = request_sentences[hash_code % len(request_sentences)]['sent']
    
    variant = hash_code % 4
    if variant == 0:
        prompt = "\n".join([cleaned_problem, sent])
    elif variant == 1:
        prompt = "\n\n".join([cleaned_problem, sent])
    elif variant == 2:
        prompt = "\n".join([sent, cleaned_problem])
    else:
        prompt = "\n\n".join([sent, cleaned_problem])
    
    return prompt


# ============================================================================
# OpenAI API 호출
# ============================================================================

def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """OpenAI API를 호출하여 응답을 받습니다."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        top_p=1.0,
    )
    return completion.choices[0].message.content


# ============================================================================
# 메인 테스트
# ============================================================================

def load_dotenv_local() -> None:
    """.env.local, env.local, .env 파일에서 환경변수 로드"""
    for fname in (".env.local", "env.local", ".env"):
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            continue


def main():
    import argparse
    
    # 환경변수 로드 (.env.local 등)
    load_dotenv_local()
    
    parser = argparse.ArgumentParser(description="SFT 데이터 생성 테스트")
    parser.add_argument("--data_dir", default="./data", type=str,
                        help="데이터 디렉토리")
    parser.add_argument("--model", default="gpt-4o-mini", type=str,
                        help="OpenAI 모델 이름")
    parser.add_argument("--file", default="2025_math.jsonl", type=str,
                        help="테스트할 파일 이름")
    parser.add_argument("--max_problems", default=5, type=int,
                        help="테스트할 최대 문제 수 (0이면 전체)")
    
    args = parser.parse_args()
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요. (.env.local 파일에 넣어주세요)")
    
    # boxed 요청 문장 로드
    sentences_path = os.path.join(args.data_dir, "sentences_ask_boxed.jsonl")
    if not os.path.exists(sentences_path):
        raise FileNotFoundError(f"sentences_ask_boxed.jsonl not found: {sentences_path}")
    request_sentences = open_jsonl(sentences_path)
    print(f"Loaded {len(request_sentences)} request sentences")
    
    # 수학 파일 로드
    math_file = os.path.join(args.data_dir, args.file)
    if not os.path.exists(math_file):
        raise FileNotFoundError(f"File not found: {math_file}")
    
    problems = open_jsonl(math_file)
    print(f"\nLoaded {len(problems)} problems from {args.file}")
    
    # 문제 수 제한
    if args.max_problems > 0:
        problems = problems[:args.max_problems]
        print(f"  - 테스트할 문제 수: {len(problems)}개 (제한됨)")
    
    # 통계 출력
    mc_count = sum(1 for p in problems if is_multiple_choice(p['problem']))
    sc_count = len(problems) - mc_count
    print(f"  - 객관식 문제: {mc_count}개")
    print(f"  - 주관식 문제: {sc_count}개")
    
    # 각 문제당 1번씩 주관식 버전 생성
    print(f"\n{'='*60}")
    print(f"주관식 버전 테스트 (각 문제당 1번 생성)")
    print(f"{'='*60}\n")
    
    subjective_results = []
    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] Problem {problem['id']} ({problem['name']})")
        print(f"  - 원본: {'객관식' if is_multiple_choice(problem['problem']) else '주관식'}")
        
        # 주관식 버전 프롬프트 생성
        prompt = get_prompt(problem['problem'], request_sentences, generation_id=0, as_subjective=True)
        
        print(f"  - 프롬프트 길이: {len(prompt)} chars")
        print(f"  - 프롬프트 미리보기:\n    {prompt[:200]}...\n")
        
        # 주관식 버전이면서 원본이 객관식인 경우, 실제 답 값을 추출
        original_answer = problem.get('answer')
        if is_multiple_choice(problem['problem']) and original_answer is not None:
            real_answer = extract_choice_value(problem['problem'], original_answer)
            print(f"  - 원본 답: {original_answer}번 → 실제 값: {real_answer}")
        else:
            real_answer = original_answer
        
        try:
            solution = call_openai(prompt, model=args.model)
            print(f"  - 생성 완료 (응답 길이: {len(solution)} chars)")
            print(f"  - 응답 미리보기: {solution[:150]}...\n")
            
            subjective_results.append({
                "problem_id": problem['id'],
                "problem_name": problem['name'],
                "is_mc": is_multiple_choice(problem['problem']),
                "prompt": prompt,
                "solution": solution,
                "original_answer": original_answer,  # 원본 선택지 번호
                "real_answer": real_answer,  # 실제 답 값
            })
        except Exception as e:
            print(f"  - ERROR: {e}\n")
            subjective_results.append({
                "problem_id": problem['id'],
                "problem_name": problem['name'],
                "is_mc": is_multiple_choice(problem['problem']),
                "error": str(e),
            })
    
    # 각 문제당 1번씩 객관식 버전 생성
    print(f"\n{'='*60}")
    print(f"객관식 버전 테스트 (각 문제당 1번 생성)")
    print(f"{'='*60}\n")
    
    multiple_results = []
    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] Problem {problem['id']} ({problem['name']})")
        print(f"  - 원본: {'객관식' if is_multiple_choice(problem['problem']) else '주관식'}")
        
        # 객관식 버전 프롬프트 생성 (선택지 유지)
        prompt = get_prompt(problem['problem'], request_sentences, generation_id=0, as_subjective=False)
        
        print(f"  - 프롬프트 길이: {len(prompt)} chars")
        print(f"  - 프롬프트 미리보기:\n    {prompt[:200]}...\n")
        
        try:
            solution = call_openai(prompt, model=args.model)
            print(f"  - 생성 완료 (응답 길이: {len(solution)} chars)")
            print(f"  - 응답 미리보기: {solution[:150]}...\n")
            
            multiple_results.append({
                "problem_id": problem['id'],
                "problem_name": problem['name'],
                "is_mc": is_multiple_choice(problem['problem']),
                "prompt": prompt,
                "solution": solution,
                "answer": problem.get('answer'),
            })
        except Exception as e:
            print(f"  - ERROR: {e}\n")
            multiple_results.append({
                "problem_id": problem['id'],
                "problem_name": problem['name'],
                "is_mc": is_multiple_choice(problem['problem']),
                "error": str(e),
            })
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("테스트 결과 요약")
    print(f"{'='*60}")
    print(f"\n주관식 버전:")
    print(f"  - 성공: {sum(1 for r in subjective_results if 'solution' in r)}개")
    print(f"  - 실패: {sum(1 for r in subjective_results if 'error' in r)}개")
    
    print(f"\n객관식 버전:")
    print(f"  - 성공: {sum(1 for r in multiple_results if 'solution' in r)}개")
    print(f"  - 실패: {sum(1 for r in multiple_results if 'error' in r)}개")
    
    # 검증: 객관식 문제 판별이 올바른지 확인
    print(f"\n{'='*60}")
    print("객관식 판별 검증")
    print(f"{'='*60}")
    
    mc_problems = [p for p in problems if is_multiple_choice(p['problem'])]
    sc_problems = [p for p in problems if not is_multiple_choice(p['problem'])]
    
    print(f"\n객관식으로 판별된 문제 ({len(mc_problems)}개):")
    for p in mc_problems[:5]:  # 처음 5개만 출력
        print(f"  - {p['id']}: {p['name']} (선택지 패턴 확인됨)")
    
    print(f"\n주관식으로 판별된 문제 ({len(sc_problems)}개):")
    for p in sc_problems[:5]:  # 처음 5개만 출력
        print(f"  - {p['id']}: {p['name']}")
    
    # 결과 저장 (선택적)
    output_file = "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "subjective_results": subjective_results,
            "multiple_results": multiple_results,
            "summary": {
                "total_problems": len(problems),
                "mc_count": mc_count,
                "sc_count": sc_count,
                "subjective_success": sum(1 for r in subjective_results if 'solution' in r),
                "subjective_failed": sum(1 for r in subjective_results if 'error' in r),
                "multiple_success": sum(1 for r in multiple_results if 'solution' in r),
                "multiple_failed": sum(1 for r in multiple_results if 'error' in r),
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()
