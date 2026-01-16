"""
SFT 학습 데이터 생성 스크립트
수학 수능 문제에 대한 풀이를 vLLM으로 생성하여 SFT 학습 데이터를 만듭니다.
"""
import os
import re
import json
import time
import glob
import aiohttp
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# no_proxy 설정 (vLLM 서버 주소)
os.environ["no_proxy"] = "localhost,127.0.0.1,10.0.74.208"

# ============================================================================
# 유틸리티 함수
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


def to_jsonl(out_path, data):
    """데이터를 JSONL 형식으로 저장"""
    with open(out_path, mode='w', encoding='utf8') as wf:
        for row in data:
            wf.write(json.dumps(row, ensure_ascii=False))
            wf.write('\n')


def append_jsonl(out_path, row):
    """단일 항목을 JSONL 파일에 추가"""
    with open(out_path, mode='a', encoding='utf8') as wf:
        wf.write(json.dumps(row, ensure_ascii=False))
        wf.write('\n')


# ============================================================================
# 문제 전처리 함수
# ============================================================================

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
    
    Args:
        problem_text: 문제 텍스트
        choice_num: 선택지 번호 (1-5)
    
    Returns:
        선택지 값 (추출 실패 시 원본 choice_num을 문자열로 반환)
    """
    # \item[N] 다음의 값을 추출 (다음 \item이나 \end{itemize} 전까지)
    pattern = rf"\\item\[{choice_num}\]\s*(.+?)(?=\\item\[|\\end\{{itemize\}}|$)"
    match = re.search(pattern, problem_text, re.DOTALL)
    if match:
        value = match.group(1).strip()
        # 줄바꿈 제거
        value = re.sub(r'\s+', ' ', value)
        return value
    return str(choice_num)


def remove_choices(problem_text: str) -> str:
    """
    객관식 문제에서 선택지를 제거하여 주관식으로 변환합니다.
    
    제거 패턴:
    - \\begin{itemize} ... \\end{itemize} 블록 전체
    """
    text = problem_text
    
    # LaTeX itemize 환경 제거 (여러 패턴 시도)
    # 패턴 1: \begin{itemize} ... \end{itemize}
    text = re.sub(r'\\begin\{itemize\}.*?\\end\{itemize\}', '', text, flags=re.DOTALL)
    
    # 패턴 2: \\begin{itemize} ... \\end{itemize} (이스케이프된 버전)
    text = re.sub(r'\\\\begin\{itemize\}.*?\\\\end\{itemize\}', '', text, flags=re.DOTALL)
    
    # 선택지 번호 패턴 제거 (1) (2) (3) (4) (5) 또는 ① ② ③ ④ ⑤
    text = re.sub(r'\s*\([1-5]\)\s*[^\(\n]*', '', text)
    text = re.sub(r'\s*[①②③④⑤]\s*[^\n]*', '', text)
    
    # 연속 공백/줄바꿈 정리
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


def clean_problem_text(problem_text: str, remove_mc_choices: bool = False) -> str:
    """
    문제 텍스트에서 번호와 점수를 제거합니다.
    
    제거 패턴:
    - 문제 번호: "1. ", "12. " 등 (문자열 시작 부분)
    - 점수 표시: "[2점]", "[3점]", "[4점]" 등
    
    Args:
        remove_mc_choices: True면 객관식 선택지도 제거
    
    LaTeX 형식은 유지합니다.
    """
    text = problem_text.strip()
    
    # 문제 번호 제거 (시작 부분의 "숫자. " 패턴)
    text = re.sub(r'^(\d+)\.\s*', '', text)
    
    # 점수 표시 제거 ("[2점]", "[3점]" 등)
    text = re.sub(r'\s*\[\d+점\]\s*', ' ', text)
    
    # 객관식 선택지 제거 (옵션)
    if remove_mc_choices:
        text = remove_choices(text)
    
    # 연속 공백 정리
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


# ============================================================================
# 프롬프트 생성
# ============================================================================

def get_prompt(problem_text: str, request_sentences: list, generation_id: int = 0,
               as_subjective: bool = False) -> str:
    """
    문제와 boxed 요청 문장을 조합하여 프롬프트를 생성합니다.
    
    Args:
        problem_text: 원본 문제 텍스트
        request_sentences: boxed 요청 문장 리스트
        generation_id: 생성 인덱스 (프롬프트 변형에 사용)
        as_subjective: True면 객관식 선택지를 제거하여 주관식으로 변환
    """
    # 문제 전처리 (번호/점수 제거, 선택지 제거 옵션)
    cleaned_problem = clean_problem_text(problem_text, remove_mc_choices=as_subjective)
    
    # 해시 + generation_id로 문장 선택
    hash_code = hash(cleaned_problem) + generation_id
    sent = request_sentences[hash_code % len(request_sentences)]['sent']
    
    # 배치 방식 결정 (4가지 변형)
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
# vLLM API 호출
# ============================================================================

def send_msg(prompt: str, base_url: str, model: str = "gpt-oss-120b"):
    """vLLM 서버에 요청을 보내고 응답을 받습니다."""
    client = OpenAI(base_url=base_url, api_key="dummy")
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        reasoning_effort='high',
        temperature=1.0,
        top_p=1.0,
        timeout=60 * 60 * 24  # 24시간 타임아웃
    )
    return completion


# ============================================================================
# 출력 형식 변환
# ============================================================================

def format_output(problem: str, solution: str, answer: int, source: str, 
                  generation_id: int, format_type: str = "simple",
                  prompt: str = None) -> dict:
    """
    결과를 지정된 형식으로 변환합니다.
    
    지원 형식:
    - simple: 간단한 problem/solution 형식
    - sharegpt: ShareGPT 대화 형식
    - alpaca: Alpaca instruction 형식
    
    Args:
        prompt: get_prompt()로 생성된 실제 프롬프트 (sharegpt/alpaca에서 사용)
    """
    cleaned_problem = clean_problem_text(problem)
    
    if format_type == "simple":
        return {
            "problem": cleaned_problem,
            "solution": solution,
            "answer": answer,
            "source": source,
            "generation_id": generation_id
        }
    
    elif format_type == "sharegpt":
        # prompt가 제공되면 그대로 사용, 아니면 cleaned_problem 사용
        human_message = prompt if prompt else cleaned_problem
        return {
            "messages": [
                {"role": "user", "content": human_message},
                {"role": "assistant", "content": solution}
            ],
            "answer": answer
        }
    
    elif format_type == "alpaca":
        # prompt가 제공되면 그대로 사용
        if prompt:
            return {
                "instruction": prompt,
                "input": "",
                "output": solution,
                "answer": answer,
                "source": source,
                "generation_id": generation_id
            }
        else:
            return {
                "instruction": "다음 수학 문제를 풀고, 최종 답을 \\boxed{} 안에 넣어주세요.",
                "input": cleaned_problem,
                "output": solution,
                "answer": answer,
                "source": source,
                "generation_id": generation_id
            }
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


# ============================================================================
# 항목 처리
# ============================================================================

def process_item(idx: tuple, problems: list, request_sentences: list, 
                 output_dir: str, base_url: str, model: str, 
                 source: str, format_type: str, question_type: str = "multiples"):
    """
    단일 문제-생성 쌍을 처리합니다.
    
    Args:
        idx: (문제 인덱스, 생성 인덱스)
        question_type: "multiples" (객관식) 또는 "subjectives" (주관식)
    """
    problem_idx, gen_idx = idx
    output_path = f"{output_dir}/{problem_idx}_{gen_idx}.jsonl"
    
    # 이미 처리된 경우 스킵
    if os.path.exists(output_path):
        return None
    
    req_start = time.time()
    start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"SEND [{problem_idx}_{gen_idx}] ({question_type}) | st={start_stamp}")
    
    MAX_RETRIES = 10
    BACKOFF_FACTOR = 2
    
    item = problems[problem_idx]
    # 주관식이면 선택지 제거
    as_subjective = (question_type == "subjectives")
    prompt = get_prompt(item['problem'], request_sentences, gen_idx, as_subjective=as_subjective)
    
    resp = None
    trial = 0
    for trial in range(MAX_RETRIES):
        try:
            resp = send_msg(prompt, base_url, model)
            if not resp:
                raise ValueError("ERROR WITHOUT RESPONSE")
            break
        except aiohttp.ClientError as e:
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{problem_idx}_{gen_idx}] RETRY {trial + 1}/{MAX_RETRIES} | ERROR: {e}")
            if trial < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** trial)
        except asyncio.TimeoutError as e:
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{problem_idx}_{gen_idx}] TIMEOUT RETRY {trial + 1}/{MAX_RETRIES} | ERROR: {e}")
            if trial < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** trial)
        except Exception as e:
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{problem_idx}_{gen_idx}] EXCEPTION: {e}")
            return None
    
    if resp is None:
        print(f"[{problem_idx}_{gen_idx}] FAILED after {MAX_RETRIES} retries")
        return None
    
    # 응답 추출
    solution = resp.choices[0].message.content
    answer = item.get('answer', None)
    
    # 주관식 버전이면서 원본이 객관식인 경우, 실제 답 값을 추출
    # (예: answer=2, 2번 선택지가 "\frac{1}{2}"이면 → real_answer="\frac{1}{2}")
    if as_subjective and is_multiple_choice(item['problem']) and answer is not None:
        real_answer = extract_choice_value(item['problem'], answer)
    else:
        real_answer = answer
    
    # 형식에 맞게 변환 (생성된 프롬프트도 전달)
    formatted = format_output(
        problem=item['problem'],
        solution=solution,
        answer=real_answer,  # 주관식은 실제 답 값, 객관식은 선택지 번호
        source=source,
        generation_id=gen_idx,
        format_type=format_type,
        prompt=prompt  # boxed 문장이 포함된 실제 프롬프트
    )
    
    # 저장
    to_jsonl(output_path, [formatted])
    
    end_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    req_duration = time.time() - req_start
    
    print(f"DONE [{problem_idx}_{gen_idx}] | time={req_duration:.2f}s | trials={trial + 1}")
    return formatted


def run_generation(problems: list, request_sentences: list, output_dir: str,
                   base_url: str, model: str, source: str, format_type: str,
                   n: int = 10, max_workers: int = 200, question_type: str = "multiples",
                   retry_problems: list = None):
    """
    모든 문제에 대해 n번씩 풀이를 생성합니다.
    
    Args:
        question_type: "multiples" (객관식) 또는 "subjectives" (주관식)
        retry_problems: 재생성할 문제 인덱스 목록 (None이면 모든 문제 처리)
    """
    inputs = []
    
    if retry_problems is not None:
        # 재생성 모드: 지정된 문제만 처리
        for problem_idx in retry_problems:
            for j in range(n):
                inputs.append((problem_idx, j))
        print(f"Retry mode: {len(inputs)} tasks ({len(retry_problems)} problems x {n} generations) [{question_type}]")
    else:
        # 일반 모드: 모든 문제 처리
        for i in range(len(problems)):
            for j in range(n):
                inputs.append((i, j))
        print(f"Total tasks: {len(inputs)} ({len(problems)} problems x {n} generations) [{question_type}]")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_item, idx, problems, request_sentences, 
                output_dir, base_url, model, source, format_type, question_type
            ) 
            for idx in inputs
        ]
        for future in futures:
            future.result()


# ============================================================================
# 결과 병합
# ============================================================================

def merge_results(input_dirs: list, output_path: str):
    """
    여러 디렉토리의 결과를 하나의 JSONL 파일로 병합합니다.
    """
    all_data = []
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Warning: Directory not found: {input_dir}")
            continue
        
        for entry in os.scandir(input_dir):
            if entry.name.endswith('.jsonl'):
                try:
                    data = open_jsonl(entry.path)
                    all_data.extend(data)
                except Exception as e:
                    print(f"Error reading {entry.path}: {e}")
    
    print(f"Total merged: {len(all_data)} items")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    to_jsonl(output_path, all_data)
    print(f"Saved to: {output_path}")
    
    return all_data


def find_math_files(data_dir: str) -> list:
    """data 디렉토리에서 수학 JSONL 파일들을 찾습니다."""
    pattern = os.path.join(data_dir, "*_math.jsonl")
    files = glob.glob(pattern)
    return sorted(files)


# ============================================================================
# 메인
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT 학습 데이터 생성기")
    parser.add_argument("--data_dir", default="./data", type=str,
                        help="수학 데이터가 있는 디렉토리")
    parser.add_argument("--output_dir", default="./sft_output", type=str,
                        help="출력 디렉토리")
    parser.add_argument("--n", default=10, type=int,
                        help="문제당 생성 횟수")
    parser.add_argument("--worker", default=20, type=int,
                        help="동시 워커 수")
    parser.add_argument("--format", default="simple", type=str,
                        choices=["simple", "sharegpt", "alpaca"],
                        help="출력 형식")
    parser.add_argument("--base_url", 
                        default="http://10.0.74.208:8000/v1",
                        type=str, help="vLLM 서버 URL")
    parser.add_argument("--model", default="glm-4.7", type=str,
                        help="모델 이름")
    parser.add_argument("--merge_only", action="store_true",
                        help="생성 없이 기존 결과만 병합")
    parser.add_argument("--input_file", type=str, default=None,
                        help="특정 파일만 처리 (지정하지 않으면 모든 수학 파일 처리)")
    parser.add_argument("--retry_file", type=str, default=None,
                        help="재생성할 문제 목록 파일 (retry_queue.jsonl)")
    
    args = parser.parse_args()
    
    # boxed 요청 문장 로드
    sentences_path = os.path.join(args.data_dir, "sentences_ask_boxed.jsonl")
    if not os.path.exists(sentences_path):
        raise FileNotFoundError(f"sentences_ask_boxed.jsonl not found: {sentences_path}")
    request_sentences = open_jsonl(sentences_path)
    
    # 처리할 수학 파일 목록
    if args.input_file:
        math_files = [args.input_file]
    else:
        math_files = find_math_files(args.data_dir)
    
    if not math_files:
        raise FileNotFoundError(f"No math JSONL files found in {args.data_dir}")
    
    print(f"Found {len(math_files)} math files: {[os.path.basename(f) for f in math_files]}")
    
    # 재생성 파일 로드
    retry_queue = {}  # {(source, question_type): [problem_indices]}
    if args.retry_file and os.path.exists(args.retry_file):
        retry_items = open_jsonl(args.retry_file)
        for item in retry_items:
            key = (item["source"], item["question_type"])
            if key not in retry_queue:
                retry_queue[key] = []
            retry_queue[key].append(item["problem_idx"])
        print(f"\n재생성 모드: {len(retry_items)}개 문제 재생성 예정")
        for (src, qtype), indices in retry_queue.items():
            print(f"  - {src}/{qtype}: 문제 {indices}")
    
    result_dirs = []
    
    if not args.merge_only:
        # 각 파일 처리
        for file_path in math_files:
            file_name = os.path.basename(file_path)
            source = file_name.replace('.jsonl', '')
            
            print(f"\n{'='*60}")
            print(f"Processing: {file_name}")
            print(f"{'='*60}")
            
            problems = open_jsonl(file_path)
            print(f"Loaded {len(problems)} problems from {file_name}")
            
            # 통계 출력
            mc_count = sum(1 for p in problems if is_multiple_choice(p['problem']))
            print(f"  - 객관식 문제: {mc_count}개, 주관식 문제: {len(problems) - mc_count}개")
            
            # 주관식 버전 생성 (subjectives)
            subj_output_dir = os.path.join(args.output_dir, source, "subjectives")
            os.makedirs(subj_output_dir, exist_ok=True)
            result_dirs.append(subj_output_dir)
            
            # 재생성할 문제 목록 확인
            subj_retry = retry_queue.get((source, "subjectives"), None)
            
            # 재생성 모드일 경우 기존 파일 삭제
            if subj_retry:
                print(f"\n[주관식 재생성] 문제 {subj_retry} 기존 파일 삭제 중...")
                for problem_idx in subj_retry:
                    for gen_idx in range(args.n):
                        old_file = os.path.join(subj_output_dir, f"{problem_idx}_{gen_idx}.jsonl")
                        if os.path.exists(old_file):
                            os.remove(old_file)
                            print(f"  삭제: {old_file}")
            
            if subj_retry or not retry_queue:
                print(f"\n[주관식 버전 생성 시작] ({len(subj_retry) if subj_retry else len(problems)}개 문제)")
                run_generation(
                    problems=problems,
                    request_sentences=request_sentences,
                    output_dir=subj_output_dir,
                    base_url=args.base_url,
                    model=args.model,
                    source=source,
                    format_type=args.format,
                    n=args.n,
                    max_workers=args.worker,
                    question_type="subjectives",
                    retry_problems=subj_retry
                )
            
            # 객관식 버전 생성 (multiples)
            mc_output_dir = os.path.join(args.output_dir, source, "multiples")
            os.makedirs(mc_output_dir, exist_ok=True)
            result_dirs.append(mc_output_dir)
            
            # 재생성할 문제 목록 확인
            mc_retry = retry_queue.get((source, "multiples"), None)
            
            # 재생성 모드일 경우 기존 파일 삭제
            if mc_retry:
                print(f"\n[객관식 재생성] 문제 {mc_retry} 기존 파일 삭제 중...")
                for problem_idx in mc_retry:
                    for gen_idx in range(args.n):
                        old_file = os.path.join(mc_output_dir, f"{problem_idx}_{gen_idx}.jsonl")
                        if os.path.exists(old_file):
                            os.remove(old_file)
                            print(f"  삭제: {old_file}")
            
            if mc_retry or not retry_queue:
                print(f"\n[객관식 버전 생성 시작] ({len(mc_retry) if mc_retry else len(problems)}개 문제)")
                run_generation(
                    problems=problems,
                    request_sentences=request_sentences,
                    output_dir=mc_output_dir,
                    base_url=args.base_url,
                    model=args.model,
                    source=source,
                    format_type=args.format,
                    n=args.n,
                    max_workers=args.worker,
                    question_type="multiples",
                    retry_problems=mc_retry
                )
    else:
        # merge_only 모드: 기존 디렉토리 찾기
        for file_path in math_files:
            source = os.path.basename(file_path).replace('.jsonl', '')
            # multiples와 subjectives 폴더 모두 찾기
            for qtype in ["multiples", "subjectives"]:
                each_output_dir = os.path.join(args.output_dir, source, qtype)
                if os.path.exists(each_output_dir):
                    result_dirs.append(each_output_dir)
    
    # 결과 병합
    print(f"\n{'='*60}")
    print("Merging results...")
    print(f"{'='*60}")
    
    merged_dir = os.path.join(args.output_dir, "merged")
    merged_path = os.path.join(merged_dir, f"sft_math_all_{args.format}.jsonl")
    
    merge_results(result_dirs, merged_path)
    
    print("\nDONE!")


if __name__ == "__main__":
    main()