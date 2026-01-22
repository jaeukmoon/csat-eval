"""
SFT 학습 데이터 생성 스크립트 (로컬 모델 버전)
로컬 HuggingFace 모델을 사용하여 수학 수능 문제에 대한 풀이를 생성합니다.

사용법:
    python generate_sft_data_local.py --model_path /path/to/gpt-oss-12b [옵션]
"""
import os
import time
import glob
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# generate_sft_data.py에서 공통 함수 임포트
from generate_sft_data import (
    open_jsonl,
    to_jsonl,
    is_multiple_choice,
    extract_choice_value,
    clean_problem_text,
    get_prompt,
    format_output,
    merge_results,
    find_math_files,
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    🔧 기본 설정                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DEFAULT_DATA_DIR = "./data"
DEFAULT_OUTPUT_DIR = "./sft_output_local"
DEFAULT_MODEL_PATH = "/data/hf_models/gpt-oss-12b"
DEFAULT_N = 1
DEFAULT_FORMAT = "sharegpt"


# ============================================================================
# 로컬 모델 추론
# ============================================================================

def load_local_model(model_path: str):
    """로컬 모델과 토크나이저를 로드합니다."""
    print(f"[INFO] Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    
    print(f"[INFO] Model loaded successfully. Device: {next(model.parameters()).device}")
    return model, tokenizer


def generate_with_local_model(model, tokenizer, prompt: str, reasoning_effort: str = "high",
                               max_new_tokens: int = 10000, temperature: float = 1.0) -> str:
    """로컬 모델로 응답을 생성합니다."""
    messages = [{"role": "user", "content": prompt}]
    
    chat_kwargs = {}
    if reasoning_effort != "none":
        chat_kwargs["reasoning_effort"] = reasoning_effort
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **chat_kwargs
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return decoded


def parse_gpt_oss_output(decoded_output: str) -> str:
    """GPT-OSS 출력에서 assistant 응답을 추출합니다."""
    # final<|message|> 태그 이후가 최종 응답
    if 'final<|message|>' in decoded_output:
        parts = decoded_output.split('final<|message|>')
        if len(parts) >= 2:
            solution = parts[-1].strip()
            # reasoning 부분 추출 (있는 경우)
            if '<|reasoning|>' in decoded_output:
                reasoning_parts = decoded_output.split('<|reasoning|>')
                if len(reasoning_parts) >= 2:
                    reasoning = reasoning_parts[-1].split('final<|message|>')[0].strip()
                    solution = f"<think>\n{reasoning}\n</think>\n{solution}"
            return solution
    
    # 일반적인 assistant 응답 추출
    if '<|assistant|>' in decoded_output:
        parts = decoded_output.split('<|assistant|>')
        if len(parts) >= 2:
            return parts[-1].strip()
    
    return decoded_output


# ============================================================================
# 항목 처리
# ============================================================================

def process_item_local(idx: tuple, problems: list, request_sentences: list, 
                       output_dir: str, model, tokenizer, source: str, 
                       format_type: str, question_type: str = "multiples",
                       reasoning_effort: str = "high"):
    """단일 문제-생성 쌍을 로컬 모델로 처리합니다."""
    problem_idx, gen_idx = idx
    output_path = f"{output_dir}/{problem_idx}_{gen_idx}.jsonl"
    
    if os.path.exists(output_path):
        return None
    
    req_start = time.time()
    start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"SEND [{problem_idx}_{gen_idx}] ({question_type}) | st={start_stamp}")
    
    item = problems[problem_idx]
    as_subjective = (question_type == "subjectives")
    prompt = get_prompt(item['problem'], request_sentences, gen_idx, as_subjective=as_subjective)
    
    try:
        decoded = generate_with_local_model(model, tokenizer, prompt, reasoning_effort)
        solution = parse_gpt_oss_output(decoded)
    except Exception as e:
        print(f"[{problem_idx}_{gen_idx}] EXCEPTION: {e}")
        return None
    
    answer = item.get('answer', None)
    
    if as_subjective and is_multiple_choice(item['problem']) and answer is not None:
        real_answer = extract_choice_value(item['problem'], answer)
    else:
        real_answer = answer
    
    formatted = format_output(
        problem=item['problem'],
        solution=solution,
        answer=real_answer,
        source=source,
        generation_id=gen_idx,
        format_type=format_type,
        prompt=prompt
    )
    
    to_jsonl(output_path, [formatted])
    
    req_duration = time.time() - req_start
    print(f"DONE [{problem_idx}_{gen_idx}] | time={req_duration:.2f}s")
    return formatted


def run_generation_local(problems: list, request_sentences: list, output_dir: str,
                          model, tokenizer, source: str, format_type: str,
                          n: int = 1, question_type: str = "multiples",
                          reasoning_effort: str = "high"):
    """로컬 모델로 모든 문제에 대해 n번씩 풀이를 생성합니다."""
    inputs = [(i, j) for i in range(len(problems)) for j in range(n)]
    print(f"Total tasks: {len(inputs)} ({len(problems)} problems x {n} generations) [{question_type}]")
    
    for idx in inputs:
        process_item_local(
            idx, problems, request_sentences,
            output_dir, model, tokenizer, source, format_type, question_type,
            reasoning_effort
        )


# ============================================================================
# 메인
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SFT 학습 데이터 생성기 (로컬 모델 버전)",
        epilog="예시: python generate_sft_data_local.py --model_path /path/to/gpt-oss-12b"
    )
    
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=str)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, type=str)
    parser.add_argument("--reasoning_effort", default="high", type=str,
                        choices=["none", "low", "medium", "high"])
    parser.add_argument("--n", default=DEFAULT_N, type=int)
    parser.add_argument("--format", default=DEFAULT_FORMAT, type=str,
                        choices=["simple", "sharegpt", "alpaca"])
    parser.add_argument("--instruction_file", type=str, default="sentences_ask_boxed_kr.jsonl")
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--subjectives_only", action="store_true")
    parser.add_argument("--multiples_only", action="store_true")
    
    args = parser.parse_args()
    
    # instruction 문장 로드
    sentences_path = os.path.join(args.data_dir, args.instruction_file)
    if not os.path.exists(sentences_path):
        raise FileNotFoundError(f"{args.instruction_file} not found: {sentences_path}")
    request_sentences = open_jsonl(sentences_path)
    print(f"Loaded instruction file: {args.instruction_file}")
    
    # 처리할 수학 파일 목록
    math_files = [args.input_file] if args.input_file else find_math_files(args.data_dir)
    if not math_files:
        raise FileNotFoundError(f"No math JSONL files found in {args.data_dir}")
    
    print(f"Found {len(math_files)} math files: {[os.path.basename(f) for f in math_files]}")
    
    result_dirs = []
    
    if not args.merge_only:
        model, tokenizer = load_local_model(args.model_path)
        
        for file_path in math_files:
            file_name = os.path.basename(file_path)
            source = file_name.replace('.jsonl', '')
            
            print(f"\n{'='*60}\nProcessing: {file_name}\n{'='*60}")
            
            problems = open_jsonl(file_path)
            print(f"Loaded {len(problems)} problems from {file_name}")
            
            mc_count = sum(1 for p in problems if is_multiple_choice(p['problem']))
            print(f"  - 객관식: {mc_count}개, 주관식: {len(problems) - mc_count}개")
            
            subj_output_dir = os.path.join(args.output_dir, source, "subjectives")
            mc_output_dir = os.path.join(args.output_dir, source, "multiples")
            os.makedirs(subj_output_dir, exist_ok=True)
            os.makedirs(mc_output_dir, exist_ok=True)
            result_dirs.extend([subj_output_dir, mc_output_dir])
            
            if not args.multiples_only:
                print(f"\n[주관식 버전 생성] ({len(problems)}개 문제)")
                run_generation_local(problems, request_sentences, subj_output_dir,
                                      model, tokenizer, source, args.format, args.n,
                                      "subjectives", args.reasoning_effort)
            
            if not args.subjectives_only:
                print(f"\n[객관식 버전 생성] ({len(problems)}개 문제)")
                run_generation_local(problems, request_sentences, mc_output_dir,
                                      model, tokenizer, source, args.format, args.n,
                                      "multiples", args.reasoning_effort)
    else:
        for file_path in math_files:
            source = os.path.basename(file_path).replace('.jsonl', '')
            for qtype in ["multiples", "subjectives"]:
                each_output_dir = os.path.join(args.output_dir, source, qtype)
                if os.path.exists(each_output_dir):
                    result_dirs.append(each_output_dir)
    
    # 결과 병합
    print(f"\n{'='*60}\nMerging results...\n{'='*60}")
    merged_path = os.path.join(args.output_dir, "merged", f"sft_math_all_{args.format}.jsonl")
    merge_results(result_dirs, merged_path)
    
    print("\nDONE!")


if __name__ == "__main__":
    main()
