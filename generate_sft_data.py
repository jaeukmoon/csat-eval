"""
SFT ?™ìŠµ ?°ì´???ì„± ?¤í¬ë¦½íŠ¸
?˜í•™ ?˜ëŠ¥ ë¬¸ì œ???€???€?´ë? vLLM?¼ë¡œ ?ì„±?˜ì—¬ SFT ?™ìŠµ ?°ì´?°ë? ë§Œë“­?ˆë‹¤.

?¬ìš©ë²?
    python generate_sft_data.py [?µì…˜]
    
    ?ëŠ” run_sft_pipeline.shë¥??µí•´ ?¤í–‰ (ê¶Œì¥)
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

# ?”â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•—
# ??                   ?”§ ê¸°ë³¸ ?¤ì • (?„ìš”???˜ì •)                            ??# ?šâ•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•?â•

# ----------------------------------------------------------------------------
# ?“ ê²½ë¡œ ?¤ì •
# ----------------------------------------------------------------------------

# ?…ë ¥ ?°ì´???”ë ‰? ë¦¬ (ê¸°ë³¸ê°?
# - *_math.jsonl ?Œì¼ê³?sentences_ask_boxed_kr.jsonl???ˆëŠ” ?´ë”
DEFAULT_DATA_DIR = "./data"

# ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸ê°?  
# - ?ì„±??SFT ?°ì´?°ê? ?€?¥ë˜???´ë”
DEFAULT_OUTPUT_DIR = "./sft_output"

# ----------------------------------------------------------------------------
# ?¤– vLLM ?œë²„ ?¤ì •
# ----------------------------------------------------------------------------

# vLLM API ?œë²„ URL (ê¸°ë³¸ê°?
DEFAULT_BASE_URL = "http://10.0.74.208:8000/v1"

# ?¬ìš©??ëª¨ë¸ ?´ë¦„ (ê¸°ë³¸ê°?
DEFAULT_MODEL = "glm-4.7"

# ----------------------------------------------------------------------------
# ?™ï¸ ?ì„± ?¤ì •
# ----------------------------------------------------------------------------

# ë¬¸ì œ???ì„± ?Ÿìˆ˜ (ê¸°ë³¸ê°?
# - ê°?ë¬¸ì œ???€??ëª?ë²??€?´ë? ?ì„±? ì?
DEFAULT_N = 10

# ?™ì‹œ ?Œì»¤ ??(ê¸°ë³¸ê°?
# - vLLM ?œë²„???™ì‹œ??ë³´ë‚´???”ì²­ ??DEFAULT_WORKER = 20

# ì¶œë ¥ ?•ì‹ (ê¸°ë³¸ê°?
# - simple:   {"problem": ..., "solution": ..., "answer": ...}
# - sharegpt: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}  
# - alpaca:   {"instruction": ..., "input": ..., "output": ...}
DEFAULT_FORMAT = "simple"

# ----------------------------------------------------------------------------
# ?Œ ?¤íŠ¸?Œí¬ ?¤ì •
# ----------------------------------------------------------------------------

# no_proxy ?¤ì • (vLLM ?œë²„ ì£¼ì†Œ - ?„ë¡???°íšŒ)
os.environ["no_proxy"] = "localhost,127.0.0.1,10.0.74.208"

# ============================================================================
# ? í‹¸ë¦¬í‹° ?¨ìˆ˜
# ============================================================================

def open_jsonl(path):
    """JSONL ?Œì¼???½ì–´ ë¦¬ìŠ¤?¸ë¡œ ë°˜í™˜"""
    data = []
    with open(path, mode='r', encoding='utf8') as rf:
        for line in rf:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def to_jsonl(out_path, data):
    """?°ì´?°ë? JSONL ?•ì‹?¼ë¡œ ?€??""
    with open(out_path, mode='w', encoding='utf8') as wf:
        for row in data:
            wf.write(json.dumps(row, ensure_ascii=False))
            wf.write('\n')


def append_jsonl(out_path, row):
    """?¨ì¼ ??ª©??JSONL ?Œì¼??ì¶”ê?"""
    with open(out_path, mode='a', encoding='utf8') as wf:
        wf.write(json.dumps(row, ensure_ascii=False))
        wf.write('\n')


# ============================================================================
# ë¬¸ì œ ?„ì²˜ë¦??¨ìˆ˜
# ============================================================================

def is_multiple_choice(problem_text: str) -> bool:
    """ê°ê???ë¬¸ì œ?¸ì? ?ë³„?©ë‹ˆ??"""
    # ? íƒì§€ ?¨í„´ ?•ì¸: \item[1], \item[2], ... ?±ì´ 5ê°??´ìƒ ?ˆëŠ”ì§€
    choice_pattern = r"\\item\[[1-5]\]"
    matches = re.findall(choice_pattern, problem_text)
    return len(matches) >= 5


def extract_choice_value(problem_text: str, choice_num: int) -> str:
    """
    ê°ê???ë¬¸ì œ?ì„œ ?¹ì • ë²ˆí˜¸ ? íƒì§€??ê°’ì„ ì¶”ì¶œ?©ë‹ˆ??
    
    ?? \\item[2] \\frac{1}{2} ??choice_num=2 ??"\\frac{1}{2}"
    
    Args:
        problem_text: ë¬¸ì œ ?ìŠ¤??        choice_num: ? íƒì§€ ë²ˆí˜¸ (1-5)
    
    Returns:
        ? íƒì§€ ê°?(ì¶”ì¶œ ?¤íŒ¨ ???ë³¸ choice_num??ë¬¸ì?´ë¡œ ë°˜í™˜)
    """
    # \item[N] ?¤ìŒ??ê°’ì„ ì¶”ì¶œ (?¤ìŒ \item?´ë‚˜ \end{itemize} ?„ê¹Œì§€)
    pattern = rf"\\item\[{choice_num}\]\s*(.+?)(?=\\item\[|\\end\{{itemize\}}|$)"
    match = re.search(pattern, problem_text, re.DOTALL)
    if match:
        value = match.group(1).strip()
        # ì¤„ë°”ê¿??œê±°
        value = re.sub(r'\s+', ' ', value)
        return value
    return str(choice_num)


def remove_choices(problem_text: str) -> str:
    """
    ê°ê???ë¬¸ì œ?ì„œ ? íƒì§€ë¥??œê±°?˜ì—¬ ì£¼ê??ìœ¼ë¡?ë³€?˜í•©?ˆë‹¤.
    
    ?œê±° ?¨í„´:
    - \\begin{itemize} ... \\end{itemize} ë¸”ë¡ ?„ì²´
    """
    text = problem_text
    
    # LaTeX itemize ?˜ê²½ ?œê±° (?¬ëŸ¬ ?¨í„´ ?œë„)
    # ?¨í„´ 1: \begin{itemize} ... \end{itemize}
    text = re.sub(r'\\begin\{itemize\}.*?\\end\{itemize\}', '', text, flags=re.DOTALL)
    
    # ?¨í„´ 2: \\begin{itemize} ... \\end{itemize} (?´ìŠ¤ì¼€?´í”„??ë²„ì „)
    text = re.sub(r'\\\\begin\{itemize\}.*?\\\\end\{itemize\}', '', text, flags=re.DOTALL)
    
    # ? íƒì§€ ë²ˆí˜¸ ?¨í„´ ?œê±° (1) (2) (3) (4) (5) ?ëŠ” ??????????    text = re.sub(r'\s*\([1-5]\)\s*[^\(\n]*', '', text)
    text = re.sub(r'\s*[? â‘¡?¢â‘£??\s*[^\n]*', '', text)
    
    # ?°ì† ê³µë°±/ì¤„ë°”ê¿??•ë¦¬
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


def clean_problem_text(problem_text: str, remove_mc_choices: bool = False) -> str:
    """
    ë¬¸ì œ ?ìŠ¤?¸ì—??ë²ˆí˜¸?€ ?ìˆ˜ë¥??œê±°?©ë‹ˆ??
    
    ?œê±° ?¨í„´:
    - ë¬¸ì œ ë²ˆí˜¸: "1. ", "12. " ??(ë¬¸ì???œì‘ ë¶€ë¶?
    - ?ìˆ˜ ?œì‹œ: "[2??", "[3??", "[4??" ??    
    Args:
        remove_mc_choices: Trueë©?ê°ê???? íƒì§€???œê±°
    
    LaTeX ?•ì‹?€ ? ì??©ë‹ˆ??
    """
    text = problem_text.strip()
    
    # ë¬¸ì œ ë²ˆí˜¸ ?œê±° (?œì‘ ë¶€ë¶„ì˜ "?«ì. " ?¨í„´)
    text = re.sub(r'^(\d+)\.\s*', '', text)
    
    # ?ìˆ˜ ?œì‹œ ?œê±° ("[2??", "[3??" ??
    text = re.sub(r'\s*\[\d+??]\s*', ' ', text)
    
    # ê°ê???? íƒì§€ ?œê±° (?µì…˜)
    if remove_mc_choices:
        text = remove_choices(text)
    
    # ?°ì† ê³µë°± ?•ë¦¬
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


# ============================================================================
# ?„ë¡¬?„íŠ¸ ?ì„±
# ============================================================================

def get_prompt(problem_text: str, request_sentences: list, generation_id: int = 0,
               as_subjective: bool = False) -> str:
    """
    ë¬¸ì œ?€ boxed ?”ì²­ ë¬¸ì¥??ì¡°í•©?˜ì—¬ ?„ë¡¬?„íŠ¸ë¥??ì„±?©ë‹ˆ??
    
    Args:
        problem_text: ?ë³¸ ë¬¸ì œ ?ìŠ¤??        request_sentences: boxed ?”ì²­ ë¬¸ì¥ ë¦¬ìŠ¤??        generation_id: ?ì„± ?¸ë±??(?„ë¡¬?„íŠ¸ ë³€?•ì— ?¬ìš©)
        as_subjective: Trueë©?ê°ê???? íƒì§€ë¥??œê±°?˜ì—¬ ì£¼ê??ìœ¼ë¡?ë³€??    """
    # ë¬¸ì œ ?„ì²˜ë¦?(ë²ˆí˜¸/?ìˆ˜ ?œê±°, ? íƒì§€ ?œê±° ?µì…˜)
    cleaned_problem = clean_problem_text(problem_text, remove_mc_choices=as_subjective)
    
    # ?´ì‹œ + generation_idë¡?ë¬¸ì¥ ? íƒ
    hash_code = hash(cleaned_problem) + generation_id
    sent = request_sentences[hash_code % len(request_sentences)]['sent']
    
    # ë°°ì¹˜ ë°©ì‹ ê²°ì • (4ê°€ì§€ ë³€??
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
# vLLM API ?¸ì¶œ
# ============================================================================

def send_msg(prompt: str, base_url: str, model: str = "gpt-oss-120b"):
    """vLLM ?œë²„???”ì²­??ë³´ë‚´ê³??‘ë‹µ??ë°›ìŠµ?ˆë‹¤."""
    client = OpenAI(base_url=base_url, api_key="dummy")
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        reasoning_effort='high',
        temperature=1.0,
        top_p=1.0,
        timeout=60 * 60 * 24  # 24?œê°„ ?€?„ì•„??    )
    return completion


# ============================================================================
# ì¶œë ¥ ?•ì‹ ë³€??# ============================================================================

def format_output(problem: str, solution: str, answer: int, source: str, 
                  generation_id: int, format_type: str = "simple",
                  prompt: str = None) -> dict:
    """
    ê²°ê³¼ë¥?ì§€?•ëœ ?•ì‹?¼ë¡œ ë³€?˜í•©?ˆë‹¤.
    
    ì§€???•ì‹:
    - simple: ê°„ë‹¨??problem/solution ?•ì‹
    - sharegpt: ShareGPT ?€???•ì‹
    - alpaca: Alpaca instruction ?•ì‹
    
    Args:
        prompt: get_prompt()ë¡??ì„±???¤ì œ ?„ë¡¬?„íŠ¸ (sharegpt/alpaca?ì„œ ?¬ìš©)
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
        # promptê°€ ?œê³µ?˜ë©´ ê·¸ë?ë¡??¬ìš©, ?„ë‹ˆë©?cleaned_problem ?¬ìš©
        human_message = prompt if prompt else cleaned_problem
        return {
            "messages": [
                {"role": "user", "content": human_message},
                {"role": "assistant", "content": solution}
            ],
            "answer": answer
        }
    
    elif format_type == "alpaca":
        # promptê°€ ?œê³µ?˜ë©´ ê·¸ë?ë¡??¬ìš©
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
                "instruction": "?¤ìŒ ?˜í•™ ë¬¸ì œë¥??€ê³? ìµœì¢… ?µì„ \\boxed{} ?ˆì— ?£ì–´ì£¼ì„¸??",
                "input": cleaned_problem,
                "output": solution,
                "answer": answer,
                "source": source,
                "generation_id": generation_id
            }
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


# ============================================================================
# ??ª© ì²˜ë¦¬
# ============================================================================

def process_item(idx: tuple, problems: list, request_sentences: list, 
                 output_dir: str, base_url: str, model: str, 
                 source: str, format_type: str, question_type: str = "multiples"):
    """
    ?¨ì¼ ë¬¸ì œ-?ì„± ?ì„ ì²˜ë¦¬?©ë‹ˆ??
    
    Args:
        idx: (ë¬¸ì œ ?¸ë±?? ?ì„± ?¸ë±??
        question_type: "multiples" (ê°ê??? ?ëŠ” "subjectives" (ì£¼ê???
    """
    problem_idx, gen_idx = idx
    output_path = f"{output_dir}/{problem_idx}_{gen_idx}.jsonl"
    
    # ?´ë? ì²˜ë¦¬??ê²½ìš° ?¤í‚µ
    if os.path.exists(output_path):
        return None
    
    req_start = time.time()
    start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"SEND [{problem_idx}_{gen_idx}] ({question_type}) | st={start_stamp}")
    
    MAX_RETRIES = 10
    BACKOFF_FACTOR = 2
    
    item = problems[problem_idx]
    # ì£¼ê??ì´ë©?? íƒì§€ ?œê±°
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
    
    # ?‘ë‹µ ì¶”ì¶œ (reasoning_contentê°€ ?ˆìœ¼ë©?<think> ?œê·¸ë¡?ê°ì‹¸???¬í•¨)
    resp_dict = resp.choices[0].to_dict()
    message = resp_dict.get("message", {})
    
    if message.get("reasoning_content"):
        solution = f"<think>\n{message['reasoning_content'].strip()}\n</think>\n{message['content'].strip()}"
    else:
        solution = message.get("content", "")
    
    answer = item.get('answer', None)
    
    # ì£¼ê???ë²„ì „?´ë©´???ë³¸??ê°ê??ì¸ ê²½ìš°, ?¤ì œ ??ê°’ì„ ì¶”ì¶œ
    # (?? answer=2, 2ë²?? íƒì§€ê°€ "\frac{1}{2}"?´ë©´ ??real_answer="\frac{1}{2}")
    if as_subjective and is_multiple_choice(item['problem']) and answer is not None:
        real_answer = extract_choice_value(item['problem'], answer)
    else:
        real_answer = answer
    
    # ?•ì‹??ë§ê²Œ ë³€??(?ì„±???„ë¡¬?„íŠ¸???„ë‹¬)
    formatted = format_output(
        problem=item['problem'],
        solution=solution,
        answer=real_answer,  # ì£¼ê??ì? ?¤ì œ ??ê°? ê°ê??ì? ? íƒì§€ ë²ˆí˜¸
        source=source,
        generation_id=gen_idx,
        format_type=format_type,
        prompt=prompt  # boxed ë¬¸ì¥???¬í•¨???¤ì œ ?„ë¡¬?„íŠ¸
    )
    
    # ?€??    to_jsonl(output_path, [formatted])
    
    end_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    req_duration = time.time() - req_start
    
    print(f"DONE [{problem_idx}_{gen_idx}] | time={req_duration:.2f}s | trials={trial + 1}")
    return formatted


def run_generation(problems: list, request_sentences: list, output_dir: str,
                   base_url: str, model: str, source: str, format_type: str,
                   n: int = 10, max_workers: int = 200, question_type: str = "multiples",
                   retry_problems: list = None):
    """
    ëª¨ë“  ë¬¸ì œ???€??në²ˆì”© ?€?´ë? ?ì„±?©ë‹ˆ??
    
    Args:
        question_type: "multiples" (ê°ê??? ?ëŠ” "subjectives" (ì£¼ê???
        retry_problems: ?¬ìƒ?±í•  ë¬¸ì œ ?¸ë±??ëª©ë¡ (None?´ë©´ ëª¨ë“  ë¬¸ì œ ì²˜ë¦¬)
    """
    inputs = []
    
    if retry_problems is not None:
        # ?¬ìƒ??ëª¨ë“œ: ì§€?•ëœ ë¬¸ì œë§?ì²˜ë¦¬
        for problem_idx in retry_problems:
            for j in range(n):
                inputs.append((problem_idx, j))
        print(f"Retry mode: {len(inputs)} tasks ({len(retry_problems)} problems x {n} generations) [{question_type}]")
    else:
        # ?¼ë°˜ ëª¨ë“œ: ëª¨ë“  ë¬¸ì œ ì²˜ë¦¬
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
# ê²°ê³¼ ë³‘í•©
# ============================================================================

def merge_results(input_dirs: list, output_path: str):
    """
    ?¬ëŸ¬ ?”ë ‰? ë¦¬??ê²°ê³¼ë¥??˜ë‚˜??JSONL ?Œì¼ë¡?ë³‘í•©?©ë‹ˆ??
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
    
    # ì¶œë ¥ ?”ë ‰? ë¦¬ ?ì„±
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    to_jsonl(output_path, all_data)
    print(f"Saved to: {output_path}")
    
    return all_data


def find_math_files(data_dir: str) -> list:
    """data ?”ë ‰? ë¦¬?ì„œ ?˜í•™ JSONL ?Œì¼?¤ì„ ì°¾ìŠµ?ˆë‹¤."""
    pattern = os.path.join(data_dir, "*_math.jsonl")
    files = glob.glob(pattern)
    return sorted(files)


# ============================================================================
# ë©”ì¸
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SFT ?™ìŠµ ?°ì´???ì„±ê¸?,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
?ˆì‹œ:
  python generate_sft_data.py
  python generate_sft_data.py --n 20 --worker 40
  python generate_sft_data.py --input_file ./data/2025_math.jsonl
  python generate_sft_data.py --retry_file ./sft_output/.retry_queue.jsonl
        """
    )
    
    # ê²½ë¡œ ?¤ì •
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str,
                        help=f"?…ë ¥ ?°ì´???”ë ‰? ë¦¬ (ê¸°ë³¸: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=str,
                        help=f"ì¶œë ¥ ?”ë ‰? ë¦¬ (ê¸°ë³¸: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--input_file", type=str, default=None,
                        help="?¹ì • ?Œì¼ë§?ì²˜ë¦¬ (ë¯¸ì??•ì‹œ data_dir ??ëª¨ë“  *_math.jsonl ì²˜ë¦¬)")
    
    # ?ì„± ?¤ì •
    parser.add_argument("--n", default=DEFAULT_N, type=int,
                        help=f"ë¬¸ì œ???ì„± ?Ÿìˆ˜ (ê¸°ë³¸: {DEFAULT_N})")
    parser.add_argument("--worker", default=DEFAULT_WORKER, type=int,
                        help=f"?™ì‹œ ?Œì»¤ ??(ê¸°ë³¸: {DEFAULT_WORKER})")
    parser.add_argument("--format", default=DEFAULT_FORMAT, type=str,
                        choices=["simple", "sharegpt", "alpaca"],
                        help=f"ì¶œë ¥ ?•ì‹ (ê¸°ë³¸: {DEFAULT_FORMAT})")
    
    # vLLM ?œë²„ ?¤ì •
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, type=str,
                        help=f"vLLM ?œë²„ URL (ê¸°ë³¸: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, type=str,
                        help=f"ëª¨ë¸ ?´ë¦„ (ê¸°ë³¸: {DEFAULT_MODEL})")
    
    # ?¤í–‰ ëª¨ë“œ
    parser.add_argument("--merge_only", action="store_true",
                        help="?ì„± ?†ì´ ê¸°ì¡´ ê²°ê³¼ë§?ë³‘í•©")
    parser.add_argument("--retry_file", type=str, default=None,
                        help="?¬ìƒ?±í•  ë¬¸ì œ ëª©ë¡ ?Œì¼ ê²½ë¡œ (.retry_queue.jsonl)")
    
    args = parser.parse_args()
    
    # boxed ?”ì²­ ë¬¸ì¥ ë¡œë“œ
    sentences_path = os.path.join(args.data_dir, "sentences_ask_boxed_kr.jsonl")
    if not os.path.exists(sentences_path):
        raise FileNotFoundError(f"sentences_ask_boxed_kr.jsonl not found: {sentences_path}")
    request_sentences = open_jsonl(sentences_path)
    
    # ì²˜ë¦¬???˜í•™ ?Œì¼ ëª©ë¡
    if args.input_file:
        math_files = [args.input_file]
    else:
        math_files = find_math_files(args.data_dir)
    
    if not math_files:
        raise FileNotFoundError(f"No math JSONL files found in {args.data_dir}")
    
    print(f"Found {len(math_files)} math files: {[os.path.basename(f) for f in math_files]}")
    
    # ?¬ìƒ???Œì¼ ë¡œë“œ
    retry_queue = {}  # {(source, question_type): [problem_indices]}
    if args.retry_file and os.path.exists(args.retry_file):
        retry_items = open_jsonl(args.retry_file)
        for item in retry_items:
            key = (item["source"], item["question_type"])
            if key not in retry_queue:
                retry_queue[key] = []
            retry_queue[key].append(item["problem_idx"])
        print(f"\n?¬ìƒ??ëª¨ë“œ: {len(retry_items)}ê°?ë¬¸ì œ ?¬ìƒ???ˆì •")
        for (src, qtype), indices in retry_queue.items():
            print(f"  - {src}/{qtype}: ë¬¸ì œ {indices}")
    
    result_dirs = []
    
    if not args.merge_only:
        # ê°??Œì¼ ì²˜ë¦¬
        for file_path in math_files:
            file_name = os.path.basename(file_path)
            source = file_name.replace('.jsonl', '')
            
            print(f"\n{'='*60}")
            print(f"Processing: {file_name}")
            print(f"{'='*60}")
            
            problems = open_jsonl(file_path)
            print(f"Loaded {len(problems)} problems from {file_name}")
            
            # ?µê³„ ì¶œë ¥
            mc_count = sum(1 for p in problems if is_multiple_choice(p['problem']))
            print(f"  - ê°ê???ë¬¸ì œ: {mc_count}ê°? ì£¼ê???ë¬¸ì œ: {len(problems) - mc_count}ê°?)
            
            # ì¶œë ¥ ?”ë ‰? ë¦¬ ì¤€ë¹?            subj_output_dir = os.path.join(args.output_dir, source, "subjectives")
            mc_output_dir = os.path.join(args.output_dir, source, "multiples")
            os.makedirs(subj_output_dir, exist_ok=True)
            os.makedirs(mc_output_dir, exist_ok=True)
            result_dirs.append(subj_output_dir)
            result_dirs.append(mc_output_dir)
            
            # ?¬ìƒ?±í•  ë¬¸ì œ ëª©ë¡ ?•ì¸
            subj_retry = retry_queue.get((source, "subjectives"), None)
            mc_retry = retry_queue.get((source, "multiples"), None)
            
            # ?¬ìƒ??ëª¨ë“œ??ê²½ìš° ê¸°ì¡´ ?Œì¼ ?? œ
            if subj_retry:
                print(f"\n[ì£¼ê????¬ìƒ?? ë¬¸ì œ {subj_retry} ê¸°ì¡´ ?Œì¼ ?? œ ì¤?..")
                for problem_idx in subj_retry:
                    for gen_idx in range(args.n):
                        old_file = os.path.join(subj_output_dir, f"{problem_idx}_{gen_idx}.jsonl")
                        if os.path.exists(old_file):
                            os.remove(old_file)
                            print(f"  ?? œ: {old_file}")
            
            if mc_retry:
                print(f"\n[ê°ê????¬ìƒ?? ë¬¸ì œ {mc_retry} ê¸°ì¡´ ?Œì¼ ?? œ ì¤?..")
                for problem_idx in mc_retry:
                    for gen_idx in range(args.n):
                        old_file = os.path.join(mc_output_dir, f"{problem_idx}_{gen_idx}.jsonl")
                        if os.path.exists(old_file):
                            os.remove(old_file)
                            print(f"  ?? œ: {old_file}")
            
            # ?ì„±???€??ê²°ì •
            run_subj = subj_retry or not retry_queue
            run_mc = mc_retry or not retry_queue
            
            # ?Œì»¤ ë¶„ë°°: ?????¤í–‰?˜ë©´ ë°˜ë°˜, ?˜ë‚˜ë§Œì´ë©??„ì²´ ?¬ìš©
            if run_subj and run_mc:
                # ?? ì£¼ê???ê°ê????™ì‹œ ?ì„± (?Œì»¤ ë°˜ë°˜ ë¶„ë°°)
                subj_workers = args.worker // 2
                mc_workers = args.worker - subj_workers  # ?€?˜ì¼ ê²½ìš° ê°ê??ì— +1
                
                print(f"\n[?™ì‹œ ?ì„± ëª¨ë“œ] ì£¼ê????Œì»¤: {subj_workers}, ê°ê????Œì»¤: {mc_workers}")
                print(f"  - ì£¼ê??? {len(subj_retry) if subj_retry else len(problems)}ê°?ë¬¸ì œ")
                print(f"  - ê°ê??? {len(mc_retry) if mc_retry else len(problems)}ê°?ë¬¸ì œ")
                
                # ???ì„± ?‘ì—…???™ì‹œ???¤í–‰
                with ThreadPoolExecutor(max_workers=2) as type_executor:
                    subj_future = type_executor.submit(
                        run_generation,
                        problems=problems,
                        request_sentences=request_sentences,
                        output_dir=subj_output_dir,
                        base_url=args.base_url,
                        model=args.model,
                        source=source,
                        format_type=args.format,
                        n=args.n,
                        max_workers=subj_workers,
                        question_type="subjectives",
                        retry_problems=subj_retry
                    )
                    mc_future = type_executor.submit(
                        run_generation,
                        problems=problems,
                        request_sentences=request_sentences,
                        output_dir=mc_output_dir,
                        base_url=args.base_url,
                        model=args.model,
                        source=source,
                        format_type=args.format,
                        n=args.n,
                        max_workers=mc_workers,
                        question_type="multiples",
                        retry_problems=mc_retry
                    )
                    
                    # ?„ë£Œ ?€ê¸?                    subj_future.result()
                    mc_future.result()
                    
            elif run_subj:
                # ì£¼ê??ë§Œ ?ì„± (?„ì²´ ?Œì»¤ ?¬ìš©)
                print(f"\n[ì£¼ê???ë²„ì „ ?ì„± ?œì‘] ({len(subj_retry) if subj_retry else len(problems)}ê°?ë¬¸ì œ)")
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
                
            elif run_mc:
                # ê°ê??ë§Œ ?ì„± (?„ì²´ ?Œì»¤ ?¬ìš©)
                print(f"\n[ê°ê???ë²„ì „ ?ì„± ?œì‘] ({len(mc_retry) if mc_retry else len(problems)}ê°?ë¬¸ì œ)")
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
        # merge_only ëª¨ë“œ: ê¸°ì¡´ ?”ë ‰? ë¦¬ ì°¾ê¸°
        for file_path in math_files:
            source = os.path.basename(file_path).replace('.jsonl', '')
            # multiples?€ subjectives ?´ë” ëª¨ë‘ ì°¾ê¸°
            for qtype in ["multiples", "subjectives"]:
                each_output_dir = os.path.join(args.output_dir, source, qtype)
                if os.path.exists(each_output_dir):
                    result_dirs.append(each_output_dir)
    
    # ê²°ê³¼ ë³‘í•©
    print(f"\n{'='*60}")
    print("Merging results...")
    print(f"{'='*60}")
    
    merged_dir = os.path.join(args.output_dir, "merged")
    merged_path = os.path.join(merged_dir, f"sft_math_all_{args.format}.jsonl")
    
    merge_results(result_dirs, merged_path)
    
    print("\nDONE!")


if __name__ == "__main__":
    main()
