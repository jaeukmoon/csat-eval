import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline as hf_pipeline, AutoModelForCausalLM

from data_provider.data_factory import data_provider
from data_load.dataloader import DataLoader_
from explain_module.expagents import PredictAgent
from predict_module.sft_dataloader import SFTDataLoader_generate
from prompts.prompts_baseline import *
from prompts.prompts_new import *
from utils.args import get_base_parser, add_test_args
from utils.tools import strip_thinking

warnings.filterwarnings('ignore')

# === Argument parsing (centralized in utils/args.py) ===
parser = get_base_parser(description='GPT4TS Test')
add_test_args(parser)
args = parser.parse_args()


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")


args.trained = _str2bool(args.trained)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def build_prompt_type(args):
    """
    args.prediction_prompt 이름에서 prompt_type 추출
    """
    name_raw = (args.prediction_prompt or "").strip()
    name = name_raw.lower()
    if (args.trained is False) and (name_raw == "ILI_PREDICT_INSTRUCTION_PROMPT_INSTRUCT_CoT_date_1217"):
        return "baseline_full"

    if "baseline_news_exp" in name:
        return "baseline_news_exp"
    elif "baseline_news" in name:
        return "baseline_news"
    elif "cot" in name:
        return "baseline_CoT"
    elif "icl" in name:
        return "baseline_ICL"
    else:
        return "baseline"


def testset_json_generator(args, prompt_type):
    """
    - ./testset/{exp_name}/{exp_name}_{target}.json 존재 여부 확인
    - 없으면 data_provider로 불러와서 JSON 저장
    - baseline_ICL에서 explanations 생성/주입
    - ✅ trained=True면 prompt_type이 ICL이 아니어도 explanations 생성/주입
    """
    # exp 이름 결정 (기존 유지)
    if "baseline" in prompt_type:
        exp_name = prompt_type
    else:
        exp_name = args.exp_llm if getattr(args, "exp_llm", None) else prompt_type

    testset_dir = f"./testset/{exp_name}"
    os.makedirs(testset_dir, exist_ok=True)

    filename = f"{exp_name}_{args.target.replace(' ', '_')}.json"
    testset_path = os.path.join(testset_dir, filename)

    if os.path.exists(testset_path):
        print(f"[INFO] Found existing testset JSON: {testset_path}")
        return testset_path

    print(f"[INFO] No testset JSON found. Building testset at: {testset_path}")

    # --------------------------
    # 1) 데이터 로드
    # --------------------------
    test_data, test_loader = data_provider(args, 'test')
    dataloader = DataLoader_(args)
    data_news_test = dataloader.load(args, test_data, flag="test")
    data_news_test_pastpop = data_news_test[data_news_test['past_data_X'].notna()].reset_index(drop=True)

    # --------------------------
    # 2) explanations 생성/주입
    # --------------------------
    need_explanations = (prompt_type == 'baseline_full') or (prompt_type == 'baseline_ICL') or (args.trained is True)

    if need_explanations:
        exp_agent = PredictAgent(args=args, exp_llm=args.exp_llm)
        # 여기서 사용하는 explanation 프롬프트는 args.explanation_prompt (bash가 매번 다르게 넘김)
        exp_list, past_exp_list = exp_agent.explanation_files_generation(
            data=data_news_test_pastpop,
            explanation_path_name='explanations'
        )
        data_news_test_pastpop['explanations'] = exp_list
        data_news_test_pastpop['past_explanations'] = past_exp_list
    else:
        if 'explanations' not in data_news_test_pastpop.columns:
            data_news_test_pastpop['explanations'] = ""
        if 'past_explanations' not in data_news_test_pastpop.columns:
            data_news_test_pastpop['past_explanations'] = ""

    # --------------------------
    # 3) prediction 프롬프트 템플릿 로드 (✅ args.prediction_prompt 사용)
    # --------------------------
    prediction_prompt = eval(args.prediction_prompt)

    # --------------------------
    # 4) 안전한 format 매핑 생성
    # --------------------------
    class SafeDict(defaultdict):
        def __missing__(self, key):
            return ""

    root = {}

    for idx, row in data_news_test_pastpop.iterrows():
        mapping = SafeDict(str)

        mapping.update({
            "start_X_date": row.get("start_X_date", ""),
            "end_X_date": row.get("end_X_date", ""),
            "data_X": row.get("data_X", ""),
            "summary": row.get("news_summary", ""),
            "Y_date": row.get("Y_date", ""),

            "start_past_X_date": row.get("start_past_X_date", ""),
            "end_past_X_date": row.get("end_past_X_date", ""),
            "past_data_X": row.get("past_data_X", ""),
            "past_summary": row.get("past_news_summary", ""),
            "past_Y_date": row.get("past_Y_date", ""),
            "past_data_Y": row.get("past_data_Y", ""),

            "region_number": row.get("region_number", ""),
            "past_region_number": row.get("past_region_number", ""),

            "pred_len": str(getattr(args, "pred_len", "")),
            "seq_len": str(getattr(args, "seq_len", "")),
        })

        mapping["past_explanations"] = row.get("past_explanations", "")
        mapping["explanations"] = row.get("explanations", "")

        prompt = prediction_prompt.format_map(mapping)

        root[str(idx)] = {
            "text": {
                "0": prompt,               # user
                "1": str(row.get("data_Y", ""))  # assistant (label)
            }
        }

    # --------------------------
    # 5) json 저장
    # --------------------------
    with open(testset_path, "w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False)

    print(f"[INFO] Saved testset JSON: {testset_path} (num_samples={len(data_news_test_pastpop)})")
    return testset_path


def parse_prediction(generated_text: str, isgpt):
    if isgpt:
        split_output = generated_text.split('final<|message|>')
    else:
        split_output = generated_text.split('[OUTPUT]')

    if len(split_output) < 2:
        prompt_part = generated_text
        prediction_part = generated_text
    else:
        prompt_part = split_output[-2]
        prediction_part = split_output[-1]

    ili_occurrences = prediction_part.split('Explanation:')[0] \
        .split('Future ILI occurrences:')[-1].strip()

    explanation = prediction_part.split('Explanation:')[-1].strip()
    predicted_Y = prediction_part.strip()

    return ili_occurrences, explanation, predicted_Y, prompt_part


def run_inference_with_pipeline(args, prompt_type, testset_path):
    """
    - "호출 1회 = inference 1회"
    - (model, prompt_type, target, trained, prediction_prompt, explanation_prompt) 조합별로
      run 파일 개수를 세서 existing_runs >= args.runs 이면 skip
    """
    model_path_basename = os.path.basename(args.model_path.rstrip("/"))

    if not args.trained:
        # baseline: model_path가 베이스모델 경로
        model_name = model_path_basename
        model_dir = os.path.join("baseline_test", model_name, prompt_type)

        output_name = f"{model_name}_{prompt_type}_{args.target}"
        base_csv_name = f"test_predicted_{output_name}.csv"

    else:
        # trained: output_dir에서 MODEL_NAME과 save_group 추출
        if not args.output_dir:
            raise ValueError("--output_dir is required when --trained true")

        # OUTPUT_DIR = .../trained_models/<MODEL_NAME>/<SAVE_GROUP>
        model_name = os.path.basename(os.path.dirname(args.output_dir.rstrip("/")))
        save_group = os.path.basename(args.output_dir.rstrip("/"))

        model_dir = os.path.join("trained_test", model_name)

        save_group = save_group.replace(".csv", "")
        base_csv_name = f"test_predicted_{save_group}.csv"

    os.makedirs(model_dir, exist_ok=True)
    safe_base_csv_name = base_csv_name.replace(os.sep, "_")

    def _existing_run_nums(directory: str, filename: str) -> list[int]:
        base, ext = os.path.splitext(filename)
        pattern = re.compile(rf"^{re.escape(base)}_run(\d+){re.escape(ext)}$")
        nums = []
        if not os.path.isdir(directory):
            return nums
        for f in os.listdir(directory):
            m = pattern.match(f)
            if m:
                nums.append(int(m.group(1)))
        return nums

    def _next_run_path(directory: str, filename: str) -> tuple[str, int]:
        base, ext = os.path.splitext(filename)
        nums = _existing_run_nums(directory, filename)
        next_n = (max(nums) + 1) if nums else 1
        next_path = os.path.join(directory, f"{base}_run{next_n}{ext}")
        return next_path, next_n

    run_nums = _existing_run_nums(model_dir, safe_base_csv_name)
    existing_runs = len(run_nums)

    if existing_runs >= args.runs:
        print(
            f"[INFO] Already have {existing_runs} runs (requested={args.runs}). "
            f"Skip: {model_name} | {prompt_type} | {args.target} | trained={args.trained} | "
            f"pred={args.prediction_prompt} | exp={args.explanation_prompt}"
        )
        return

    final_path, run_idx = _next_run_path(model_dir, safe_base_csv_name)
    print(f"[INFO] Run {run_idx}/{args.runs} => saving to {final_path} | seed={args.seed} | trained={args.trained}")

    # -----------------------
    # 1) JSON 로드
    # -----------------------
    with open(testset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    keys = sorted(dataset.keys(), key=lambda x: int(x))
    prompts, labels = [], []
    for k in keys:
        text_block = dataset[k]["text"]
        prompts.append(text_block["0"])
        labels.append(text_block["1"])

    # -----------------------
    # 2) 모델 타입 판별
    # -----------------------
    model_path_lower = args.model_path.lower()
    is_qwen = "qwen" in model_path_lower
    is_gpt_oss = ("gpt-oss" in model_path_lower) or ("gpt_oss" in model_path_lower)

    # -----------------------
    # 3) tokenizer / model
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    # -----------------------
    # Qwen
    # -----------------------
    if is_qwen:
        print("[INFO] Detected Qwen family model. Using Qwen-specific path.")
        text_gen_pipe = hf_pipeline("text-generation", model=model, device_map="auto", tokenizer=tokenizer)

        results = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            processed_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(args.qwen_thinking == "on")
            )
            out = text_gen_pipe([processed_prompt], do_sample=True, max_new_tokens=1000, temperature=args.temperature)
            results.append(out)

        df = pd.DataFrame({"input_prompt": prompts, "data_Y": labels})
        df["run_idx"] = run_idx
        df["seed"] = args.seed
        df["trained"] = args.trained

        ili_occurs, explanations, predicted_Ys = [], [], []
        for idx, (result, gt) in enumerate(zip(results, labels), start=1):
            inner = result[0]
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                model_output = inner[0].get("generated_text", "")
            elif isinstance(inner, dict):
                model_output = inner.get("generated_text", "")
            else:
                model_output = str(inner)

            clean_output = strip_thinking(model_output)
            ILI_occur, explanation, predicted_Y, _ = parse_prediction(clean_output, isgpt=is_gpt_oss)
            ili_occurs.append(ILI_occur)
            explanations.append(explanation)
            predicted_Ys.append(predicted_Y)

            print(f"[Qwen][{idx}/{len(prompts)}] Actual: {gt}, Predicted(trunc): {predicted_Y}")

        df["ILI_occur"] = ili_occurs
        df["explanation"] = explanations
        df["predicted_Y"] = predicted_Ys
        df.to_csv(final_path, index=False)
        print(f"[INFO] Qwen inference finished. CSV saved to {final_path}")
        return

    # -----------------------
    # GPT-OSS
    # -----------------------
    if is_gpt_oss:
        print("[INFO] Detected gpt-oss model.")
        df = pd.DataFrame({"input_prompt": prompts, "data_Y": labels})
        df["run_idx"] = run_idx
        df["seed"] = args.seed
        df["trained"] = args.trained

        ili_occurs, explanations, predicted_Ys = [], [], []

        for idx, (p, gt) in enumerate(zip(prompts, labels), start=1):
            messages = [{"role": "user", "content": p}]
            chat_kwargs = {}
            if args.reasoning_effort != "none":
                chat_kwargs["reasoning_effort"] = args.reasoning_effort

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **chat_kwargs
            ).to(model.device)

            outputs = model.generate(**inputs, max_new_tokens=10000, temperature=args.temperature)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

            ILI_occur, explanation, predicted_Y, _ = parse_prediction(decoded, isgpt=is_gpt_oss)
            ili_occurs.append(ILI_occur)
            explanations.append(explanation)
            predicted_Ys.append(predicted_Y)

            print(f"[gpt-oss][{idx}/{len(prompts)}] Actual: {gt}, Predicted(trunc): {predicted_Y}")

        df["ILI_occur"] = ili_occurs
        df["explanation"] = explanations
        df["predicted_Y"] = predicted_Ys
        df.to_csv(final_path, index=False)
        print(f"[INFO] gpt-oss inference finished. CSV saved to {final_path}")
        return

    # -----------------------
    # Generic HF models
    # -----------------------
    print("[INFO] Using generic HF pipeline path (non-Qwen, non-gpt-oss).")

    text_gen_pipe = hf_pipeline("text-generation", model=model, device_map="auto", tokenizer=tokenizer)
    outputs = text_gen_pipe(
        prompts,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        max_new_tokens=500,
        temperature=args.temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    df = pd.DataFrame({"input_prompt": prompts, "data_Y": labels})
    df["run_idx"] = run_idx
    df["seed"] = args.seed
    df["trained"] = args.trained

    ili_occurs, explanations, predicted_Ys = [], [], []

    for idx, (result, gt) in enumerate(zip(outputs, labels), start=1):
        if isinstance(result, list) and len(result) > 0:
            model_output = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            model_output = result.get("generated_text", "")
        else:
            model_output = str(result)

        clean_output = strip_thinking(model_output) if callable(strip_thinking) else model_output
        ILI_occur, explanation, predicted_Y, _ = parse_prediction(clean_output, isgpt=is_gpt_oss)

        ili_occurs.append(ILI_occur)
        explanations.append(explanation)
        predicted_Ys.append(predicted_Y)

        print(f"[Generic][{idx}/{len(prompts)}] Actual: {gt}, Predicted(trunc): {predicted_Y}")

    df["ILI_occur"] = ili_occurs
    df["explanation"] = explanations
    df["predicted_Y"] = predicted_Ys
    df.to_csv(final_path, index=False)
    print(f"[INFO] Generic inference finished. CSV saved to {final_path}")


def run_inference_with_vllm(args, prompt_type, testset_path):
    """
    vLLM offline batch inference.
    Same output format as run_inference_with_pipeline but uses vLLM for faster batch processing.
    """
    from vllm import LLM, SamplingParams

    model_path_basename = os.path.basename(args.model_path.rstrip("/"))

    # -----------------------
    # 1) 저장 경로 결정 (pipeline과 동일)
    # -----------------------
    if not args.trained:
        model_name = model_path_basename
        model_dir = os.path.join("baseline_test", model_name, prompt_type)
        output_name = f"{model_name}_{prompt_type}_{args.target}"
        base_csv_name = f"test_predicted_{output_name}.csv"
    else:
        if not args.output_dir:
            raise ValueError("--output_dir is required when --trained true")
        model_name = os.path.basename(os.path.dirname(args.output_dir.rstrip("/")))
        save_group = os.path.basename(args.output_dir.rstrip("/"))
        model_dir = os.path.join("trained_test", model_name)
        save_group = save_group.replace(".csv", "")
        base_csv_name = f"test_predicted_{save_group}.csv"

    os.makedirs(model_dir, exist_ok=True)
    safe_base_csv_name = base_csv_name.replace(os.sep, "_")

    def _existing_run_nums(directory: str, filename: str) -> list[int]:
        base, ext = os.path.splitext(filename)
        pattern = re.compile(rf"^{re.escape(base)}_run(\d+){re.escape(ext)}$")
        nums = []
        if not os.path.isdir(directory):
            return nums
        for f in os.listdir(directory):
            m = pattern.match(f)
            if m:
                nums.append(int(m.group(1)))
        return nums

    def _next_run_path(directory: str, filename: str) -> tuple[str, int]:
        base, ext = os.path.splitext(filename)
        nums = _existing_run_nums(directory, filename)
        next_n = (max(nums) + 1) if nums else 1
        next_path = os.path.join(directory, f"{base}_run{next_n}{ext}")
        return next_path, next_n

    run_nums = _existing_run_nums(model_dir, safe_base_csv_name)
    existing_runs = len(run_nums)

    if existing_runs >= args.runs:
        print(
            f"[INFO] Already have {existing_runs} runs (requested={args.runs}). "
            f"Skip: {model_name} | {prompt_type} | {args.target} | trained={args.trained} | "
            f"pred={args.prediction_prompt} | exp={args.explanation_prompt}"
        )
        return

    final_path, run_idx = _next_run_path(model_dir, safe_base_csv_name)
    print(f"[INFO][vLLM] Run {run_idx}/{args.runs} => saving to {final_path} | seed={args.seed} | trained={args.trained}")

    # -----------------------
    # 2) JSON 로드
    # -----------------------
    with open(testset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    keys = sorted(dataset.keys(), key=lambda x: int(x))
    prompts, labels = [], []
    for k in keys:
        text_block = dataset[k]["text"]
        prompts.append(text_block["0"])
        labels.append(text_block["1"])

    # -----------------------
    # 3) vLLM 모델 로딩
    # -----------------------
    print(f"[INFO][vLLM] Loading model from: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="auto",
    )

    # -----------------------
    # 4) SamplingParams 설정 (pipeline과 동일한 파라미터)
    # -----------------------
    sampling_params = SamplingParams(
        temperature=args.temperature if args.temperature > 0 else 0.01,  # vLLM requires temp > 0 for sampling
        max_tokens=500,
        top_k=10,
    )

    # -----------------------
    # 5) 배치 생성
    # -----------------------
    print(f"[INFO][vLLM] Generating {len(prompts)} prompts in batch...")
    outputs = llm.generate(prompts, sampling_params)

    # -----------------------
    # 6) 결과 파싱 및 저장
    # -----------------------
    model_path_lower = args.model_path.lower()
    is_gpt_oss = ("gpt-oss" in model_path_lower) or ("gpt_oss" in model_path_lower)

    df = pd.DataFrame({"input_prompt": prompts, "data_Y": labels})
    df["run_idx"] = run_idx
    df["seed"] = args.seed
    df["trained"] = args.trained

    ili_occurs, explanations, predicted_Ys = [], [], []

    for idx, (output, gt) in enumerate(zip(outputs, labels), start=1):
        generated_text = output.outputs[0].text if output.outputs else ""

        clean_output = strip_thinking(generated_text) if callable(strip_thinking) else generated_text
        ILI_occur, explanation, predicted_Y, _ = parse_prediction(clean_output, isgpt=is_gpt_oss)

        ili_occurs.append(ILI_occur)
        explanations.append(explanation)
        predicted_Ys.append(predicted_Y)

        print(f"[vLLM][{idx}/{len(prompts)}] Actual: {gt}, Predicted(trunc): {predicted_Y[:80] if len(predicted_Y) > 80 else predicted_Y}")

    df["ILI_occur"] = ili_occurs
    df["explanation"] = explanations
    df["predicted_Y"] = predicted_Ys
    df.to_csv(final_path, index=False)
    print(f"[INFO][vLLM] Inference finished. CSV saved to {final_path}")


def run_eval(args):
    prompt_type = build_prompt_type(args)
    testset_path = testset_json_generator(args, prompt_type)

    if args.inference_mode == "vllm":
        run_inference_with_vllm(args, prompt_type, testset_path)
    else:
        run_inference_with_pipeline(args, prompt_type, testset_path)


if __name__ == "__main__":
    run_eval(args)
