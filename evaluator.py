from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm
from datasets import load_dataset


def print_human_summary(summary: dict) -> None:
    backend = summary.get("backend", "")
    split = summary.get("split", "")
    n = summary.get("n", 0)

    correct = summary.get("correct", 0)
    acc = summary.get("accuracy", 0.0)
    score = summary.get("score", 0)
    max_score = summary.get("max_score", 0)

    split_desc = f"{split} (수능 문제)"
    model = summary.get("model") or summary.get("model_name") or summary.get("model_id") or "N/A"

    acc_pct = acc * 100.0
    score_pct = (score / max_score * 100.0) if max_score else 0.0

    print("\n" + "=" * 60)
    print(f"평가 백엔드: {backend}")
    print(f"모델: {model}")
    print(f"split: {split_desc}")
    print(f"문항 수(n): {n}")
    print("-" * 60)
    print(f"맞은 개수(correct): {correct} / {n}")
    print(f"정확도(accuracy): {acc_pct:.2f}%")
    print(f"획득 점수(score): {score} / {max_score}  ({score_pct:.2f}%)")
    print("=" * 60 + "\n")


def load_eval_split(args: Any):
    if getattr(args, "source", "local") != "local":
        raise ValueError("현재 설정은 로컬 JSONL 전용입니다. --source local 로 실행하세요.")

    data_dir = Path(getattr(args, "data_dir", "./data"))
    path = data_dir / f"{args.split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Local JSONL not found: {path}")

    ds = load_dataset("json", data_files=str(path), split="train")
    return ds


def run_openai_eval(common: Any, args: Any) -> None:
    import os
    import openai
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")
    openai.api_key = api_key

    ds = load_eval_split(args)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def call_openai(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        completion = openai.chat.completions.create(
            model=args.model,
            messages=messages,
        )
        return completion.choices[0].message.content or ""

    total_score = 0
    max_score = 0
    correct_cnt = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, desc=f"OpenAI eval ({args.model})"):
            qtype = common.infer_qtype(ex)
            subject = getattr(common, "subject", "math")
            prompt = common.build_prompt(ex["problem"], qtype, subject)

            text = call_openai(prompt)

            pred = common.extract_final_answer(text, qtype)
            gt = int(ex["answer"])
            is_correct = common.grade(pred, gt)

            sc = int(ex["score"])
            max_score += sc
            if is_correct:
                total_score += sc
                correct_cnt += 1

            row = common.EvalRow(
                id=int(ex["id"]),
                name=str(ex["name"]),
                question_type=qtype,
                ground_truth=gt,
                model_answer=pred,
                correct=is_correct,
                score=sc,
                raw_output=text,
                problem=ex["problem"],
                review=ex.get("review"),
            )
            f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")

    summary = {
        "backend": "openai",
        "model": args.model,
        "data_dir": getattr(args, "data_dir", "./data"),
        "split": args.split,
        "n": len(ds),
        "correct": correct_cnt,
        "accuracy": (correct_cnt / len(ds)) if len(ds) else 0.0,
        "score": total_score,
        "max_score": max_score,
    }

    print_human_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))



def run_transformers_eval(common: Any, args: Any) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ds = load_eval_split(args)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    kwargs = dict(device_map="auto")
    if bool(getattr(args, "load_in_4bit", False)) or bool(getattr(args, "load_in_8bit", False)):
        kwargs["load_in_4bit"] = bool(getattr(args, "load_in_4bit", False))
        kwargs["load_in_8bit"] = bool(getattr(args, "load_in_8bit", False))

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **kwargs,
    )
    mdl.eval()

    def format_as_chat(prompt: str) -> str:
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            msgs = [{"role": "user", "content": prompt}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt

    @torch.inference_mode()
    def generate(prompt: str) -> str:
        text_in = format_as_chat(prompt)
        inputs = tok(text_in, return_tensors="pt").to(mdl.device)
        do_sample = float(getattr(args, "temperature", 0.0)) > 0.0
        gen = mdl.generate(
            **inputs,
            max_new_tokens=int(args.max_tokens),
            do_sample=do_sample,
            temperature=float(args.temperature) if do_sample else None,
            pad_token_id=tok.eos_token_id,
        )
        out = tok.decode(gen[0], skip_special_tokens=True)
        idx = out.rfind("문제:")
        return out[idx:] if idx != -1 else out

    total_score = 0
    max_score = 0
    correct_cnt = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, desc=f"HF eval ({args.model})"):
            qtype = common.infer_qtype(ex)
            subject = getattr(common, "subject", "math")
            prompt = common.build_prompt(ex["problem"], qtype, subject)

            text = generate(prompt)

            pred = common.extract_final_answer(text, qtype)
            gt = int(ex["answer"])
            is_correct = common.grade(pred, gt)

            sc = int(ex["score"])
            max_score += sc
            if is_correct:
                total_score += sc
                correct_cnt += 1

            row = common.EvalRow(
                id=int(ex["id"]),
                name=str(ex["name"]),
                question_type=qtype,
                ground_truth=gt,
                model_answer=pred,
                correct=is_correct,
                score=sc,
                raw_output=text,
                problem=ex["problem"],
                review=ex.get("review"),
            )
            f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")

    summary = {
        "backend": "transformers",
        "model_name": args.model,
        "data_dir": getattr(args, "data_dir", "./data"),
        "split": args.split,
        "n": len(ds),
        "correct": correct_cnt,
        "accuracy": (correct_cnt / len(ds)) if len(ds) else 0.0,
        "score": total_score,
        "max_score": max_score,
        "max_new_tokens": int(args.max_tokens),
        "temperature": float(getattr(args, "temperature", 0.0)),
        "load_in_4bit": bool(getattr(args, "load_in_4bit", False)),
        "load_in_8bit": bool(getattr(args, "load_in_8bit", False)),
    }

    print_human_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

