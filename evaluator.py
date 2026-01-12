from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, List, Dict

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
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
    """OpenAI API를 사용한 비동기 병렬 평가"""
    asyncio.run(_run_openai_eval_async(common, args))


async def _run_openai_eval_async(common: Any, args: Any) -> None:
    import os
    from openai import AsyncOpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

    client = AsyncOpenAI(api_key=api_key)

    ds = load_eval_split(args)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # 동시 요청 수 제한 (rate limit 고려)
    semaphore = asyncio.Semaphore(getattr(args, "concurrency", 10))
    file_lock = asyncio.Lock()  # 파일 쓰기 동기화용

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def call_openai(prompt: str) -> str:
        async with semaphore:
            messages = [{"role": "user", "content": prompt}]
            completion = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                reasoning_effort="high",
            )
            return completion.choices[0].message.content or ""

    # 통계를 위한 공유 변수
    total_score = 0
    max_score = 0
    correct_cnt = 0
    completed = 0
    total = len(ds)

    # 파일 초기화 (덮어쓰기 모드)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        pass

    async def process_example(ex: Dict) -> Dict:
        nonlocal total_score, max_score, correct_cnt, completed
        
        qtype = common.infer_qtype(ex)
        subject = getattr(common, "subject", "math")
        prompt = common.build_prompt(ex["problem"], qtype, subject)

        text = await call_openai(prompt)

        pred = common.extract_final_answer(text, qtype)
        gt = int(ex["answer"])
        is_correct = common.grade(pred, gt)
        sc = int(ex["score"])

        result = {
            "id": int(ex["id"]),
            "name": str(ex["name"]),
            "question_type": qtype,
            "ground_truth": gt,
            "model_answer": pred,
            "correct": is_correct,
            "score": sc,
            "raw_output": text,
            "problem": ex["problem"],
            "review": ex.get("review"),
        }

        # 결과를 즉시 파일에 쓰고 출력
        async with file_lock:
            with open(args.out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            # 통계 업데이트
            max_score += sc
            if is_correct:
                total_score += sc
                correct_cnt += 1
            completed += 1
            
            # 중간 결과 출력
            status = "✓" if is_correct else "✗"
            acc_pct = (correct_cnt / completed * 100.0) if completed > 0 else 0.0
            score_pct = (total_score / max_score * 100.0) if max_score > 0 else 0.0
            print(f"[{completed}/{total}] 문항 {result['id']}: 예측={pred if pred is not None else '(포기)'}, 정답={gt} {status} "
                  f"(점수: {total_score}/{max_score} ({score_pct:.1f}%), 정확도: {correct_cnt}/{completed} ({acc_pct:.1f}%))")

        return result

    # 모든 태스크를 병렬로 실행
    tasks = [process_example(ex) for ex in ds]
    results = await atqdm.gather(*tasks, desc=f"OpenAI eval ({args.model})")

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


def run_gpt_oss_eval(common: Any, args: Any) -> None:
    """gpt-oss/Qwen 모델을 로컬에서 로드하여 평가 (reasoning_effort/enable_thinking 지원)"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ds = load_eval_split(args)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    kwargs = dict(device_map="auto")
    if bool(getattr(args, "load_in_4bit", False)) or bool(getattr(args, "load_in_8bit", False)):
        kwargs["load_in_4bit"] = bool(getattr(args, "load_in_4bit", False))
        kwargs["load_in_8bit"] = bool(getattr(args, "load_in_8bit", False))

    tok = AutoTokenizer.from_pretrained(args.model, add_eos_token=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        **kwargs,
    )
    mdl.eval()

    # Qwen 모델 여부 판별
    model_lower = args.model.lower()
    is_qwen = "qwen" in model_lower

    @torch.inference_mode()
    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        reasoning_effort = getattr(args, "reasoning_effort", "high")
        
        if is_qwen:
            # Qwen: tokenize=False로 문자열 생성 후 별도 토큰화
            text_in = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(reasoning_effort == "high")
            )
            inputs = tok(text_in, return_tensors="pt").to(mdl.device)
        else:
            # GPT-OSS: reasoning_effort 사용
            chat_kwargs = {}
            if reasoning_effort != "none":
                chat_kwargs["reasoning_effort"] = reasoning_effort
            
            inputs = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                **chat_kwargs
            )
            # 텐서를 dict로 변환 (generate에서 사용)
            if isinstance(inputs, torch.Tensor):
                inputs = {"input_ids": inputs.to(mdl.device)}
            else:
                inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        
        # generate 호출 (do_sample 처리)
        temperature = float(getattr(args, "temperature", 0.0))  # GPT-OSS 기본값 0.0
        do_sample = temperature > 0
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=int(args.max_tokens),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tok.eos_token_id,
        )
        decoded = tok.decode(outputs[0], skip_special_tokens=False)
        return decoded

    total_score = 0
    max_score = 0
    correct_cnt = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, desc=f"gpt-oss eval ({args.model})"):
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
        "backend": "gpt-oss",
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
        "reasoning_effort": getattr(args, "reasoning_effort", "high"),
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


def run_vllm_eval(common: Any, args: Any) -> None:
    """vLLM 서버를 통한 비동기 병렬 평가 (OpenAI 호환 API)"""
    asyncio.run(_run_vllm_eval_async(common, args))


async def _run_vllm_eval_async(common: Any, args: Any) -> None:
    from openai import AsyncOpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    vllm_base_url = getattr(args, "vllm_base_url", None)
    vllm_model_id = getattr(args, "vllm_model_id", None)
    
    if not vllm_base_url:
        raise RuntimeError("--vllm_base_url을 설정하세요. (예: http://localhost:8000/v1)")
    if not vllm_model_id:
        raise RuntimeError("--vllm_model_id를 설정하세요.")

    # vLLM 서버는 OpenAI 호환 API를 제공
    client = AsyncOpenAI(
        base_url=vllm_base_url,
        api_key="dummy-key",  # vLLM은 API 키가 필요 없지만 필수 필드
    )

    ds = load_eval_split(args)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # 동시 요청 수 제한
    semaphore = asyncio.Semaphore(getattr(args, "concurrency", 20))
    file_lock = asyncio.Lock()  # 파일 쓰기 동기화용

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def call_vllm(prompt: str) -> str:
        async with semaphore:
            messages = [{"role": "user", "content": prompt}]
            temperature = float(getattr(args, "temperature", 0.0))  # vLLM 기본값 0.0
            completion = await client.chat.completions.create(
                model=vllm_model_id,
                messages=messages,
                max_tokens=int(args.max_tokens),
                temperature=temperature,
            )
            return completion.choices[0].message.content or ""

    # 통계를 위한 공유 변수
    total_score = 0
    max_score = 0
    correct_cnt = 0
    completed = 0
    total = len(ds)

    # 파일 초기화 (덮어쓰기 모드)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        pass

    async def process_example(ex: Dict) -> Dict:
        nonlocal total_score, max_score, correct_cnt, completed
        
        qtype = common.infer_qtype(ex)
        subject = getattr(common, "subject", "math")
        prompt = common.build_prompt(ex["problem"], qtype, subject)

        text = await call_vllm(prompt)

        pred = common.extract_final_answer(text, qtype)
        gt = int(ex["answer"])
        is_correct = common.grade(pred, gt)
        sc = int(ex["score"])

        result = {
            "id": int(ex["id"]),
            "name": str(ex["name"]),
            "question_type": qtype,
            "ground_truth": gt,
            "model_answer": pred,
            "correct": is_correct,
            "score": sc,
            "raw_output": text,
            "problem": ex["problem"],
            "review": ex.get("review"),
        }

        # 결과를 즉시 파일에 쓰고 출력
        async with file_lock:
            with open(args.out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            # 통계 업데이트
            max_score += sc
            if is_correct:
                total_score += sc
                correct_cnt += 1
            completed += 1
            
            # 중간 결과 출력
            status = "✓" if is_correct else "✗"
            acc_pct = (correct_cnt / completed * 100.0) if completed > 0 else 0.0
            score_pct = (total_score / max_score * 100.0) if max_score > 0 else 0.0
            print(f"[{completed}/{total}] 문항 {result['id']}: 예측={pred if pred is not None else '(포기)'}, 정답={gt} {status} "
                  f"(점수: {total_score}/{max_score} ({score_pct:.1f}%), 정확도: {correct_cnt}/{completed} ({acc_pct:.1f}%))")

        return result

    # 모든 태스크를 병렬로 실행
    tasks = [process_example(ex) for ex in ds]
    results = await atqdm.gather(*tasks, desc=f"vLLM eval ({vllm_model_id})")

    summary = {
        "backend": "vllm",
        "vllm_base_url": vllm_base_url,
        "model": vllm_model_id,
        "data_dir": getattr(args, "data_dir", "./data"),
        "split": args.split,
        "n": len(ds),
        "correct": correct_cnt,
        "accuracy": (correct_cnt / len(ds)) if len(ds) else 0.0,
        "score": total_score,
        "max_score": max_score,
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }

    print_human_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
