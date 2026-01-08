from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Literal
from csv_results import build_split_csv

from evaluator import run_openai_eval, run_transformers_eval

QType = Literal["mc", "sa"]

BOXED_RE = re.compile(r"\\boxed\{(-?\d+)\}")
FINAL_RE = re.compile(r"FINAL\s*[:：]\s*(-?\d+)", re.IGNORECASE)
ANS_RE = re.compile(r"(?:정답|답|Answer)\s*[:：]?\s*(-?\d+)", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")
MC_RE = re.compile(r"\b([1-5])\b")


def infer_qtype(example: Dict[str, Any]) -> QType:
    a = int(example["answer"])
    return "mc" if 1 <= a <= 5 else "sa"


def build_prompt(problem_text: str, qtype: QType) -> str:
    if qtype == "mc":
        rule = "선택지는 1~5 중 하나다. 마지막 줄에만 FINAL: <1~5> 형태로 정답만 써라."
    else:
        rule = "단답형이므로 정수로 답하라. 마지막 줄에만 FINAL: <정수> 형태로 정답만 써라."
    return (
        "다음은 한국 수능 수학 문제다.\n"
        f"{rule}\n\n"
        "문제:\n"
        f"{problem_text}\n"
    )


def extract_final_answer(text: str, qtype: QType) -> Optional[int]:
    if not text:
        return None

    m = FINAL_RE.search(text)
    if m:
        val = int(m.group(1))
        return val if (qtype == "sa" or 1 <= val <= 5) else None

    m = ANS_RE.search(text)
    if m:
        val = int(m.group(1))
        return val if (qtype == "sa" or 1 <= val <= 5) else None

    m = BOXED_RE.search(text)
    if m:
        val = int(m.group(1))
        return val if (qtype == "sa" or 1 <= val <= 5) else None

    if qtype == "mc":
        all_mc = MC_RE.findall(text)
        return int(all_mc[-1]) if all_mc else None

    all_int = INT_RE.findall(text)
    return int(all_int[-1]) if all_int else None


def grade(pred: Optional[int], gold: int) -> bool:
    return (pred is not None) and (int(pred) == int(gold))


@dataclass
class EvalRow:
    id: int
    name: str
    question_type: QType
    ground_truth: int
    model_answer: Optional[int]
    correct: bool
    score: int
    raw_output: str
    problem: str
    review: Optional[str]


def _model_basename(model: str) -> str:
    s = (model or "").strip().rstrip("/")
    base = s.split("/")[-1] if "/" in s else s
    base = "".join(c if c.isalnum() or c in "._-+" else "_" for c in base)
    return base or "model"


def _split_to_dirs(split: str) -> tuple[str, str]:
    if "_" not in split:
        return split, "unknown"
    year, subject = split.split("_", 1)
    year = year.strip() or "unknown"
    subject = subject.strip() or "unknown"
    return year, subject


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["openai", "transformers"], default="openai")
    ap.add_argument("--source", choices=["local"], default="local")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--split", default="2025_math")
    ap.add_argument("--max_samples", type=int, default=0)

    ap.add_argument("--out_jsonl", default="")
    # "/group-volume/models/meta-llama/Llama-3.3-70B-Instruct"
    # "/group-volume/models/LGAI-EXAONE/K-EXAONE-236B-A23B"
    ap.add_argument("--model",default="gpt-5.1", help="openai model id OR hf model_name (mode에 따라 해석)")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="high")

    ap.add_argument("--store", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")

    args = ap.parse_args()

    model_name = _model_basename(args.model)
    year, subject = _split_to_dirs(args.split)

    results_dir = Path("./results") / year / subject
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.out_jsonl:
        args.out_jsonl = str(results_dir / f"{model_name}.jsonl")

    return args


def main():
    args = parse_args()

    common = SimpleNamespace(
        infer_qtype=infer_qtype,
        build_prompt=build_prompt,
        extract_final_answer=extract_final_answer,
        grade=grade,
        EvalRow=EvalRow,
    )

    if args.mode == "openai":
        # if not os.getenv("OPENAI_API_KEY"):
        #     raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")
        run_openai_eval(common, args)
        csv_path = build_split_csv(split=args.split, results_root="./results", out_dir=".")
        print(f"CSV saved: {csv_path}") 
        return

    run_transformers_eval(common, args)
    csv_path = build_split_csv(split=args.split, results_root="./results", out_dir=".")
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
