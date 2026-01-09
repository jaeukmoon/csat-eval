from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mahalisyarifuddin/korean-csat-2025-english")
    p.add_argument("--config", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--splits", default="*")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def resolve_splits(ds_dict, splits_arg: str) -> List[str]:
    if splits_arg.strip() in ["*", ""]:
        return list(ds_dict.keys())
    return [s.strip() for s in splits_arg.split(",") if s.strip()]


def convert_english_to_standard_format(examples: dict) -> dict:
    """영어 데이터를 수학 데이터와 동일한 형식으로 변환"""
    converted = {
        "id": [],
        "name": [],
        "problem": [],
        "answer": [],
        "score": [],
        "review": [],
    }
    
    for i in range(len(examples["number"])):
        number = examples["number"][i]
        question = examples["question"][i]
        option_1 = examples.get("option_1", [None] * len(examples["number"]))[i]
        option_2 = examples.get("option_2", [None] * len(examples["number"]))[i]
        option_3 = examples.get("option_3", [None] * len(examples["number"]))[i]
        option_4 = examples.get("option_4", [None] * len(examples["number"]))[i]
        option_5 = examples.get("option_5", [None] * len(examples["number"]))[i]
        correct_option = examples["correct_option"][i]
        
        # problem 필드 구성: question + 선택지
        # 수학 데이터 형식과 유사하게 LaTeX 형식으로 선택지 추가
        problem_parts = [f"{number}. {question}"]
        
        # 선택지 추가 (LaTeX itemize 형식으로)
        if option_1 and option_2 and option_3 and option_4 and option_5:
            problem_parts.append("\n\n\\begin{itemize}")
            problem_parts.append(f" \\item[1] {option_1}")
            problem_parts.append(f" \\item[2] {option_2}")
            problem_parts.append(f" \\item[3] {option_3}")
            problem_parts.append(f" \\item[4] {option_4}")
            problem_parts.append(f" \\item[5] {option_5}")
            problem_parts.append(" \\end{itemize}")
        
        # score는 기본값 2점 (영어 문제는 보통 2점 또는 3점)
        # 문제 텍스트에서 [3점] 같은 패턴이 있으면 추출, 없으면 2점
        score = 2
        if "[3점]" in question or "[3 점]" in question:
            score = 3
        elif "[2점]" in question or "[2 점]" in question:
            score = 2
        
        converted["id"].append(int(number))
        converted["name"].append(str(number))
        converted["problem"].append("".join(problem_parts))
        converted["answer"].append(int(correct_option))
        converted["score"].append(score)
        converted["review"].append(None)
    
    return converted


def main():
    args = parse_args()
    out_dir = Path("./data")
    out_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {}
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    if args.revision:
        load_kwargs["revision"] = args.revision

    ds_dict = load_dataset(args.dataset, args.config, **load_kwargs)
    splits = resolve_splits(ds_dict, args.splits)

    meta = {
        "dataset": args.dataset,
        "config": args.config,
        "revision": args.revision,
        "format": "jsonl",
        "splits": splits,
        "max_samples": args.max_samples,
        "out_dir": str(out_dir.resolve()),
        "naming": "2025_english.jsonl",
    }
    (out_dir / "export_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for split in splits:
        if split not in ds_dict:
            raise KeyError(f"Split not found: {split}. Available: {list(ds_dict.keys())}")

        ds = ds_dict[split]
        if args.max_samples and args.max_samples > 0:
            ds = ds.select(range(min(args.max_samples, len(ds))))

        out_path = out_dir / "2025_english.jsonl"
        if out_path.exists() and not args.force:
            continue

        # 영어 데이터를 표준 형식으로 변환
        converted_data = convert_english_to_standard_format(ds.to_dict())
        from datasets import Dataset
        converted_ds = Dataset.from_dict(converted_data)

        # Hugging Face datasets의 to_json 메서드 사용
        converted_ds.to_json(str(out_path), orient="records", lines=True, force_ascii=False)

    print(f"Saved JSONL to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()