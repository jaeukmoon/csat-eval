"""HuggingFace에서 데이터셋 다운로드"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset, Dataset


# HuggingFace 데이터셋 매핑: split -> (dataset_id, config, transform_fn)
HF_DATASETS = {
    # 수학 데이터셋 (cfpark00/KoreanSAT)
    "2022_math": ("cfpark00/KoreanSAT", None, None),
    "2023_math": ("cfpark00/KoreanSAT", None, None),
    "2024_math": ("cfpark00/KoreanSAT", None, None),
    "2025_math": ("cfpark00/KoreanSAT", None, None),
    # 영어 데이터셋 (mahalisyarifuddin/korean-csat-2025-english)
    "2025_english": ("mahalisyarifuddin/korean-csat-2025-english", None, "english"),
}


def _convert_english_to_standard(examples: dict) -> dict:
    """영어 데이터를 표준 형식으로 변환"""
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
        
        # problem 필드 구성
        problem_parts = [f"{number}. {question}"]
        
        if option_1 and option_2 and option_3 and option_4 and option_5:
            problem_parts.append("\n\n\\begin{itemize}")
            problem_parts.append(f" \\item[1] {option_1}")
            problem_parts.append(f" \\item[2] {option_2}")
            problem_parts.append(f" \\item[3] {option_3}")
            problem_parts.append(f" \\item[4] {option_4}")
            problem_parts.append(f" \\item[5] {option_5}")
            problem_parts.append(" \\end{itemize}")
        
        # score 추출
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


def download_from_huggingface(
    split: str,
    out_path: Path,
    force: bool = False,
    cache_dir: Optional[str] = None,
) -> Optional[Path]:
    """HuggingFace에서 데이터셋 다운로드하여 JSONL로 저장
    
    Returns:
        Path if successful, None if split not available in HF
    """
    if split not in HF_DATASETS:
        return None
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[huggingface] Already exists: {out_path}")
        return out_path
    
    dataset_id, config, transform = HF_DATASETS[split]
    
    load_kwargs = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    
    try:
        ds_dict = load_dataset(dataset_id, config, **load_kwargs)
    except Exception as e:
        print(f"[huggingface] Failed to load {dataset_id}: {e}")
        return None
    
    # split 이름으로 데이터셋 찾기
    if split in ds_dict:
        ds = ds_dict[split]
    elif "train" in ds_dict:
        ds = ds_dict["train"]
    else:
        # 첫 번째 split 사용
        first_split = list(ds_dict.keys())[0]
        ds = ds_dict[first_split]
    
    # 영어 데이터셋 변환
    if transform == "english":
        converted_data = _convert_english_to_standard(ds.to_dict())
        ds = Dataset.from_dict(converted_data)
    
    # JSONL로 저장
    ds.to_json(str(out_path), orient="records", lines=True, force_ascii=False)
    print(f"[huggingface] Created: {out_path}")
    return out_path


def is_available_on_huggingface(split: str) -> bool:
    """해당 split이 HuggingFace에서 다운로드 가능한지 확인"""
    return split in HF_DATASETS
