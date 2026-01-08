from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _get_gt(it: Dict[str, Any]) -> Optional[int]:
    v = it.get("ground_truth", it.get("gold", it.get("answer", None)))
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _get_model_answer(it: Dict[str, Any]) -> Any:
    if "model_answer" in it:
        return it.get("model_answer", None)
    return it.get("pred", None)


def _format_answer(v: Any) -> str:
    if v is None:
        return "(포기)"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    s = str(v).strip()
    if not s:
        return "(포기)"
    return s


def _split_to_year_subject(split: str) -> tuple[str, str]:
    if "_" not in split:
        return split, "unknown"
    year, subject = split.split("_", 1)
    year = year.strip() or "unknown"
    subject = subject.strip() or "unknown"
    return year, subject


def build_split_csv(
    split: str,
    results_root: str = "./results",
    out_dir: str = ".",
    out_name: Optional[str] = None,
) -> str:
    year, subject = _split_to_year_subject(split)
    subject_dir = Path(results_root) / year / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {subject_dir}")

    model_files = sorted([p for p in subject_dir.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)
    if not model_files:
        raise FileNotFoundError(f"No model jsonl found under: {subject_dir}")

    model_names = [p.stem for p in model_files]

    per_model: Dict[str, Dict[int, Dict[str, Any]]] = {}
    question_meta: Dict[int, Dict[str, Any]] = {}

    for mf in model_files:
        items = _read_jsonl(mf)
        idx: Dict[int, Dict[str, Any]] = {}
        for it in items:
            qid = int(it["id"])
            idx[qid] = it
            if qid not in question_meta:
                gt = _get_gt(it)
                question_meta[qid] = {
                    "id": qid,
                    "ground_truth": gt,
                    "score": int(it.get("score", 0)),
                }
        per_model[mf.stem] = idx

    qids = sorted(question_meta.keys())
    max_score = sum(int(question_meta[qid].get("score", 0)) for qid in qids)

    totals: Dict[str, int] = {mn: 0 for mn in model_names}
    for mn in model_names:
        for qid in qids:
            it = per_model.get(mn, {}).get(qid, None)
            if not it:
                continue
            if bool(it.get("correct", False)):
                totals[mn] += int(it.get("score", 0))

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if out_name is None:
        out_name = f"{split}_수능 LLM 풀이 결과.csv"

    out_path = out_dir_p / out_name

    header = ["문항 번호", "정답"] + model_names

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)

        for qid in qids:
            gt = question_meta[qid].get("ground_truth", None)
            row = [str(qid), "" if gt is None else str(gt)]

            for mn in model_names:
                it = per_model.get(mn, {}).get(qid, None)
                if it is None:
                    row.append("(포기)")
                    continue
                ans = _get_model_answer(it)
                row.append(_format_answer(ans))

            w.writerow(row)

        total_row = ["총점", str(max_score)] + [str(totals[mn]) for mn in model_names]
        w.writerow(total_row)

    return str(out_path)
