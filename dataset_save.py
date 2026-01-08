from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="cfpark00/KoreanSAT")
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
        "naming": "{split}.jsonl",
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

        out_path = out_dir / f"{split}.jsonl"
        if out_path.exists() and not args.force:
            continue

        ds.to_json(str(out_path), orient="records", lines=True, force_ascii=False)

    print(f"Saved JSONL to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
