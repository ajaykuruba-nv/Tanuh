#!/usr/bin/env python3
import argparse
import glob
import json
import os
import random

import pyarrow.parquet as pq


def iter_rows(parquet_glob: str, prompt_col: str, label_col: str):
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise RuntimeError(f"No parquet files matched: {parquet_glob}")

    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(columns=[prompt_col, label_col], batch_size=10000):
            prompts = batch.column(0)
            labels = batch.column(1)
            for p, l in zip(prompts, labels):
                prompt = (p.as_py() or "").strip()
                label = (l.as_py() or "").strip()
                if not prompt or not label:
                    continue
                yield {"prompt": prompt, "label": label}


def main():
    parser = argparse.ArgumentParser(description="Convert parquet prompt/label SFT data to JSONL train/validation files.")
    parser.add_argument(
        "--input-glob",
        default="/userhome/home/ekacarecpt/datasets/sft_dataset/structuring_dataset_with_qna/*.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="/userhome/home/ekacarecpt/datasets/sft_dataset/structuring_dataset_with_qna_jsonl",
    )
    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "validation.jsonl")

    rng = random.Random(args.seed)
    n_train = 0
    n_val = 0

    with open(train_path, "w", encoding="utf-8") as f_train, open(val_path, "w", encoding="utf-8") as f_val:
        for row in iter_rows(args.input_glob, args.prompt_col, args.label_col):
            if rng.random() < args.val_ratio:
                f_val.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_val += 1
            else:
                f_train.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_train += 1

    print(f"Wrote train: {train_path} ({n_train} rows)")
    print(f"Wrote valid: {val_path} ({n_val} rows)")


if __name__ == "__main__":
    main()
