#!/usr/bin/env python3

import glob
import argparse
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise RuntimeError(f"No files matched: {args.input_glob}")

    seen = 0

    for fp in files:
        pf = pq.ParquetFile(fp)

        for batch in pf.iter_batches(
            columns=[args.prompt_col, args.label_col],
            batch_size=1000,
        ):
            prompts = batch.column(0)
            labels = batch.column(1)

            for p, l in zip(prompts, labels):
                if seen == args.index:
                    prompt = p.as_py() or ""
                    label = l.as_py() or ""

                    print("=" * 80)
                    print(f"FILE: {fp}")
                    print(f"GLOBAL INDEX: {seen}")
                    print("=" * 80)

                    print("\n--- PROMPT ---")
                    print(prompt)

                    print("\n--- LABEL ---")
                    print(label)

                    print("\n--- COMBINED TEXT USED IN YOUR BIN/IDX SCRIPT ---")
                    print(prompt + "\n" + label)

                    return

                seen += 1

    raise RuntimeError(f"Index {args.index} out of range. Total rows seen: {seen}")


if __name__ == "__main__":
    main()