#!/usr/bin/env python3

import glob
import argparse
import random
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def build_gemma_sft_text(prompt: str, label: str):
    prompt = prompt.strip()
    label = label.strip()

    user_part = (
        "<start_of_turn>user\n"
        f"{prompt}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    full_text = (
        user_part
        + f"{label}"
        + "<end_of_turn>"
    )

    return user_part, full_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise RuntimeError(f"No parquet files found for: {args.input_glob}")

    print(f"Found {len(files)} parquet files")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    total_rows = 0
    empty_prompt = 0
    empty_label = 0
    over_seq_len = 0
    checked = 0

    examples = []

    for fp in files:
        pf = pq.ParquetFile(fp)

        schema_cols = pf.schema.names
        if args.prompt_col not in schema_cols:
            raise RuntimeError(f"Missing prompt column `{args.prompt_col}` in {fp}")
        if args.label_col not in schema_cols:
            raise RuntimeError(f"Missing label column `{args.label_col}` in {fp}")

        for batch in pf.iter_batches(
            columns=[args.prompt_col, args.label_col],
            batch_size=1000,
        ):
            prompts = batch.column(0)
            labels = batch.column(1)

            for p, l in zip(prompts, labels):
                total_rows += 1

                prompt = p.as_py() or ""
                label = l.as_py() or ""

                if not prompt.strip():
                    empty_prompt += 1
                    continue

                if not label.strip():
                    empty_label += 1
                    continue

                if len(examples) < args.num_samples:
                    examples.append((prompt, label))

                user_part, full_text = build_gemma_sft_text(prompt, label)

                user_ids = tokenizer(
                    user_part,
                    add_special_tokens=False,
                )["input_ids"]

                full_ids = tokenizer(
                    full_text,
                    add_special_tokens=False,
                )["input_ids"]

                if len(full_ids) > args.seq_length:
                    over_seq_len += 1

                # Correct SFT labels:
                # prompt/user tokens ignored with -100
                # assistant answer tokens trained
                labels_masked = [-100] * len(user_ids) + full_ids[len(user_ids):]

                if len(labels_masked) != len(full_ids):
                    raise RuntimeError("Label mask length mismatch")

                # Sanity check: prompt tokens must be ignored
                if any(x != -100 for x in labels_masked[:len(user_ids)]):
                    raise RuntimeError("Prompt tokens are not masked correctly")

                checked += 1

    print("\n====== DATA SUMMARY ======")
    print(f"Total rows: {total_rows}")
    print(f"Valid checked rows: {checked}")
    print(f"Empty prompts: {empty_prompt}")
    print(f"Empty labels: {empty_label}")
    print(f"Rows over seq_length={args.seq_length}: {over_seq_len}")

    if total_rows > 0:
        print(f"Over-length %: {100 * over_seq_len / total_rows:.2f}%")

    print("\n====== SAMPLE CHECK ======")
    sample_prompt, sample_label = random.choice(examples)
    user_part, full_text = build_gemma_sft_text(sample_prompt, sample_label)

    user_ids = tokenizer(user_part, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    labels_masked = [-100] * len(user_ids) + full_ids[len(user_ids):]

    print("\n--- Prompt ---")
    print(sample_prompt[:1000])

    print("\n--- Label ---")
    print(sample_label[:1000])

    print("\n--- Gemma formatted full text ---")
    print(full_text[:1500])

    print("\n--- Token stats ---")
    print(f"Prompt tokens masked out: {len(user_ids)}")
    print(f"Assistant/labeled tokens trained: {len(full_ids) - len(user_ids)}")
    print(f"Total tokens: {len(full_ids)}")

    print("\n====== IMPORTANT RESULT ======")
    print("This script creates correct SFT labels like:")
    print("prompt/user tokens     -> -100 ignored")
    print("assistant/model tokens -> trained")
    print()
    print("If your bin/idx was created using `text = prompt + '\\n' + label`,")
    print("then your current MegatronPretraining path does NOT preserve this masking.")
    print("That means it is next-token training on prompt+label, not clean SFT.")


if __name__ == "__main__":
    main()