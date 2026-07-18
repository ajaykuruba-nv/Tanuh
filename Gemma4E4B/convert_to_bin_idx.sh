#!/usr/bin/env bash
set -euo pipefail

# Convert parquet shards to Megatron-Core indexed dataset (.bin/.idx)
# using NVIDIA NeMo AutoModel's preprocessing utility.

# ====== EDIT IF NEEDED ======
AUTOMODEL_DIR="/userhome/home/ekacarecpt/Automodel"
INPUT_GLOB="/userhome/home/ekacarecpt/datasets/sft_dataset/structuring_dataset_with_qna/*.parquet"
OUTPUT_DIR="/userhome/home/ekacarecpt/datasets/sft_dataset/structuring_dataset_with_qna_megatron_binidx"
OUTPUT_PREFIX="structuring_qna"
# Use local tokenizer to avoid gated-model auth/download failures.
TOKENIZER="/userhome/home/ekacarecpt/models/gemma4_cpt/5pj19ug4/LATEST/model/consolidated"
# If your parquet has a single text column, set TEXT_COLUMN to that field and
# leave COMBINE_PROMPT_LABEL=0.
TEXT_COLUMN="text"
# This dataset uses prompt/label columns; when enabled, creates temporary JSONL
# shards with text = prompt + "\n" + label before tokenization.
COMBINE_PROMPT_LABEL=1
PROMPT_COLUMN="prompt"
LABEL_COLUMN="label"
WORKERS=16
# ===========================

mkdir -p "$OUTPUT_DIR"
cd "$AUTOMODEL_DIR"

# Use existing virtualenv if active; otherwise create/use a local one.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="python"
  PIP_BIN="pip"
else
  VENV_DIR="$AUTOMODEL_DIR/.venv_preprocess"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
  PIP_BIN="$VENV_DIR/bin/pip"
fi

# Ensure pip exists for the selected interpreter (some envs are created without pip).
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi

# If pip is still unavailable, switch to a dedicated preprocessing venv.
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  VENV_DIR="$AUTOMODEL_DIR/.venv_preprocess"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
  PIP_BIN="$VENV_DIR/bin/pip"
  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m ensurepip --upgrade
  fi
fi

# Required for parquet loading + tokenizer + AutoModel dataset imports
"$PYTHON_BIN" -m pip install --upgrade pip
"$PIP_BIN" install pyarrow transformers sentencepiece datasets

if [[ "$COMBINE_PROMPT_LABEL" -eq 1 ]]; then
  TMP_JSONL_DIR="$OUTPUT_DIR/_tmp_jsonl"
  mkdir -p "$TMP_JSONL_DIR"

  "$PYTHON_BIN" - <<PY
import glob
import json
import os
from pyarrow import parquet as pq

input_glob = """$INPUT_GLOB"""
prompt_col = """$PROMPT_COLUMN"""
label_col = """$LABEL_COLUMN"""
out_dir = """$TMP_JSONL_DIR"""

files = sorted(glob.glob(input_glob))
if not files:
    raise SystemExit(f"No parquet files matched: {input_glob}")

for i, fp in enumerate(files):
    out_fp = os.path.join(out_dir, f"part_{i:05d}.jsonl")
    pf = pq.ParquetFile(fp)
    with open(out_fp, "w", encoding="utf-8") as fout:
        for batch in pf.iter_batches(columns=[prompt_col, label_col], batch_size=10000):
            prompts = batch.column(0)
            labels = batch.column(1)
            for p, l in zip(prompts, labels):
                prompt = p.as_py() or ""
                label = l.as_py() or ""
                text = prompt + "\n" + label
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
PY

  "$PYTHON_BIN" tools/preprocess_megatron_dataset.py \
    --input "$TMP_JSONL_DIR/*.jsonl" \
    --input-type json \
    --json-keys text \
    --output-prefix "$OUTPUT_PREFIX" \
    --output-path "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --pretrained-model-name-or-path "$TOKENIZER" \
    --append-eod
else
  "$PYTHON_BIN" tools/preprocess_megatron_dataset.py \
    --input "$INPUT_GLOB" \
    --input-type parquet \
    --json-keys text \
    --text-column "$TEXT_COLUMN" \
    --output-prefix "$OUTPUT_PREFIX" \
    --output-path "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --pretrained-model-name-or-path "$TOKENIZER" \
    --append-eod
fi

echo "Done. Generated files:"
ls -lh "$OUTPUT_DIR"/*_text_document.{bin,idx}
