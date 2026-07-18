# Continued Pre-Training (CPT): `google/gemma-4-E4B`

This document is the single source of truth for running text-only CPT of `google/gemma-4-E4B` on one node with 8x H200 using NeMo AutoModel.

It includes all runtime fixes validated during setup, including distributed launch gotchas and container version compatibility notes.

## Target setup

- Model: `google/gemma-4-E4B`
- Hardware: `8x H200` (single node)
- Sequence length: `8192`
- Global batch size: `8` (1 sample/GPU)
- Precision: `bf16`
- Distributed strategy: `fsdp2` + CPU offload + activation checkpointing

## Project layout

```text
/userhome/home/ekacareomni/
├── CPT/
│   ├── Automodel/
│   ├── gemma4_e4b_cpt_30b_text_h200x8.yaml
│   ├── checkpoints/
│   └── README.md
└── datasets/
    └── 30B_Text_CPT_Dataset_17_Apr/
        └── megatron_gemma4_e4b/
```

Inside the container this is mounted as:

```text
/workspace -> /userhome/home/ekacareomni
```

## 1) Environment variables

```bash
export HF_TOKEN=<huggingface_token>
export WANDB_API_KEY=<wandb_api_key>   # optional
```

## 2) Launch container

### Recommended (uses your mounted workspace)

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --shm-size=16g \
  -e HF_TOKEN="$HF_TOKEN" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -v /userhome/home/ekacareomni:/workspace \
  -v /userhome/home/ekacareomni/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/CPT/Automodel \
  nvcr.io/nvidia/nemo-automodel:26.02
```

## 3) Install from local source (recommended)

Running from your repo clone is more stable than relying on `/opt/Automodel` in-container defaults.

```bash
pip install -e ".[all]"
pip install liger-kernel
```

## 4) Optional dataset preprocess (one-time)

Skip if `megatron_gemma4_e4b/processed_data_*_text_document*` already exists.

```bash
python tools/preprocess_megatron_dataset.py \
  --input "/workspace/datasets/30B_Text_CPT_Dataset_17_Apr/train-*.parquet" \
  --input-type parquet \
  --text-column text \
  --json-keys text \
  --output-prefix processed_data \
  --output-path /workspace/datasets/30B_Text_CPT_Dataset_17_Apr/megatron_gemma4_e4b \
  --workers 32 \
  --pretrained-model-name-or-path google/gemma-4-E4B \
  --append-eod
```

## 5) YAML expectations

Use `CPT/gemma4_e4b_cpt_30b_text_h200x8.yaml` with:

- `distributed.strategy: fsdp2`
- `distributed.dp_size: 8`
- `distributed.tp_size: 1`
- `distributed.cp_size: 1`
- `distributed.pp_size: 1`

For `26.02`, do not include these keys (unsupported in this image branch):

- `distributed.fsdp2_backward_prefetch_depth`
- `distributed.patch_is_packed_sequence`

## 6) Launch training (preferred command)

Use direct `torchrun` to avoid CLI wrapper inconsistencies:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  /opt/Automodel/nemo_automodel/recipes/llm/train_ft.py \
  -c /workspace/CPT/gemma4_e4b_cpt_30b_text_h200x8.yaml
```

## 7) Alternate CLI command (only if wrapper works in your image)

In some builds, argument order matters:

```bash
automodel pretrain llm -c /workspace/CPT/gemma4_e4b_cpt_30b_text_h200x8.yaml --nproc-per-node 8
```

If wrapper launches with world size 1 (`initializing torch distributed with 1 workers`), switch back to direct `torchrun` (Section 6).

## 8) Known issues and fixes

### A) `RuntimeError: Mesh should not be bigger than default world size 1, but found 8 ranks!`

Cause:
- launcher started with world size 1 while YAML requested `dp_size: 8`.

Fix:
- run with direct `torchrun --nproc-per-node=8` (Section 6).

### B) `ValueError: Unknown options for strategy 'fsdp2': ['fsdp2_backward_prefetch_depth', 'patch_is_packed_sequence']`

Cause:
- these options are not supported by your `26.02` parser path.

Fix:
- remove those two keys from YAML.

### C) `TypeError: check_model_inputs() missing 1 required positional argument: 'func'`

Cause:
- compatibility bug between AutoModel code path and installed `transformers` API shape.

Hotfix (inside container):

```bash
python - <<'PY'
from pathlib import Path
p = Path("/opt/Automodel/nemo_automodel/shared/import_utils.py")
s = p.read_text()
s = s.replace("return check_model_inputs()", "return check_model_inputs")
p.write_text(s)
print("patched:", p)
PY
```

Then relaunch with direct `torchrun`.

### D) `fatal error: Python.h: No such file or directory` when building megatron helpers

Cause:
- Python dev headers missing in the active environment used for native extension build.

Fix options:
- use container/environment where headers are already present for the active python.
- or install dev headers in the image (admin/image-maintainer path).
- or use prebuilt path where that extension is not rebuilt at runtime.

### E) Protobuf/WandB mismatch (`gencode runtime` version error)

Cause:
- incompatible `protobuf` runtime versus WandB-generated stubs.

Fix:
- align protobuf and wandb versions in the container environment before launch.

## 9) Resume training

Re-run the same command from Section 6. Checkpoint resume follows your YAML checkpoint settings.

## 10) Optional checkpoint conversion to HF format

```bash
python /workspace/CPT/convert_automodel_checkpoint_to_hf.py \
  --checkpoint-dir /workspace/CPT/checkpoints/gemma4_e4b_cpt_30b_text/LATEST \
  --output-dir /workspace/CPT/checkpoints/gemma4_e4b_cpt_30b_text_hf/
```

## 11) Quick sanity checks

Inside container:

```bash
python -c "import torch; print(torch.cuda.device_count())"
```

Expected:
- output is `8`

If not 8, container does not see all GPUs, and distributed launch will fail regardless of YAML.

## References

- [NeMo AutoModel LLM recipes](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/recipes/llm)
- [NeMo AutoModel Gemma4 guide](https://docs.nvidia.com/nemo/automodel/latest/guides/vlm/gemma4.html)
- [NeMo AutoModel docs](https://docs.nvidia.com/nemo/automodel/latest/index.html)
