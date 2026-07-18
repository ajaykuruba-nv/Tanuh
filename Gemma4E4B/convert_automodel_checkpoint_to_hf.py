#!/usr/bin/env python3
"""
Convert a NeMo AutoModel sharded safetensors checkpoint to standard Hugging Face format.

Example:
  python convert_automodel_checkpoint_to_hf.py \
    --checkpoint /userhome/home/ekacareomni/CPT/checkpoints/gemma4_e4b_it_structuring_qna_sft_withconstant_lr/LATEST \
    --output /userhome/home/ekacareomni/CPT/checkpoints/gemma4_e4b_it_structuring_qna_sft_withconstant_lr/LATEST/model/consolidated
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AUTOMODEL_DIR = SCRIPT_DIR / "Automodel"
if str(AUTOMODEL_DIR) not in sys.path:
    sys.path.insert(0, str(AUTOMODEL_DIR))


def _parse_args() -> argparse.Namespace:
    default_checkpoint = (
        "/userhome/home/ekacareomni/CPT/checkpoints/"
        "gemma4_e4b_it_structuring_qna_sft_withconstant_lr/LATEST"
    )
    parser = argparse.ArgumentParser(
        description="Consolidate an AutoModel sharded checkpoint into Hugging Face format."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(default_checkpoint),
        help="Path to checkpoint root (e.g. .../LATEST or .../epoch_x_step_y).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HF directory. Defaults to <checkpoint>/model/consolidated.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=5,
        help="Number of worker threads used for writing output shards.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
        help="Distributed backend when launched with torchrun.",
    )
    return parser.parse_args()


def _resolve_model_paths(checkpoint_dir: Path, output_dir: Path | None) -> tuple[Path, Path, Path]:
    ckpt = checkpoint_dir.expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt}")

    model_dir = ckpt / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Expected model directory at: {model_dir}")

    hf_metadata_dir = model_dir / ".hf_metadata"
    if not hf_metadata_dir.is_dir():
        raise FileNotFoundError(f"Expected metadata directory at: {hf_metadata_dir}")

    if output_dir is None:
        out_dir = model_dir / "consolidated"
    else:
        out_dir = output_dir.expanduser().resolve()
    return model_dir, hf_metadata_dir, out_dir


def _copy_metadata(hf_metadata_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in hf_metadata_dir.iterdir():
        if src.name == "fqn_to_file_index_mapping.json":
            continue
        dst = output_dir / src.name
        if src.is_file():
            shutil.copy2(src, dst)


def main() -> None:
    args = _parse_args()
    model_dir, hf_metadata_dir, output_dir = _resolve_model_paths(args.checkpoint, args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torch.distributed as dist
        from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
            consolidate_safetensors_files_on_every_rank,
        )
        from nemo_automodel.components.distributed.init_utils import (
            get_rank_safe,
            get_world_size_safe,
            initialize_distributed,
        )
    except ImportError as e:
        raise ImportError(
            "Missing dependencies. Run this script in the same Python environment used for AutoModel "
            "(must include torch and nemo_automodel)."
        ) from e

    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.device_count() > 0 else "gloo"
    initialize_distributed(backend)

    mapping_file = hf_metadata_dir / "fqn_to_file_index_mapping.json"
    if not mapping_file.is_file():
        raise FileNotFoundError(f"Missing FQN mapping file: {mapping_file}")

    with mapping_file.open("r", encoding="utf-8") as f:
        fqn_to_index_mapping = json.load(f)

    consolidate_safetensors_files_on_every_rank(
        input_dir=str(model_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping=fqn_to_index_mapping,
        num_threads=args.num_threads,
    )

    if get_world_size_safe() > 1:
        dist.barrier()

    if get_rank_safe() == 0:
        _copy_metadata(hf_metadata_dir, output_dir)
        print(f"HF checkpoint written to: {output_dir}")

    if get_world_size_safe() > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
