# Megatron Bridge 16-node workshop runbook

This directory contains a cluster-specific continued-pretraining benchmark for
[NVIDIA NeMo Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).
It launches Nemotron-3-Nano on 16 nodes / 128 GPUs through Slurm, Pyxis/Enroot,
and the Megatron Bridge installation included in the NeMo `26.06` container.

> **Workshop warning:** the filenames still say `GBS2048`, but the checked-in
> YAML uses an effective global batch size of **1920**. The launcher also has
> `#SBATCH --nodes=20`, while this recipe and its topology are for **16 nodes**.
> Use the workshop submission command below, which explicitly requests 16
> nodes. Do not submit the script with a bare `sbatch` until those names and
> defaults are reconciled.

## What is in this directory

| File | Purpose |
| --- | --- |
| `phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.sh` | Slurm, container, cache, NCCL, preflight, and `torchrun` launcher |
| `phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.yaml` | Workshop run, model, parallelism, data, logging, and profiling configuration |
| `phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.py` | Builds the Nemotron-3-Nano Megatron Bridge recipe and starts pretraining |

The Python entry point uses `AutoBridge.from_hf_pretrained(...)` to load the
local Hugging Face model definition and convert it to a Megatron provider. The
NeMo container is the recommended installation path in the official
[Megatron Bridge documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/).
This launcher already adds `/opt/Megatron-Bridge/src` and `/opt/megatron-lm` to
`PYTHONPATH`; workshop participants do not need a separate pip installation.

## Effective workshop configuration

| Setting | Effective value |
| --- | --- |
| Container | `nvcr.io/nvidia/nemo:26.06` |
| Allocation | 16 nodes x 8 GPUs = 128 ranks |
| Model parallelism | TP=1, EP=8, PP=1, CP=1 |
| Data parallelism | DP=16; each EP group stays within one 8-GPU node |
| Precision | BF16 |
| Sequence length | 8192 |
| Micro/global batch | MBS=1, GBS=1920 |
| Gradient accumulation | 15 microbatches per DP replica |
| Token target | 1,677,721,600 tokens |
| Effective iterations | `floor(tokens / (GBS x sequence_length))` = 106 |
| Optimizer schedule | peak LR `1e-5`, WSD, 10 warmup iterations |
| Checkpoint initialization | local Hugging Face weights (`CPT_INIT_FROM_HF=1`) |
| Checkpoint saving | effectively disabled for this benchmark (`999999`) |
| Evaluation | effectively disabled for this benchmark (`999999`) |
| Weights & Biases | offline run data; API key is still required by launcher preflight |
| Profiling | disabled by default |

`router_force_load_balancing: true` is a **benchmark-only** setting. It replaces
learned router behavior with balanced routing. Set it to `false` before using
this recipe for a real continued-pretraining or convergence workshop.

## 1. Prerequisites

Run from a login node that has:

- Slurm commands (`sbatch`, `srun`, `squeue`, and `scontrol`).
- The Slurm Pyxis plugin and Enroot runtime.
- Access to 16 nodes with 8 NVIDIA GPUs per node and working NCCL/InfiniBand.
- Permission to pull `nvcr.io/nvidia/nemo:26.06` from NGC, or a warmed Enroot
  container cache.
- Shared access to `/mnt/pfs1`, `/mnt/sfs-raw`, and `/mnt/sfs/llm-data-01` from
  every compute node.

This run is tailored to the local `slinky` cluster. For a different cluster,
update the `#SBATCH` resource directives, node selection, mount paths, network
interface (`eth0`), InfiniBand HCA list, and all absolute paths in the YAML and
launcher.

## 2. Prepare the workshop environment

```bash
cd /mnt/pfs1/avinash/pre-training/ajay/GPU16_Run

export PHASE2_ROOT=/mnt/pfs1/avinash/pre-training/Phase2
export CONFIG="$PWD/phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.yaml"
export SCRIPT="$PWD/phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.py"

test -r "$PHASE2_ROOT/scripts/lib/phase2_paths.sh"
test -r "$PHASE2_ROOT/scripts/lib/phase2_run_config.py"
test -r "$SCRIPT"
test -r "$CONFIG"
test -f "$PHASE2_ROOT/models/warmed_focus_init_mn/tokenizer.json"
test -d "$PHASE2_ROOT/data/mixv2_equal_cmx_rom74k"
test -d /mnt/sfs/llm-data-01/cpt-ckpt/step0_embedding_warmup_mn

bash -n phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.sh
```

All `test` commands should return silently with exit status 0. The Step-0
checkpoint is not read while `CPT_INIT_FROM_HF=1`, but keeping the path valid
makes switching to checkpoint initialization less error-prone.

Create a readable environment file for W&B. Do not place credentials in the
launcher or YAML:

```bash
export PHASE2_ENV_FILE="$PHASE2_ROOT/.env"
test -r "$PHASE2_ENV_FILE"
grep -q '^WANDB_API_KEY=' "$PHASE2_ENV_FILE"
```

The file should contain `WANDB_API_KEY=<token>`. The key is required because
the current all-node preflight checks for it even when `WANDB_MODE=offline`.
The run writes local W&B data under `$PHASE2_ROOT/cpt-ckpt/wandb`.

## 3. Check nodes and container

The YAML records `slinky-[16-31]` as the workshop placement, but its `slurm:`
section does not allocate nodes. Confirm that all 16 nodes are healthy:

```bash
sinfo -N -n 'slinky-[16-31]' -o '%N %t %G'
scontrol show hostnames 'slinky-[16-31]'
```

Confirm that Pyxis can start the pinned container and import the required
packages before reserving the full workshop allocation:

```bash
srun --partition=all --nodes=1 --ntasks=1 --gpus=1 --time=00:10:00 \
  --container-image='nvcr.io#nvidia/nemo:26.06' \
  --container-mounts=/mnt/pfs1:/mnt/pfs1,/mnt/sfs-raw:/mnt/sfs-raw,/mnt/sfs/llm-data-01:/mnt/sfs/llm-data-01,/tmp:/root/.deepep \
  bash -lc "python3 -c 'import torch; from megatron.bridge import AutoBridge; print(torch.__version__, torch.cuda.device_count())'"
```

Expected result: the command prints a PyTorch version and at least one visible
CUDA device without an import, mount, or container error.

Run the local NCCL diagnostic workflow before a live workshop when nodes or
fabric health have changed; see `../NCCL/README.md` and `../NCCL/RUNBOOK.md`.

## 4. Review workshop-safe settings

Before submission, inspect these YAML keys:

```bash
sed -n '1,205p' "$CONFIG"
```

At minimum, verify:

- `slurm.nodelist` matches the placement passed to `sbatch`.
- `parallel.replica_mp` is `[1, 8, 1]` and world size is 128.
- `training.global_batch_size` is the value you intend to teach.
- `run.tokens` matches the intended iteration count.
- `paths.*`, `env.RESULTS_DIR`, and `env.FINAL_SAVE_DIR` are writable and do
  not point at another participant's output.
- `moe.router_force_load_balancing` is `true` only for a throughput benchmark.
- `wandb.entity`, `wandb.project`, `wandb.group`, and `wandb.mode` are correct.
- profiling is `false` for the baseline run.

For exactly 100 iterations with the current GBS1920 and sequence length 8192,
set `run.tokens: 1572864000`. To make the filename's GBS2048 label accurate,
set `training.global_batch_size` and `env.GBS` to 2048; the existing token
target then produces exactly 100 iterations. Keep both YAML values aligned.

## 5. Submit the 16-node workshop run

Use explicit Slurm overrides so the stale 20-node directive cannot take
effect. The output paths include the job ID.

```bash
cd /mnt/pfs1/avinash/pre-training/ajay/GPU16_Run

JOB_ID=$(sbatch --parsable \
  --nodes=16 \
  --job-name=mb-workshop-16n \
  --nodelist='slinky-[16-31]' \
  phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.sh)

echo "Submitted job: $JOB_ID"
```

If one or more nodes fail the latest NCCL or GPU health check, replace the
placement with 16 validated nodes. Do not reduce the node count without also
recalculating world size, DP, GBS divisibility, and gradient accumulation.

### Optional submission-time overrides

Slurm exports the submission environment into the job. These examples avoid
editing shared workshop files:

```bash
# Isolate outputs for one participant or workshop session.
export RESULTS_DIR="$PHASE2_ROOT/cpt-ckpt/workshop_${USER}_$(date +%Y%m%d)"
export FINAL_SAVE_DIR="${RESULTS_DIR}_hf"
export WANDB_GROUP="mb-workshop-$(date +%Y%m%d)"

# Print the resolved model parameters and exit after initialization.
export PRINT_PARAMS=1

# Enable a separate PyTorch profiler run only after the baseline.
export TORCH_PROFILE=1
export PROFILE_RANKS='0,7,8,15,64,127'
export PROFILE_STEP_START=40
export PROFILE_STEP_END=50
```

Unset diagnostic overrides before the full run:

```bash
unset PRINT_PARAMS TORCH_PROFILE PROFILE_RANKS PROFILE_STEP_START PROFILE_STEP_END
```

For this launcher, submission-time environment variables win over the
corresponding YAML values. The launcher fills unset variables from YAML and
passes the resolved training values as explicit Python arguments. Python
defaults apply only when neither source supplies a value. Slurm allocation
fields are separate: `#SBATCH` or `sbatch` command-line options control the
actual allocation.

## 6. Monitor and validate

```bash
squeue -j "$JOB_ID" -o '%.18i %.12P %.30j %.8T %.10M %.6D %R'

OUT="$PHASE2_ROOT/logs/mb-workshop-16n-${JOB_ID}.out"
ERR="$PHASE2_ROOT/logs/mb-workshop-16n-${JOB_ID}.err"
tail -f "$OUT"
```

The log should show:

1. One successful container/W&B/GPU preflight line per node.
2. `world_size=128`, `TP=1`, `EP=8`, `PP=1`, and `DP=16`.
3. `MBS / GBS / seq = 1 / 1920 / 8192` for the checked-in config.
4. Training loss, throughput, and iteration-time records after initialization.
5. Exit code 0 and a resolved config at `$RESULTS_DIR/run_config.yaml`.

Useful checks:

```bash
grep -E 'world_size|MBS / GBS / seq|iteration|throughput|exit=' "$OUT" | tail -40
grep -Ei 'error|traceback|nccl warn|timeout|oom|nan' "$OUT" "$ERR" | tail -80
sacct -j "$JOB_ID" --format=JobID,JobName%30,State,Elapsed,ExitCode,AllocNodes
```

Cancel a run cleanly with `scancel "$JOB_ID"`.

## Troubleshooting

| Symptom | Check or action |
| --- | --- |
| Job allocates 20 nodes | Submit with `--nodes=16`; the checked-in `#SBATCH --nodes=20` is stale. |
| Requested nodes are ignored | Pass `--nodelist` to `sbatch`; the YAML `slurm.nodelist` is descriptive. |
| Preflight says `WANDB_API_KEY` is missing | Export the key or set `PHASE2_ENV_FILE` to a readable file available on compute nodes. |
| `tokenizer.json` is missing | Correct `paths.phase1_modified` or `PHASE1_MODIFIED`; this is required for local HF initialization. |
| `ModuleNotFoundError: megatron.bridge` | Confirm the `26.06` image and `/opt/Megatron-Bridge/src` inside it; rerun the one-GPU container smoke test. |
| Enroot/Pyxis mount failure | Verify every host path in `CONTAINER_MOUNTS` exists on every node. |
| HybridEP tries to write under `/root/.deepep` | Keep `/tmp:/root/.deepep` in the container mounts. |
| Config lock timeout during model load | Keep `MEGATRON_CONFIG_LOCK_DIR` on node-local `/tmp`, not shared NFS. |
| NCCL timeout or HCA error | Re-run the NCCL diagnostics and verify `NCCL_IB_HCA_OVERRIDE`; HCA enumeration may differ across nodes. |
| CUDA out of memory | Keep MBS=1, confirm BF16, and inspect recompute settings before reducing sequence length. |
| Run reports 106 rather than 100 iterations | GBS is 1920 while the token target was calculated for GBS2048; use one of the fixes in section 4. |

## Workshop flow

For a reliable session, the facilitator should complete the node-health and
container smoke tests before attendees arrive. During the workshop, first walk
through the YAML and parallelism math, then run `PRINT_PARAMS=1`, launch one
short baseline, inspect `$RESULTS_DIR/run_config.yaml`, and only then enable a
profiler or submit the full benchmark. This keeps infrastructure debugging out
of the part of the session intended to teach Megatron Bridge.

## References

- [Megatron Bridge repository](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron Bridge documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [NVIDIA checkpointing guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html)
