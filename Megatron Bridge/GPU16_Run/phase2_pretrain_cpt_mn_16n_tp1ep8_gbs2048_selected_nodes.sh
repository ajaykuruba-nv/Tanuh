#!/bin/bash
#SBATCH --job-name=phase2-cpt-16n-gbs2048
#SBATCH --partition=all
#SBATCH --nodes=16
#SBATCH --nodelist=slinky-[16-31]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=/mnt/pfs1/avinash/pre-training/Phase2/logs/%x-%j.out
#SBATCH --error=/mnt/pfs1/avinash/pre-training/Phase2/logs/%x-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2a — 16n/128-GPU TP1/EP8 GBS2048/MBS1 selected-node benchmark.
#
# Dedicated launcher for:
#   phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.yaml
# Entry point:
#   phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.py
#
# YAML-first: training/logging/parallel/moe/cuda_graph/distributed values come
# from the config. Shell env overrides still win when set.
#
# Submit from this directory:
#   sbatch phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/nemo:26.06}"
# HybridEP in the NeMo 26.06 image JIT-compiles kernels under
# /root/.deepep.  The container root filesystem is read-only under Pyxis, so
# back that directory with each node's writable local /tmp filesystem.
CONTAINER_MOUNTS="/mnt/sfs-raw:/mnt/sfs-raw,/mnt/pfs1:/mnt/pfs1,/mnt/sfs/llm-data-01:/mnt/sfs/llm-data-01,/tmp:/root/.deepep"
CONTAINER_NAME="nemo-26.06"

_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -f "${_SUBMIT_DIR}/scripts/lib/phase2_paths.sh" ]]; then
    export PHASE2_ROOT="${_SUBMIT_DIR}"
elif [[ -f "${_SUBMIT_DIR}/lib/phase2_paths.sh" ]]; then
    export PHASE2_ROOT="$(cd "${_SUBMIT_DIR}/.." && pwd)"
else
    export PHASE2_ROOT="/mnt/pfs1/avinash/pre-training/Phase2"
fi
# shellcheck disable=SC1091
source "${PHASE2_ROOT}/scripts/lib/phase2_paths.sh"

# Under sbatch, BASH_SOURCE[0] is Slurm's spooled `slurm_script`, not this
# source directory. SLURM_SUBMIT_DIR remains the directory used for `sbatch`,
# so prefer the colocated ajay Python/YAML from there.
_LOCAL_SCRIPT="${_SUBMIT_DIR}/phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.py"
_LOCAL_CONFIG="${_SUBMIT_DIR}/phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.yaml"
SCRIPT="${SCRIPT:-${_LOCAL_SCRIPT}}"
CONFIG="${CONFIG:-${_LOCAL_CONFIG}}"
if [[ "${CONFIG}" != /* && -f "${PHASE2_ROOT}/${CONFIG}" ]]; then
    CONFIG="${PHASE2_ROOT}/${CONFIG}"
fi
SRUN="${SRUN:-/usr/bin/srun}"

if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: SCRIPT missing: ${SCRIPT}" >&2
    exit 1
fi
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: CONFIG missing: ${CONFIG}" >&2
    exit 1
fi

export PYTHONPATH="${PHASE2_ROOT}/scripts/lib${PYTHONPATH:+:${PYTHONPATH}}"
phase2_apply_config_env "${CONFIG}"

phase2_deploy_stamp "${SCRIPT}" "${PHASE2_ROOT}"

# LR from YAML optimizer.lr when unset.
if [ -z "${LR:-}" ]; then
    LR="$(PYTHONPATH="${PHASE2_ROOT}/scripts/lib${PYTHONPATH:+:${PYTHONPATH}}" python3 -c "
from phase2_run_config import _deep_get, _load_config_dict
from pathlib import Path
cfg = _load_config_dict(Path('${CONFIG}'))
lr = _deep_get(cfg, 'optimizer', 'lr')
if lr is None:
    raise SystemExit('optimizer.lr missing in CONFIG')
print(lr)
")"
fi
export LR

# Defaults filled from YAML via phase2_apply_config_env; these are fallbacks only.
TOKENS="${TOKENS:-1677721600}"
MIX_ID="${MIX_ID:-mixv2_equal_cmx_rom74k}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
LR_SCHED="${LR_SCHED:-wsd}"
WSD_STABLE_FRAC="${WSD_STABLE_FRAC:-0.80}"
MIN_LR_FRAC="${MIN_LR_FRAC:-0.10}"
GBS="${GBS:-2048}"
MBS="${MBS:-1}"
SEQ="${SEQ:-8192}"
SAVE_INTERVAL="${SAVE_INTERVAL:-999999}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"
PRINT_PARAMS="${PRINT_PARAMS:-0}"
CPT_INIT_FROM_HF="${CPT_INIT_FROM_HF:-1}"
REPLICA_MP="${REPLICA_MP:-1,8,1}"
NCCL_TUNED="${NCCL_TUNED:-1}"

ARM_TAG=$(python3 -c "
lr = float('${LR}')
lr_str = f'{lr:.0e}'.replace('+', '').replace('-0', '-')
print(f'lr_{lr_str}_${MIX_ID}')
")

STEP0_MEGATRON_DIR="${STEP0_MEGATRON_DIR:-/mnt/sfs/llm-data-01/cpt-ckpt/step0_embedding_warmup_mn}"
RESULTS_DIR="${RESULTS_DIR:-${PHASE2_ROOT}/cpt-ckpt/16n_tp1ep8_gbs2048_mbs1_selected_nodes_mn}"
FINAL_SAVE_DIR="${FINAL_SAVE_DIR:-${PHASE2_ROOT}/cpt-ckpt/16n_tp1ep8_gbs2048_mbs1_selected_nodes_hf}"
CPT_SKIP_HF_CONVERT="${CPT_SKIP_HF_CONVERT:-1}"
PHASE1_MODIFIED="${PHASE1_MODIFIED:-${PHASE2_ROOT}/models/warmed_focus_init_mn}"
export STEP0_MEGATRON_DIR RESULTS_DIR FINAL_SAVE_DIR CPT_SKIP_HF_CONVERT PHASE1_MODIFIED
export CPT_INIT_FROM_HF REPLICA_MP NCCL_TUNED

export HOME=/tmp
export XDG_CACHE_HOME=/mnt/sfs-raw/developers/prasanjith/.cache
export HF_HOME=/mnt/sfs-raw/huggingface_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-/tmp/megatron_config_lock_${USER:-default}}"
export TORCH_HOME=/mnt/sfs-raw/torch_cache
export FLASHINFER_WORKSPACE_BASE=/mnt/sfs-raw/developers/prasanjith
export FLASHINFER_WORKSPACE_DIR=${XDG_CACHE_HOME}/flashinfer
export TRITON_CACHE_DIR=${XDG_CACHE_HOME}/triton
export NVTE_CACHE_DIR=${XDG_CACHE_HOME}/nvte
export CUDA_CACHE_PATH=${XDG_CACHE_HOME}/cuda
export TORCH_EXTENSIONS_DIR=${XDG_CACHE_HOME}/torch_extensions
export GIT_PYTHON_REFRESH=quiet
# Never embed credentials in a launcher. Preserve a token supplied by the
# submit environment, if one is needed for a non-local Hugging Face model.
export HF_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

export WANDB_DIR="${WANDB_DIR:-${PHASE2_ROOT}/cpt-ckpt/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

mkdir -p ${XDG_CACHE_HOME} ${HF_HOME} ${HF_HUB_CACHE} ${MEGATRON_CONFIG_LOCK_DIR} ${TORCH_HOME} \
         ${FLASHINFER_WORKSPACE_DIR} \
         ${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer \
         ${TRITON_CACHE_DIR} \
         ${NVTE_CACHE_DIR} ${CUDA_CACHE_PATH} ${TORCH_EXTENSIONS_DIR} \
         ${WANDB_DIR}
chmod 777 ${WANDB_DIR} 2>/dev/null || true

for D in "${PHASE2_ROOT}/logs" "${RESULTS_DIR}" "${FINAL_SAVE_DIR}"; do
    mkdir -p "$D"
    chmod 777 "$D" 2>/dev/null || true
done

export PYTHONPATH="${PHASE2_ROOT}/scripts/lib:${PHASE2_ROOT}/scripts/convert:${PHASE2_ROOT}/scripts/pretrain:${PHASE2_ROOT}/scripts/data:${PHASE2_ROOT}/scripts/multinode:${PHASE2_ROOT}/scripts:/opt/Megatron-Bridge/src:/opt/megatron-lm${PYTHONPATH:+:${PYTHONPATH}}"

phase2_apply_config_env "${CONFIG}"

export PHASE2_ENV_FILE="${PHASE2_ENV_FILE:-${PHASE2_ROOT}/.env}"
phase2_setup_wandb || true

if [ "${NVIDIA_MOE_STACK:-0}" = "1" ]; then
    export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
    export NCCL_PROTO="${NCCL_PROTO:-simple}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
fi

if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
    if [ "${NCCL_IB_DISABLE:-0}" = "1" ] || [ "${NCCL_NET:-}" = "Socket" ]; then
        export NCCL_IB_DISABLE=1
        export NCCL_NET=Socket
        echo "[nccl] Multi-node BISECT: Socket (NCCL_IB_DISABLE=1, no IB)"
    else
        export NCCL_NET=IB
        # Do not force a cluster-wide HCA list: device availability differs by
        # node and known selected rails are down on slinky-19/21. Let NCCL pick
        # active interfaces unless an explicit override is supplied.
        if [ -n "${NCCL_IB_HCA_OVERRIDE:-}" ]; then
            export NCCL_IB_HCA="${NCCL_IB_HCA_OVERRIDE}"
        else
            unset NCCL_IB_HCA
        fi
        export NCCL_IB_GID_INDEX=3
        echo "[nccl] Multi-node config: IB with HCA=${NCCL_IB_HCA:-auto}"
    fi
else
    export NCCL_NET=Socket
    echo "[nccl] Single-node config: Socket"
fi
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=7200000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_NCCL_DUMP_ON_TIMEOUT=0

if [ "${NCCL_TUNED}" = "1" ] && [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
    export NCCL_ALGO="${NCCL_ALGO:-Ring,Tree}"
    export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
    export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
    echo "[nccl] Tuned: NCCL_ALGO=${NCCL_ALGO} MIN_NCHANNELS=${NCCL_MIN_NCHANNELS} IB_QPS=${NCCL_IB_QPS_PER_CONNECTION}"
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500

echo "======================================================================"
echo " Phase 2a — best_prof_fixed (MULTI-NODE)"
echo " Job ID              : ${SLURM_JOB_ID:-<interactive>}"
echo " Nodes               : ${SLURM_JOB_NUM_NODES:-1}"
echo " Node list           : ${SLURM_JOB_NODELIST:-$(hostname)}"
echo " Master addr:port    : ${MASTER_ADDR}:${MASTER_PORT}"
echo " Script              : ${SCRIPT}"
echo " Config file         : ${CONFIG}"
echo " Arm tag             : ${ARM_TAG}"
echo " LR (peak)           : ${LR}"
echo " Decay style         : ${LR_SCHED}"
echo " WSD stable frac     : ${WSD_STABLE_FRAC}"
echo " Warmup iters        : ${WARMUP_ITERS}"
echo " Target tokens       : ${TOKENS}"
echo " Data mix            : ${MIX_ID}"
echo " MBS / GBS / seq     : ${MBS} / ${GBS} / ${SEQ}"
echo " Save interval       : ${SAVE_INTERVAL}"
echo " Eval interval       : ${EVAL_INTERVAL}"
echo " Step 0 resume dir   : ${STEP0_MEGATRON_DIR}"
echo " Megatron save dir   : ${RESULTS_DIR}"
echo " HF export dir       : ${FINAL_SAVE_DIR}"
echo " Skip HF convert     : ${CPT_SKIP_HF_CONVERT}"
echo " HF reference        : ${PHASE1_MODIFIED}"
echo " CPT_INIT_FROM_HF    : ${CPT_INIT_FROM_HF}"
echo " REPLICA_MP          : ${REPLICA_MP}"
echo " NCCL_TUNED          : ${NCCL_TUNED}"
echo " TORCH_PROFILE       : ${TORCH_PROFILE:-0}"
echo " PROFILE_RANKS       : ${PROFILE_RANKS:-0}"
echo " PROFILE_STEP        : ${PROFILE_STEP_START:-10}-${PROFILE_STEP_END:-20}"
echo " PROFILE_OUTPUT_DIR  : ${PROFILE_OUTPUT_DIR:-${RESULTS_DIR}/profile_traces}"
echo " WandB project/group : ${WANDB_PROJECT:-<unset>} / ${WANDB_GROUP:-<unset>}"
echo " WandB mode/dir      : ${WANDB_MODE:-offline} / ${WANDB_DIR:-<unset>}"
echo "======================================================================"

if [ "${CPT_INIT_FROM_HF}" != "1" ]; then
    if [ ! -d "${STEP0_MEGATRON_DIR}" ] || [ -z "$(ls -A "${STEP0_MEGATRON_DIR}" 2>/dev/null)" ]; then
        echo "ERROR: STEP0_MEGATRON_DIR is missing or empty: ${STEP0_MEGATRON_DIR}"
        exit 1
    fi
fi
if [ ! -f "${PHASE1_MODIFIED}/tokenizer.json" ]; then
    echo "ERROR: no tokenizer.json under PHASE1_MODIFIED = ${PHASE1_MODIFIED}"
    exit 1
fi

# ── Container + WandB preflight — same pattern as phase2_pretrain_cpt_mn.sh ─
# Catches pyxis/enroot failures before torchrun allocates all 128 ranks.
echo "[preflight] checking container + WANDB_API_KEY on all nodes..."
"${SRUN}" --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-name="${CONTAINER_NAME}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --export=ALL,PHASE2_ROOT,PHASE2_ENV_FILE,WANDB_API_KEY,WANDB_PROJECT,WANDB_ENTITY,WANDB_GROUP,WANDB_NAME,WANDB_MODE,WANDB_DIR,WANDB_START_METHOD \
    bash -c "\
        if [ -z \"\${WANDB_API_KEY:-}\" ] && [ -r \"\${PHASE2_ENV_FILE:-${PHASE2_ROOT}/.env}\" ]; then \
            set -a; source \"\${PHASE2_ENV_FILE:-${PHASE2_ROOT}/.env}\"; set +a; \
        fi; \
        echo \"[preflight] node=\${SLURMD_NODENAME:-\$(hostname)} wandb_key=\$([ -n \"\${WANDB_API_KEY:-}\" ] && echo set || echo MISSING) project=\${WANDB_PROJECT:-<unset>}\"; \
        nvidia-smi -L | head -1; \
        [ -n \"\${WANDB_API_KEY:-}\" ]"
PREFLIGHT_RC=$?
if [ "${PREFLIGHT_RC}" -ne 0 ]; then
    echo "ERROR: container or WANDB_API_KEY preflight failed on one or more nodes." >&2
    echo "ERROR: If pyxis reports nvidia-fabricmanager/socket, exclude that node via SLURM_EXCLUDE." >&2
    exit 1
fi

EXTRA_ARGS=()
if [ "${PRINT_PARAMS}" = "1" ]; then
    EXTRA_ARGS+=(--print-params-then-exit)
fi
if [ "${LOG_STRAGGLER:-0}" = "1" ]; then
    EXTRA_ARGS+=(--log-straggler)
fi
if [ -n "${CHECK_FOR_NAN_IN_GRAD:-}" ]; then
    if [ "${CHECK_FOR_NAN_IN_GRAD}" = "0" ] || [ "${CHECK_FOR_NAN_IN_GRAD}" = "false" ]; then
        EXTRA_ARGS+=(--no-check-for-nan-in-grad)
    else
        EXTRA_ARGS+=(--check-for-nan-in-grad)
    fi
fi

NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-${RESULTS_DIR}/nsys_traces}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,nccl}"
NSYS_DURATION="${NSYS_DURATION:-0}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
export NSYS_PROFILE NSYS_OUTPUT_DIR NSYS_TRACE NSYS_DURATION NSYS_SAMPLE
export PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-${RESULTS_DIR}/profile_traces}"
export NSIGHT_PYTHON="${NSIGHT_PYTHON:-0}"
export NSIGHT_ONLY="${NSIGHT_ONLY:-0}"
export NSIGHT_OUTPUT_DIR="${NSIGHT_OUTPUT_DIR:-${RESULTS_DIR}/nsight_kernels}"

if [ "${NSYS_PROFILE}" = "1" ]; then
    mkdir -p "${NSYS_OUTPUT_DIR}"
    chmod 777 "${NSYS_OUTPUT_DIR}" 2>/dev/null || true
fi

_build_torchrun() {
    local _nsys_prefix=""
    if [ "${NSYS_PROFILE}" = "1" ]; then
        _nsys_prefix="nsys profile --force-overwrite=true --sample=${NSYS_SAMPLE} --trace=${NSYS_TRACE}"
        if [ "${NSYS_DURATION}" != "0" ]; then
            _nsys_prefix="${_nsys_prefix} --duration=${NSYS_DURATION}"
        fi
        _nsys_prefix="${_nsys_prefix} -o ${NSYS_OUTPUT_DIR}/node\${SLURM_NODEID}"
    fi
    echo "${_nsys_prefix} torchrun \
            --nnodes=\${SLURM_JOB_NUM_NODES} \
            --nproc_per_node=8 \
            --node_rank=\${SLURM_NODEID} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${SCRIPT} \
                --config ${CONFIG} \
                --lr ${LR} \
                --tokens ${TOKENS} \
                --mix-id ${MIX_ID} \
                --warmup-iters ${WARMUP_ITERS} \
                --lr-decay-style ${LR_SCHED} \
                --wsd-stable-frac ${WSD_STABLE_FRAC} \
                --min-lr-frac ${MIN_LR_FRAC} \
                --global-batch-size ${GBS} \
                --micro-batch-size ${MBS} \
                --seq-length ${SEQ} \
                --save-interval ${SAVE_INTERVAL} \
                --eval-interval ${EVAL_INTERVAL} \
                ${EXTRA_ARGS[@]:-}"
}

"${SRUN}" --kill-on-bad-exit=1 --ntasks-per-node=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-name="${CONTAINER_NAME}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --export=ALL,PHASE2_ROOT,PHASE2_ENV_FILE,WANDB_API_KEY,WANDB_PROJECT,WANDB_ENTITY,WANDB_GROUP,WANDB_NAME,WANDB_MODE,WANDB_DIR,WANDB_START_METHOD \
    bash -c "\
        if [ -z \"\${WANDB_API_KEY:-}\" ] && [ -r \"\${PHASE2_ENV_FILE:-${PHASE2_ROOT}/.env}\" ]; then \
            set -a; source \"\${PHASE2_ENV_FILE:-${PHASE2_ROOT}/.env}\"; set +a; \
        fi; \
        export WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_GROUP WANDB_NAME WANDB_MODE WANDB_DIR WANDB_START_METHOD; \
        if [ -n \"\${WANDB_API_KEY:-}\" ]; then \
            . \"\${PHASE2_ROOT}/scripts/lib/phase2_wandb_bootstrap.sh\"; \
        fi; \
        echo \"[rank_env] node=\${SLURMD_NODENAME} SLURM_NODEID=\${SLURM_NODEID} wandb_key=\$([ -n \"\${WANDB_API_KEY:-}\" ] && echo set || echo MISSING)\" && \
        echo \"[rank_env] python=\$(python3 --version 2>&1) torch=\$(python3 -c 'import torch; print(torch.__version__)' 2>&1)\" && \
        nvidia-smi -L | head -1 && \
        $(_build_torchrun)"

RC=$?

echo ""
echo "======================================================================"
echo " CPT best_prof_fixed (MN) exit=${RC}"
echo " Megatron ckpt : ${RESULTS_DIR}"
echo " Run config    : ${RESULTS_DIR}/run_config.yaml"
echo " HF checkpoint : ${FINAL_SAVE_DIR}"
echo "======================================================================"

exit ${RC}
