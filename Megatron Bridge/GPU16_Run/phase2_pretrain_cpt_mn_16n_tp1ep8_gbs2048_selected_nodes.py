#!/usr/bin/env python3
"""
Phase 2a — multi-node pretrain for the best_prof_fixed YAML recipe.

Companion to `phase2_pretrain_cpt_mn.py` with YAML-field wiring fixes so
`phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.yaml`
is fully honored (not only env:/CLI patches):

  - training.dataset_num_workers → config.dataset.num_workers
  - logging.log_interval → config.logger.log_interval (LOG_INTERVAL still wins)
  - logging.log_throughput / moe_per_layer_logging / log_timers_level from run
  - parallel.context_parallel_size → config.model.context_parallel_size
  - distributed.* flags from YAML (stage0 defaults preserved when unset)

Default topology for this recipe: REPLICA_MP=(1, 8, 1) at 16 nodes / 128 GPUs,
with GBS=2048 and MBS=1. Launch via the colocated 8-node shell script.
"""

import os as _os

# Pyxis workers see HOME=/root (read-only). Mirror Step 0 early-init (GOTCHAS
# § 18g) so Bridge / flashinfer / HF never touch ~/.cache during import.
_os.environ["HF_HOME"] = "/mnt/sfs-raw/huggingface_cache"
_os.environ["HF_HUB_CACHE"] = "/mnt/sfs-raw/huggingface_cache/hub"

# MEGATRON_CONFIG_LOCK_DIR — Bridge's safe_config_loader.py flocks a lock
# file whose name is hashed from the config path. Every rank hashes to the
# SAME lock filename, so putting the lock dir on shared NFS causes a
# 128-way flock stampede that times out Bridge's 4-attempt retry — jobs die
# at model-load BEFORE training starts. Diagnosed 2026-07-09 via 55928/55929.
# See GOTCHAS §32.
#
# Fix: default to node-local tmpfs (/tmp). Coordination stays within a node
# (8 ranks/node → fits the retry budget); no cross-node NFS lock at all.
# Config load is read-only from a static source → node-local is safe.
#
# Override via env if you need per-job NFS lock (e.g. cross-job coordination):
#   export MEGATRON_CONFIG_LOCK_DIR=${RESULTS_DIR}/config_lock
_lock_dir_default = f"/tmp/megatron_config_lock_{_os.environ.get('USER', 'default')}"
_os.environ.setdefault("MEGATRON_CONFIG_LOCK_DIR", _lock_dir_default)

_os.environ.setdefault("TORCH_HOME", "/mnt/sfs-raw/torch_cache")
_os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "0")
_os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")
_os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
for _early in (
    _os.environ["HF_HOME"],
    _os.environ["HF_HUB_CACHE"],
    _os.environ["MEGATRON_CONFIG_LOCK_DIR"],
    _os.environ["TORCH_HOME"],
):
    try:
        _os.makedirs(_early, exist_ok=True)
    except OSError:
        pass

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import torch._dynamo
torch._dynamo.config.disable = True

# ─────────────────────────────────────────────────────────────────────────────
# Cache paths must be set BEFORE any downstream imports (Megatron / TE / etc.)
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("XDG_CACHE_HOME", "/mnt/sfs-raw/developers/prasanjith/.cache")
# flashinfer 0.6.2 jit/env.py: FLASHINFER_WORKSPACE_BASE + /.cache/flashinfer/...
os.environ["FLASHINFER_WORKSPACE_BASE"] = "/mnt/sfs-raw/developers/prasanjith"
os.environ.setdefault("FLASHINFER_WORKSPACE_DIR", f"{os.environ['XDG_CACHE_HOME']}/flashinfer")
os.environ.setdefault("TRITON_CACHE_DIR", f"{os.environ['XDG_CACHE_HOME']}/triton")
os.environ.setdefault("NVTE_CACHE_DIR", f"{os.environ['XDG_CACHE_HOME']}/nvte")
os.environ.setdefault("CUDA_CACHE_PATH", f"{os.environ['XDG_CACHE_HOME']}/cuda")
for _p in (
    os.environ["FLASHINFER_WORKSPACE_DIR"],
    f"{os.environ['FLASHINFER_WORKSPACE_BASE']}/.cache/flashinfer",
    os.environ["TRITON_CACHE_DIR"],
    os.environ["NVTE_CACHE_DIR"],
    os.environ["CUDA_CACHE_PATH"],
):
    os.makedirs(_p, exist_ok=True)

import torch.distributed as dist
# GOTCHAS § 19a — Bridge/Megatron passes explicit timeout=default_pg_timeout
# (600s / 10min) to init_process_group. setdefault is a silent no-op when
# key is already present, so the "2h override" gets ignored — bit us on
# 21844 + 21846 (32-rank end-save timeouts). Use direct ASSIGNMENT to force
# override. Validated by 21898 (Step 0 v3) + 21900 (Phase 1 mid-run saves).
_orig_init_pg = dist.init_process_group
def _patched_init_pg(*args, **kwargs):
    kwargs["timeout"] = datetime.timedelta(hours=2)
    return _orig_init_pg(*args, **kwargs)
dist.init_process_group = _patched_init_pg

import torch
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.recipes.nemotronh import nemotron_3_nano
from megatron.bridge.training.config import StragglerDetectionConfig
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.bridge.training.pretrain import pretrain

# Shared helpers live under scripts/lib/, scripts/convert/, etc.
_LIB = str(Path(__file__).resolve().parent / "lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
from phase2_bootstrap import bootstrap

bootstrap()

from phase2_data_path import (
    apply_replica_mp,
    get_data_path,
    replica_size,
    resolve_topology,
    stage_tokenizer_for_megatron,
    wire_dataset_for_training,
)
from phase2_run_config import (
    PHASE2_ROOT,
    apply_run_config_env,
    resolve_pretrain_config,
    save_resolved_config,
)

DEFAULT_BEST_PROF_FIXED_CONFIG = Path(__file__).resolve().with_name(
    "phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.yaml"
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--config",
        default=str(DEFAULT_BEST_PROF_FIXED_CONFIG),
        help="YAML run config (default: colocated "
             "phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.yaml). "
             "Hydra-style dotlist overrides may follow as extra args.",
    )
    p.add_argument("--lr", type=float, default=None,
                   help="Peak LR for this arm. Overrides optimizer.lr in the YAML.")
    p.add_argument("--tokens", type=float, default=None)
    p.add_argument("--mix-id", default=None)
    p.add_argument("--warmup-iters", type=int, default=None)
    p.add_argument("--lr-decay-style", choices=("wsd", "cosine"), default=None)
    p.add_argument("--wsd-stable-frac", type=float, default=None)
    p.add_argument("--min-lr-frac", type=float, default=None)
    p.add_argument("--global-batch-size", type=int, default=None)
    p.add_argument("--micro-batch-size",  type=int, default=None)
    p.add_argument("--seq-length",        type=int, default=None)
    p.add_argument("--save-interval",     type=int, default=None)
    p.add_argument("--eval-interval",     type=int, default=None)
    p.add_argument("--print-params-then-exit", action="store_true")
    p.add_argument("--log-straggler", action="store_true",
                   help="Enable Megatron Bridge per-GPU straggler logging "
                        "(config.straggler.log_straggler=True).")
    p.add_argument(
        "--check-for-nan-in-grad",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle DDP per-bucket grad NaN/Inf validation "
             "(config.ddp.check_for_nan_in_grad). "
             "Use --no-check-for-nan-in-grad to disable the aten::item "
             "sync path in backward.",
    )
    return p.parse_known_args()


def _arm_tag(lr: float, mix_id: str) -> str:
    lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    return f"lr_{lr_str}_{mix_id}"


# ─────────────────────────────────────────────────────────────────────────────
# WSD scheduler — see GOTCHAS § 2. Peak LR / min LR live on config.optimizer
# (§ 16); shape fields on config.scheduler.
# ─────────────────────────────────────────────────────────────────────────────
def _configure_wsd(config, target_iters: int, warmup_iters: int,
                   peak_lr: float, min_lr: float, stable_frac: float):
    config.optimizer.lr              = peak_lr
    config.optimizer.min_lr          = min_lr
    config.scheduler.lr_warmup_iters = warmup_iters

    stable_end  = int(warmup_iters + (target_iters - warmup_iters) * stable_frac)
    decay_iters = target_iters - stable_end

    config.scheduler.lr_decay_style     = "WSD"          # case-sensitive!
    config.scheduler.lr_decay_iters     = target_iters
    config.scheduler.lr_wsd_decay_iters = decay_iters
    config.scheduler.lr_wsd_decay_style = "minus_sqrt"
    return f"WSD (decay_iters={decay_iters}, stable_iters={stable_end - warmup_iters})"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args, cli_overrides = parse_args()
    run, resolved_cfg = resolve_pretrain_config(args.config, args, cli_overrides)
    apply_run_config_env(run)

    # NSIGHT_PYTHON smoke — rank 0 only, separate process preserves import order.
    if run.nsight_python and run.nsight_only:
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            sys.exit(0)
        import subprocess
        _smoke = Path(__file__).resolve().parent / "analysis" / "phase2_nsight_smoke.py"
        _out_dir = os.environ.get("NSIGHT_OUTPUT_DIR", f"{run.cpt_ckpt_root}/nsight_kernels")
        _out = str(Path(_out_dir) / "matmul.csv")
        _size = os.environ.get("NSIGHT_SIZE", "4096")
        _runs = os.environ.get("NSIGHT_RUNS", "3")
        print(f"[nsight] NSIGHT_ONLY=1 — running smoke probe → {_out}", flush=True)
        subprocess.run(
            [sys.executable, str(_smoke), "--size", _size, "--runs", _runs, "--output", _out],
            check=True,
        )
        sys.exit(0)

    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    replica_mp = resolve_topology(world_size, replica_mp=run.replica_mp)
    tp, ep, pp = replica_mp
    dp = world_size // replica_size(replica_mp)

    target_iters = run.target_iters
    arm_tag      = _arm_tag(run.lr, run.mix_id)

    results_dir  = os.environ.get(
        "RESULTS_DIR", f"{run.cpt_ckpt_root}/{arm_tag}{run.results_suffix}"
    )
    final_hf_dir = os.environ.get(
        "FINAL_SAVE_DIR", f"{run.cpt_ckpt_root}/{arm_tag}_hf"
    )
    tb_dir = run.tensorboard_dir or f"{results_dir}/tb_logs"
    profile_output_dir = run.profile_output_dir or f"{results_dir}/profile_traces"

    rank0 = int(os.environ.get("RANK", "0")) == 0
    if rank0:
        print("=" * 72)
        print(f" Phase 2a — LR sweep arm (MULTI-NODE): {arm_tag}")
        print("=" * 72)
        print(f" Config file         : {run.source_config}")
        print(f" world_size          : {world_size}  →  TP={tp}, EP={ep}, PP={pp}, DP={dp}")
        print(f" REPLICA_MP          : ({tp}, {ep}, {pp})")
        print(f" Init weights        : {'HF (init_from_hf)' if run.init_from_hf else f'Step0 ckpt {run.step0_megatron_dir}'}")
        print(f" Peak LR             : {run.lr}")
        print(f" Decay style         : {run.lr_decay_style}")
        print(f" WSD stable frac     : {run.wsd_stable_frac}")
        print(f" Target tokens       : {run.tokens:.2e}  →  iters ≈ {target_iters}")
        print(f" Tokens/step         : {run.tokens_per_step:,}")
        print(f" LR warmup iters     : {run.warmup_iters}")
        print(f" MBS / GBS / seq     : {run.micro_batch_size} / {run.global_batch_size} / {run.seq_length}")
        print(f" Save interval       : {run.save_interval}")
        print(f" Step 0 resume dir   : {run.step0_megatron_dir}")
        print(f" Save dir (Megatron) : {results_dir}")
        print(f" Save dir (HF)       : {final_hf_dir}")
        print(f" Tokenizer bundle    : {run.phase1_modified}")
        print(f" TORCH_PROFILE       : {run.torch_profile}")
        print(f" NSYS_PROFILE        : {run.nsys_profile}")
        print("=" * 72)

    # ── Preflight ─────────────────────────────────────────────────────────
    if not run.init_from_hf:
        if not os.path.isdir(run.step0_megatron_dir) or not any(os.scandir(run.step0_megatron_dir)):
            print(f"ERROR: step0_megatron_dir missing/empty: {run.step0_megatron_dir}",
                  file=sys.stderr)
            return 1
    if not os.path.isfile(f"{run.phase1_modified}/tokenizer.json"):
        print(f"ERROR: no tokenizer.json under phase1_modified = {run.phase1_modified}",
              file=sys.stderr)
        return 1

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(final_hf_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    save_resolved_config(resolved_cfg, f"{results_dir}/run_config.yaml")

    # ── Build config from the Nemotron-3-Nano recipe ─────────────────────
    config = nemotron_3_nano.nemotron_3_nano_pretrain_config()

    config.model = AutoBridge.from_hf_pretrained(
        run.phase1_modified, trust_remote_code=True,
    ).to_megatron_provider(load_weights=run.init_from_hf)

    apply_replica_mp(config.model, replica_mp)
    # YAML may say sequence_parallel: true for TP>1 arms; never enable at TP=1.
    if run.sequence_parallel is not None:
        config.model.sequence_parallel = (
            run.sequence_parallel and config.model.tensor_model_parallel_size > 1
        )

    config.model.vocab_size = run.vocab_size
    config.model.seq_length = run.seq_length

    # ── MEDIUM-tier kernel enablement (all packages verified in n3nano.sqsh, §22) ─
    # Env-var gated so smoke A/B/C runs can toggle without code changes.
    # Uses defensive hasattr checks — Bridge config attr names vary by version.
    def _try_setattr(obj, attr, value, path_label):
        """Set config.<path>=value if attribute exists; log outcome. rank0-only print."""
        try:
            if not hasattr(obj, attr):
                if rank0:
                    print(f"[kernel] SKIP {path_label} — attr not present in config", flush=True)
                return False
            setattr(obj, attr, value)
            if rank0:
                print(f"[kernel] SET {path_label} = {value!r}", flush=True)
            return True
        except (AttributeError, TypeError) as e:
            if rank0:
                print(f"[kernel] FAIL {path_label}: {type(e).__name__}: {e}", flush=True)
            return False

    def _cfg_get(cfg: dict, *keys, default=None):
        cur: object = cfg
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    # Explicitly match `_perf_precision("bf16")` from NVIDIA's performance
    # recipe rather than relying on the base recipe's current default.
    _mixed_precision = _cfg_get(
        resolved_cfg, "model", "mixed_precision", default="bf16"
    )
    if _mixed_precision == "bf16":
        config.mixed_precision = bf16_mixed()
        if rank0:
            print("[precision] mixed_precision=bf16", flush=True)
    else:
        raise ValueError(
            f"Unsupported model.mixed_precision={_mixed_precision!r}; "
            "this launcher currently supports only 'bf16'."
        )

    if run.grouped_gemm:
        # Fused MoE grouped GEMM — biggest single MoE win (~2×).
        # Package: grouped_gemm 1.1.2 (verified via job 55785).
        # Try multiple attr paths since Bridge nesting varies.
        _try_setattr(config.model, "moe_grouped_gemm", True, "config.model.moe_grouped_gemm")
        if hasattr(config.model, "transformer"):
            _try_setattr(config.model.transformer, "moe_grouped_gemm", True,
                         "config.model.transformer.moe_grouped_gemm")
        if hasattr(config, "optimization"):
            _try_setattr(config.optimization, "moe_grouped_gemm", True,
                         "config.optimization.moe_grouped_gemm")

    fp8_mode = run.use_fp8
    if fp8_mode in ("hybrid", "e4m3"):
        # FP8 GEMMs via Transformer Engine — Hopper native, ~1.8× gain.
        # Package: transformer_engine 2.9.0+70f53666 (verified via job 55785).
        _try_setattr(config.model, "fp8", fp8_mode, "config.model.fp8")
        _try_setattr(config.model, "fp8_amax_history_len", 1024,
                     "config.model.fp8_amax_history_len")
        _try_setattr(config.model, "fp8_amax_compute_algo", "max",
                     "config.model.fp8_amax_compute_algo")

    if run.selective_recompute:
        _try_setattr(config.model, "recompute_granularity", "selective",
                     "config.model.recompute_granularity")
        _recompute_modules = _cfg_get(
            resolved_cfg, "kernels", "recompute_modules", default=["moe", "layernorm"]
        )
        if isinstance(_recompute_modules, str):
            _recompute_modules = [
                module.strip() for module in _recompute_modules.split(",") if module.strip()
            ]
        _try_setattr(config.model, "recompute_modules", list(_recompute_modules),
                     f"config.model.recompute_modules={list(_recompute_modules)!r}")

    if run.use_mcore_models:
        _try_setattr(config.model, "use_mcore_models", True,
                     "config.model.use_mcore_models")

    # ── YAML-driven perf flags (NVIDIA 512-script parity) ───────────────────
    if _cfg_get(resolved_cfg, "model", "use_fused_weighted_squared_relu"):
        _try_setattr(
            config.model,
            "use_fused_weighted_squared_relu",
            True,
            "config.model.use_fused_weighted_squared_relu",
        )

    if _cfg_get(resolved_cfg, "ddp", "pad_buckets_for_high_nccl_busbw"):
        _try_setattr(
            getattr(config, "ddp", None),
            "pad_buckets_for_high_nccl_busbw",
            True,
            "config.ddp.pad_buckets_for_high_nccl_busbw",
        )

    _yaml_manual_gc = _cfg_get(resolved_cfg, "training", "manual_gc")
    _yaml_manual_gc_interval = _cfg_get(
        resolved_cfg, "training", "manual_gc_interval", default=100
    )
    if _yaml_manual_gc:
        for _tgt_path, _tgt in [
            ("config.train", getattr(config, "train", None)),
            ("config.training", getattr(config, "training", None)),
            ("config.runtime", getattr(config, "runtime", None)),
        ]:
            if _tgt is None:
                continue
            _try_setattr(_tgt, "manual_gc", True, f"{_tgt_path}.manual_gc=True")
            _try_setattr(
                _tgt,
                "manual_gc_interval",
                int(_yaml_manual_gc_interval),
                f"{_tgt_path}.manual_gc_interval={int(_yaml_manual_gc_interval)}",
            )
            _try_setattr(_tgt, "manual_gc_eval", True, f"{_tgt_path}.manual_gc_eval=True")

    # ── YAML-driven MoE A2A overlap + delayed wgrad (NVIDIA perf script parity) ─
    _moe_attr_targets = [
        ("config.model", getattr(config, "model", None)),
        ("config.model.moe", getattr(getattr(config, "model", None), "moe", None) if getattr(config, "model", None) else None),
        ("config.model.transformer", getattr(getattr(config, "model", None), "transformer", None) if getattr(config, "model", None) else None),
        ("config.model.transformer.moe",
            getattr(getattr(getattr(config, "model", None), "transformer", None), "moe", None)
            if (getattr(config, "model", None) and getattr(getattr(config, "model", None), "transformer", None))
            else None),
    ]
    _yaml_ep_overlap = _cfg_get(
        resolved_cfg, "comm_overlap", "overlap_moe_expert_parallel_comm"
    )
    if _yaml_ep_overlap is None:
        _yaml_ep_overlap = _cfg_get(
            resolved_cfg, "moe", "overlap_moe_expert_parallel_comm"
        )
    _yaml_delay_wgrad = _cfg_get(resolved_cfg, "comm_overlap", "delay_wgrad_compute")
    if _yaml_delay_wgrad is None:
        _yaml_delay_wgrad = _cfg_get(resolved_cfg, "moe", "delay_wgrad_compute")
    if _yaml_ep_overlap:
        raise RuntimeError(
            "Runtime gate: overlap_moe_expert_parallel_comm=True is blocked in this "
            "checkout (#1810 deadlock risk). Keep it false until the two-event fix is merged."
        )
    _comm = getattr(config, "comm_overlap", None)
    _yaml_tp_overlap = _cfg_get(resolved_cfg, "comm_overlap", "tp_comm_overlap")
    if _yaml_tp_overlap is not None and _comm is not None:
        _try_setattr(
            _comm,
            "tp_comm_overlap",
            bool(_yaml_tp_overlap),
            f"config.comm_overlap.tp_comm_overlap={bool(_yaml_tp_overlap)!r}",
        )
    if _yaml_ep_overlap is not None and _comm is not None:
        _try_setattr(
            _comm,
            "overlap_moe_expert_parallel_comm",
            bool(_yaml_ep_overlap),
            f"config.comm_overlap.overlap_moe_expert_parallel_comm={bool(_yaml_ep_overlap)!r}",
        )
        if _yaml_ep_overlap:
            for _tgt_path, _tgt in _moe_attr_targets:
                if _tgt is None:
                    continue
                _try_setattr(
                    _tgt,
                    "moe_shared_expert_overlap",
                    False,
                    f"{_tgt_path}.moe_shared_expert_overlap=False (EP A2A overlap)",
                )
        for _tgt_path, _tgt in _moe_attr_targets:
            if _tgt is None:
                continue
            _try_setattr(
                _tgt,
                "overlap_moe_expert_parallel_comm",
                bool(_yaml_ep_overlap),
                f"{_tgt_path}.overlap_moe_expert_parallel_comm={bool(_yaml_ep_overlap)!r}",
            )
    if _yaml_delay_wgrad is not None and _comm is not None:
        _try_setattr(
            _comm,
            "delay_wgrad_compute",
            bool(_yaml_delay_wgrad),
            f"config.comm_overlap.delay_wgrad_compute={bool(_yaml_delay_wgrad)!r}",
        )
        for _tgt_path, _tgt in _moe_attr_targets:
            if _tgt is None:
                continue
            _try_setattr(
                _tgt,
                "delay_wgrad_compute",
                bool(_yaml_delay_wgrad),
                f"{_tgt_path}.delay_wgrad_compute={bool(_yaml_delay_wgrad)!r}",
            )

    # Nemotron 3 Nano is MCoreMambaModel (hybrid), not GPTModel — EP A2A overlap
    # needs GPTModel.build_schedule_plan and will crash at iter 0 if enabled.
    _is_hybrid_nemotron = bool(getattr(config.model, "is_hybrid_model", False))
    if _yaml_ep_overlap and _is_hybrid_nemotron:
        if rank0:
            print(
                "[comm-overlap] WARNING: overlap_moe_expert_parallel_comm=True requested "
                "but Nemotron hybrid (Mamba) model lacks GPTModel.build_schedule_plan — "
                "forcing overlap_moe_expert_parallel_comm=False",
                flush=True,
            )
        if _comm is not None:
            _try_setattr(
                _comm,
                "overlap_moe_expert_parallel_comm",
                False,
                "config.comm_overlap.overlap_moe_expert_parallel_comm=False (hybrid guard)",
            )
        for _tgt_path, _tgt in _moe_attr_targets:
            if _tgt is None:
                continue
            _try_setattr(
                _tgt,
                "overlap_moe_expert_parallel_comm",
                False,
                f"{_tgt_path}.overlap_moe_expert_parallel_comm=False (hybrid guard)",
            )

    # ── YAML-driven TE CUDA graphs (NVIDIA perf script parity) ───────────────
    _cg_impl = _cfg_get(resolved_cfg, "cuda_graph", "cuda_graph_impl")
    _cg_scope = _cfg_get(resolved_cfg, "cuda_graph", "cuda_graph_scope")
    if _cg_scope is None:
        _cg_scope = _cfg_get(resolved_cfg, "cuda_graph", "cuda_graph_modules")
    _cg_te_rng_tracker = _cfg_get(resolved_cfg, "cuda_graph", "te_rng_tracker")
    _cg_warmup_steps = _cfg_get(resolved_cfg, "cuda_graph", "cuda_graph_warmup_steps")
    if _cg_impl:
        _try_setattr(
            config.model,
            "cuda_graph_impl",
            _cg_impl,
            f"config.model.cuda_graph_impl={_cg_impl!r}",
        )
        if _cg_impl != "none":
            _try_setattr(
                config.rng,
                "te_rng_tracker",
                True,
                "config.rng.te_rng_tracker=True (CUDA graph)",
            )
            _try_setattr(
                config.model,
                "use_te_rng_tracker",
                True,
                "config.model.use_te_rng_tracker=True (CUDA graph)",
            )
    if _cg_te_rng_tracker is not None:
        _try_setattr(
            config.rng,
            "te_rng_tracker",
            bool(_cg_te_rng_tracker),
            f"config.rng.te_rng_tracker={bool(_cg_te_rng_tracker)!r} (YAML)",
        )
        _try_setattr(
            config.model,
            "use_te_rng_tracker",
            bool(_cg_te_rng_tracker),
            f"config.model.use_te_rng_tracker={bool(_cg_te_rng_tracker)!r} (YAML)",
        )
    if _cg_warmup_steps is not None:
        _try_setattr(
            config.model,
            "cuda_graph_warmup_steps",
            int(_cg_warmup_steps),
            f"config.model.cuda_graph_warmup_steps={int(_cg_warmup_steps)!r}",
        )
    if _cg_scope is not None:
        if isinstance(_cg_scope, str):
            _cg_scope = [_cg_scope]
        _try_setattr(
            config.model,
            "cuda_graph_scope",
            list(_cg_scope),
            f"config.model.cuda_graph_scope={list(_cg_scope)!r}",
        )

    # ── YAML-driven MoE dispatcher / fusion flags (explicit, non-stack) ─────
    _moe_yaml = _cfg_get(resolved_cfg, "moe") or {}
    if isinstance(_moe_yaml, dict) and _moe_yaml:
        _moe_field_map = (
            ("token_dispatcher_type", "moe_token_dispatcher_type"),
            ("flex_dispatcher_backend", "moe_flex_dispatcher_backend"),
            ("enable_deepep", "moe_enable_deepep"),
            ("router_force_load_balancing", "moe_router_force_load_balancing"),
            ("hybridep_num_sms", "moe_hybridep_num_sms"),
            ("router_fusion", "moe_router_fusion"),
            ("permute_fusion", "moe_permute_fusion"),
            ("shared_expert_overlap", "moe_shared_expert_overlap"),
            ("expert_capacity_factor", "moe_expert_capacity_factor"),
            ("moe_expert_capacity_factor", "moe_expert_capacity_factor"),
            ("pad_expert_input_to_capacity", "moe_pad_expert_input_to_capacity"),
            ("moe_pad_expert_input_to_capacity", "moe_pad_expert_input_to_capacity"),
            ("moe_token_drop_policy", "moe_token_drop_policy"),
            ("moe_router_dtype", "moe_router_dtype"),
        )
        for _yaml_key, _attr in _moe_field_map:
            if _yaml_key not in _moe_yaml:
                continue
            _val = _moe_yaml[_yaml_key]
            for _tgt_path, _tgt in _moe_attr_targets:
                if _tgt is None:
                    continue
                _try_setattr(_tgt, _attr, _val, f"{_tgt_path}.{_attr}={_val!r}")
        # Capacity padding is only supported with alltoall dispatcher.
        if (
            rank0
            and _moe_yaml.get("pad_expert_input_to_capacity") is True
            and _moe_yaml.get("token_dispatcher_type") == "flex"
        ):
            print(
                "[moe-config] WARN: moe_pad_expert_input_to_capacity=True with "
                "token_dispatcher_type='flex' is not supported. Use "
                "token_dispatcher_type='alltoall' for static-shape MoE padding.",
                flush=True,
            )

    _fp8_yaml = _cfg_get(resolved_cfg, "fp8") or {}
    if isinstance(_fp8_yaml, dict) and _fp8_yaml:
        _fp8_format = _fp8_yaml.get("fp8_format")
        if _fp8_format:
            _try_setattr(config.model, "fp8", _fp8_format, f"config.model.fp8={_fp8_format!r}")
        _fp8_field_map = (
            ("fp8_recipe", "fp8_recipe"),
            ("fp8_param_gather", "fp8_param_gather"),
            ("fp8_current_scaling", "fp8_current_scaling"),
        )
        for _yaml_key, _attr in _fp8_field_map:
            if _yaml_key not in _fp8_yaml:
                continue
            _val = _fp8_yaml[_yaml_key]
            for _tgt_path, _tgt in (
                ("config.model", getattr(config, "model", None)),
                ("config.optimizer", getattr(config, "optimizer", None)),
                ("config.ddp", getattr(config, "ddp", None)),
            ):
                if _tgt is None:
                    continue
                _try_setattr(_tgt, _attr, _val, f"{_tgt_path}.{_attr}={_val!r}")

    _etp = _cfg_get(resolved_cfg, "parallel", "expert_tensor_parallel_size")
    if _etp is not None:
        _try_setattr(
            config.model,
            "expert_tensor_parallel_size",
            int(_etp),
            f"config.model.expert_tensor_parallel_size={int(_etp)}",
        )

    _cp = _cfg_get(resolved_cfg, "parallel", "context_parallel_size")
    if _cp is not None:
        _try_setattr(
            config.model,
            "context_parallel_size",
            int(_cp),
            f"config.model.context_parallel_size={int(_cp)}",
        )

    _cpu_offload_act = _cfg_get(resolved_cfg, "model", "cpu_offloading_activations")
    if _cpu_offload_act is not None:
        _try_setattr(
            config.model,
            "cpu_offloading_activations",
            bool(_cpu_offload_act),
            f"config.model.cpu_offloading_activations={bool(_cpu_offload_act)!r}",
        )

    # ── NVIDIA-recommended MoE performance stack (env-gated) ────────────────
    # NVIDIA_MOE_STACK=1 enables the full NVIDIA Megatron-Core MoE tuning
    # guide recommendations in one bundle. Ref:
    #   https://docs.nvidia.com/megatron-core/developer-guide/nightly/
    #     user-guide/features/moe.html#general-performance-tips
    #
    # SILENT-MISCONFIG NOTE: rank-0 [moe-config] print showed
    #   flex_dispatcher_backend='deepep'  enable_deepep=False
    # for weeks — the dispatcher was nominally 'deepep' but the runtime
    # gate was False, so we were on the non-deepep flex path. This gate
    # fixes that + adds the fusion/overlap/GC stack.
    #
    # Realistic H200 delta vs current baseline (grouped_gemm-only): +8-15 pp
    # MFU on our TP=1/EP=8 topology. tp_comm_overlap is a no-op at TP=1.
    #
    # Do NOT enable this on the same run as a 54859 reproduction attempt —
    # it introduces a confounder. Enable on isolated A/B arms only.
    #
    # Companion env vars must be set in the .sh wrapper (not here):
    #   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    #   export NCCL_NVLS_ENABLE=0
    if run.nvidia_moe_stack:
        # 1) DeepEP dispatcher — flip the runtime gate that was silently False
        # 2) Fusion kernels — router / permute / cross-entropy
        # 3) COMPAT FIX (post-55927): the nemotron_3_nano Bridge recipe sets
        #    `moe_shared_expert_overlap=True` by default. Bridge's config
        #    validator rejects that combined with the flex+deepep dispatcher
        #    ("moe_shared_expert_overlap only works with alltoall token
        #    dispatcher"). We MUST force it False before enabling deepep,
        #    otherwise every NVIDIA_MOE_STACK arm crashes at config
        #    validation before training starts.
        _moe_targets = [
            ("config.model", getattr(config, "model", None)),
            ("config.model.moe", getattr(getattr(config, "model", None), "moe", None) if getattr(config, "model", None) else None),
            ("config.model.transformer", getattr(getattr(config, "model", None), "transformer", None) if getattr(config, "model", None) else None),
            ("config.model.transformer.moe",
                getattr(getattr(getattr(config, "model", None), "transformer", None), "moe", None)
                if (getattr(config, "model", None) and getattr(getattr(config, "model", None), "transformer", None))
                else None),
        ]
        for _tgt_path, _tgt in _moe_targets:
            if _tgt is None:
                continue
            # Disable shared-expert-overlap FIRST — Bridge validates late,
            # so leaving it True while switching dispatcher trips the check.
            _try_setattr(_tgt, "moe_shared_expert_overlap", False,
                         f"{_tgt_path}.moe_shared_expert_overlap=False (deepep incompat)")
            _try_setattr(_tgt, "moe_enable_deepep", True,
                         f"{_tgt_path}.moe_enable_deepep=True")
            _try_setattr(_tgt, "moe_token_dispatcher_type", "flex",
                         f"{_tgt_path}.moe_token_dispatcher_type='flex'")
            _try_setattr(_tgt, "moe_flex_dispatcher_backend", "deepep",
                         f"{_tgt_path}.moe_flex_dispatcher_backend='deepep'")
            _try_setattr(_tgt, "moe_router_fusion", True,
                         f"{_tgt_path}.moe_router_fusion=True")
            _try_setattr(_tgt, "moe_permute_fusion", True,
                         f"{_tgt_path}.moe_permute_fusion=True")
            _try_setattr(_tgt, "cross_entropy_loss_fusion", True,
                         f"{_tgt_path}.cross_entropy_loss_fusion=True")
            _try_setattr(_tgt, "cross_entropy_fusion_impl", "native",
                         f"{_tgt_path}.cross_entropy_fusion_impl='native'")

        # 3) Distributed optimizer + comm overlap
        _dist_targets = [
            ("config.optimizer", getattr(config, "optimizer", None)),
            ("config.ddp", getattr(config, "ddp", None)),
            ("config.distributed", getattr(config, "distributed", None)),
            ("config.model", getattr(config, "model", None)),
        ]
        for _tgt_path, _tgt in _dist_targets:
            if _tgt is None:
                continue
            _try_setattr(_tgt, "use_distributed_optimizer", True,
                         f"{_tgt_path}.use_distributed_optimizer=True")
            _try_setattr(_tgt, "overlap_param_gather", True,
                         f"{_tgt_path}.overlap_param_gather=True")
            _try_setattr(_tgt, "overlap_grad_reduce", True,
                         f"{_tgt_path}.overlap_grad_reduce=True")
            # tp_comm_overlap is a no-op at TP=1 (our current topology)
            # but set it anyway so future TP>1 arms inherit correctly.
            _try_setattr(_tgt, "tp_comm_overlap", True,
                         f"{_tgt_path}.tp_comm_overlap=True")

        # 4) Manual GC to avoid Python-side jitter across ranks
        _gc_targets = [
            ("config.train", getattr(config, "train", None)),
            ("config.training", getattr(config, "training", None)),
            ("config.runtime", getattr(config, "runtime", None)),
        ]
        for _tgt_path, _tgt in _gc_targets:
            if _tgt is None:
                continue
            _try_setattr(_tgt, "manual_gc", True,
                         f"{_tgt_path}.manual_gc=True")
            _try_setattr(_tgt, "manual_gc_interval", 10,
                         f"{_tgt_path}.manual_gc_interval=10")
            _try_setattr(_tgt, "manual_gc_eval", True,
                         f"{_tgt_path}.manual_gc_eval=True")

        if rank0:
            print("[nvidia-moe-stack] Applied NVIDIA MoE tuning stack: "
                  "DeepEP + router/permute/CE fusion + distrib-opt overlap + "
                  "manual GC. See rank0 SET/SKIP lines above for which "
                  "attrs were accepted by Bridge.", flush=True)
            _alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")
            _nvls  = os.environ.get("NCCL_NVLS_ENABLE", "<unset>")
            print(f"[nvidia-moe-stack] env: "
                  f"PYTORCH_CUDA_ALLOC_CONF={_alloc!r}  "
                  f"NCCL_NVLS_ENABLE={_nvls!r}", flush=True)
            if _alloc == "<unset>" or "expandable_segments" not in _alloc:
                print("[nvidia-moe-stack] WARN: PYTORCH_CUDA_ALLOC_CONF "
                      "does not include 'expandable_segments:True' — set it "
                      "in the .sh wrapper for full stack benefit.", flush=True)
            if _nvls == "<unset>" or _nvls != "0":
                print("[nvidia-moe-stack] WARN: NCCL_NVLS_ENABLE not set to "
                      "'0' — set it in the .sh wrapper to reduce NCCL "
                      "memory overhead.", flush=True)

    # ── DeepEP-only ablation gate (Phase C attribution) ────────────────────
    # DEEPEP_ONLY=1 flips the DeepEP dispatcher gate + applies the
    # `moe_shared_expert_overlap=False` compat fix, and NOTHING ELSE.
    # No fusion kernels, no comm overlap, no manual GC.
    #
    # Purpose: isolate DeepEP's contribution to the NVIDIA_MOE_STACK win.
    # If Phase A→B showed +N% and Phase A→C shows +M%, then:
    #   DeepEP alone contributes M%
    #   fusion + overlap + GC contribute (N-M)%
    #
    # MUTUALLY EXCLUSIVE with NVIDIA_MOE_STACK — if both are set,
    # NVIDIA_MOE_STACK wins (it's a superset) and DEEPEP_ONLY is a no-op
    # with a WARN.
    if run.deepep_only:
        if run.nvidia_moe_stack:
            if rank0:
                print("[deepep-only] WARN: NVIDIA_MOE_STACK=1 is also set — "
                      "the full stack (which includes DeepEP) already "
                      "applied. DEEPEP_ONLY is a no-op in this configuration.",
                      flush=True)
        else:
            _deepep_targets = [
                ("config.model", getattr(config, "model", None)),
                ("config.model.moe", getattr(getattr(config, "model", None), "moe", None) if getattr(config, "model", None) else None),
                ("config.model.transformer", getattr(getattr(config, "model", None), "transformer", None) if getattr(config, "model", None) else None),
                ("config.model.transformer.moe",
                    getattr(getattr(getattr(config, "model", None), "transformer", None), "moe", None)
                    if (getattr(config, "model", None) and getattr(getattr(config, "model", None), "transformer", None))
                    else None),
            ]
            for _tgt_path, _tgt in _deepep_targets:
                if _tgt is None:
                    continue
                # Disable shared-expert-overlap FIRST — Bridge validates late,
                # so leaving it True while switching dispatcher trips the check.
                # See GOTCHAS §32 sibling learning from job 55927.
                _try_setattr(_tgt, "moe_shared_expert_overlap", False,
                             f"{_tgt_path}.moe_shared_expert_overlap=False (deepep incompat)")
                _try_setattr(_tgt, "moe_enable_deepep", True,
                             f"{_tgt_path}.moe_enable_deepep=True")
                _try_setattr(_tgt, "moe_token_dispatcher_type", "flex",
                             f"{_tgt_path}.moe_token_dispatcher_type='flex'")
                _try_setattr(_tgt, "moe_flex_dispatcher_backend", "deepep",
                             f"{_tgt_path}.moe_flex_dispatcher_backend='deepep'")

            if rank0:
                print("[deepep-only] Applied DeepEP-only ablation: "
                      "moe_enable_deepep=True + moe_shared_expert_overlap=False. "
                      "NO fusion kernels, NO overlap flags, NO manual GC. "
                      "Purpose: isolate DeepEP delta from full NVIDIA stack.",
                      flush=True)
                # Companion env-var note — DeepEP alone doesn't strictly need
                # NCCL_NVLS_ENABLE=0 or expandable_segments, but keeping them
                # unset avoids conflating benefits.
                _alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")
                _nvls  = os.environ.get("NCCL_NVLS_ENABLE", "<unset>")
                if _alloc != "<unset>" and "expandable_segments" in _alloc:
                    print("[deepep-only] NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments "
                          "is set — that's an NVIDIA-stack companion. Leaving it "
                          "on will NOT confound DeepEP isolation (memory-side, "
                          "no dispatcher interaction), but for cleanest A/B, unset "
                          "it in the .sh.", flush=True)
                if _nvls == "0":
                    print("[deepep-only] NOTE: NCCL_NVLS_ENABLE=0 is set — that's "
                          "an NVIDIA-stack companion. NVLS interacts with "
                          "AllReduce paths that DeepEP-alt-to-all also uses. "
                          "Unset in the .sh for cleanest A/B.", flush=True)

    # ── Diagnostic logging (env-gated per Avinash's isolation checklist) ────
    # Added 2026-07-09 to catch dropless-MoE router imbalance as step-time
    # bottleneck (§30 GOTCHAS). Every flag is a config-time nop if the attr
    # doesn't exist — safe to leave on for all runs.

    # Prefer resolved YAML (run.*) so logging: works without shell env: exports.
    # Env still wins when explicitly set (LOG_THROUGHPUT=0/1 etc.).
    _log_throughput = run.log_throughput
    if os.environ.get("LOG_THROUGHPUT", "") != "":
        _log_throughput = os.environ["LOG_THROUGHPUT"] not in ("0", "false", "False", "no", "No")
    if _log_throughput:
        # Enable Megatron's built-in throughput/MFU line in stdout — removes
        # need for offline phase2_mfu_calc.py.
        for _tgt in (config, getattr(config, "logger", None), getattr(config, "training", None)):
            if _tgt is None:
                continue
            _try_setattr(_tgt, "log_throughput", True, "config.<...>.log_throughput")

    # LOG_INTERVAL env wins; else honor logging.log_interval from YAML.
    if os.environ.get("LOG_INTERVAL"):
        _log_interval = int(os.environ["LOG_INTERVAL"])
    else:
        _log_interval = int(run.log_interval)
    _try_setattr(config.logger, "log_interval", _log_interval,
                 f"config.logger.log_interval={_log_interval}")

    _moe_per_layer = run.moe_per_layer_logging
    if os.environ.get("MOE_PER_LAYER_LOGGING", "") != "":
        _moe_per_layer = os.environ["MOE_PER_LAYER_LOGGING"] not in (
            "0", "false", "False", "no", "No",
        )
    if _moe_per_layer:
        # Per-layer MoE aux loss + expert-load histograms — reveals which
        # layers have skewed routing. Bridge 0.3.x has multiple candidate paths.
        for _tgt_path, _tgt in [
            ("config", config),
            ("config.model", getattr(config, "model", None)),
            ("config.model.transformer", getattr(getattr(config, "model", None), "transformer", None) if getattr(config, "model", None) else None),
            ("config.logger", getattr(config, "logger", None)),
            ("config.training", getattr(config, "training", None)),
        ]:
            if _tgt is None:
                continue
            _try_setattr(_tgt, "moe_per_layer_logging", True, f"{_tgt_path}.moe_per_layer_logging")
            _try_setattr(_tgt, "moe_log_expert_load", True, f"{_tgt_path}.moe_log_expert_load")
            _try_setattr(_tgt, "log_expert_load_to_tb", True, f"{_tgt_path}.log_expert_load_to_tb")
            _try_setattr(_tgt, "moe_log_expert_load_to_tb", True, f"{_tgt_path}.moe_log_expert_load_to_tb")

    if run.log_straggler or os.environ.get("LOG_STRAGGLER", "0") == "1":
        config.straggler = StragglerDetectionConfig(log_straggler=True)
        if rank0:
            print("[straggler] Enabled Bridge StragglerDetectionConfig "
                  "(log_straggler=True)", flush=True)

    # DDP grad NaN check — per-bucket validate_result() before reduce_scatter.
    # Disabling removes the isnan/is_nonzero/.item()/cudaStreamSynchronize chain
    # seen in profiler traces (param_and_grad_buffer.check_grads).
    _check_nan_grad = run.check_for_nan_in_grad
    if _check_nan_grad is None:
        _check_nan_grad = args.check_for_nan_in_grad
    if _check_nan_grad is None and os.environ.get("CHECK_FOR_NAN_IN_GRAD", "") != "":
        _check_nan_grad = os.environ["CHECK_FOR_NAN_IN_GRAD"] not in (
            "0", "false", "False", "no", "No",
        )
    if _check_nan_grad is not None:
        for _tgt_path, _tgt in [
            ("config.ddp", getattr(config, "ddp", None)),
            ("config.distributed", getattr(config, "distributed", None)),
        ]:
            if _tgt is None:
                continue
            _try_setattr(
                _tgt, "check_for_nan_in_grad", _check_nan_grad,
                f"{_tgt_path}.check_for_nan_in_grad={_check_nan_grad}",
            )
        if rank0:
            print(f"[ddp] check_for_nan_in_grad={_check_nan_grad!r}", flush=True)

    # Stage 0 free host-sync win: check_for_large_grads triggers the same
    # isnan/is_nonzero/.item()/cudaStreamSynchronize chain per bucket as
    # check_for_nan_in_grad (see param_and_grad_buffer.check_grads). Default
    # is already False in Megatron-Core, but set explicitly so it can't be
    # silently re-enabled by a recipe change.
    _check_large_grads = run.check_for_large_grads
    if _check_large_grads is not None:
        for _tgt_path, _tgt in [
            ("config.ddp", getattr(config, "ddp", None)),
            ("config.distributed", getattr(config, "distributed", None)),
        ]:
            if _tgt is None:
                continue
            _try_setattr(
                _tgt, "check_for_large_grads", _check_large_grads,
                f"{_tgt_path}.check_for_large_grads={_check_large_grads}",
            )
        if rank0:
            print(f"[ddp] check_for_large_grads={_check_large_grads!r}", flush=True)

    # Stage 0 free host-sync win: rerun_state_machine's loss NaN/spiky checks
    # do a .item()-style host sync on the loss tensor every iteration (Bridge
    # default check_for_nan_in_loss=True). This is the "loss" half of
    # Megatron-LM's legacy --check-for-nan-in-loss-and-grad flag; the "grad"
    # half is check_for_nan_in_grad above.
    _check_nan_loss = run.check_for_nan_in_loss
    _check_spiky_loss = run.check_for_spiky_loss
    _rerun_smc = getattr(config, "rerun_state_machine", None)
    if _rerun_smc is not None:
        if _check_nan_loss is not None:
            _try_setattr(
                _rerun_smc, "check_for_nan_in_loss", _check_nan_loss,
                f"config.rerun_state_machine.check_for_nan_in_loss={_check_nan_loss}",
            )
        if _check_spiky_loss is not None:
            _try_setattr(
                _rerun_smc, "check_for_spiky_loss", _check_spiky_loss,
                f"config.rerun_state_machine.check_for_spiky_loss={_check_spiky_loss}",
            )
        if rank0:
            print(
                f"[rerun] check_for_nan_in_loss={_check_nan_loss!r} "
                f"check_for_spiky_loss={_check_spiky_loss!r}", flush=True,
            )

    # Distributed optimizer + grad/param-gather overlap from YAML distributed:
    # (defaults True — stage0 free host-sync win / recipe parity).
    _dist_yaml = _cfg_get(resolved_cfg, "distributed") or {}
    if not isinstance(_dist_yaml, dict):
        _dist_yaml = {}
    _use_dist_opt = bool(_dist_yaml.get("use_distributed_optimizer", True))
    _overlap_grad = bool(_dist_yaml.get("overlap_grad_reduce", True))
    _overlap_param = bool(_dist_yaml.get("overlap_param_gather", True))
    for _tgt_path, _tgt in [
        ("config.ddp", getattr(config, "ddp", None)),
        ("config.distributed", getattr(config, "distributed", None)),
    ]:
        if _tgt is None:
            continue
        _try_setattr(_tgt, "use_distributed_optimizer", _use_dist_opt,
                     f"{_tgt_path}.use_distributed_optimizer={_use_dist_opt!r} (YAML/stage0)")
        _try_setattr(_tgt, "overlap_grad_reduce", _overlap_grad,
                     f"{_tgt_path}.overlap_grad_reduce={_overlap_grad!r} (YAML/stage0)")
        _try_setattr(_tgt, "overlap_param_gather", _overlap_param,
                     f"{_tgt_path}.overlap_param_gather={_overlap_param!r} (YAML/stage0)")
    if rank0:
        _ddp_check = getattr(config, "ddp", None)
        print(
            f"[stage0] use_distributed_optimizer="
            f"{getattr(_ddp_check, 'use_distributed_optimizer', None)!r} "
            f"overlap_grad_reduce={getattr(_ddp_check, 'overlap_grad_reduce', None)!r} "
            f"overlap_param_gather={getattr(_ddp_check, 'overlap_param_gather', None)!r} "
            f"CUDA_DEVICE_MAX_CONNECTIONS="
            f"{os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', '<unset>')!r}",
            flush=True,
        )

    if os.environ.get("LOG_TIMERS_LEVEL", "") != "":
        _timers_level = os.environ["LOG_TIMERS_LEVEL"]
    else:
        _timers_level = str(run.log_timers_level) if run.log_timers_level else ""
    if _timers_level and int(_timers_level) > 0:
        # Level-2 timers with minmax option — reveals fwd/bwd max-min gap
        # across ranks. Big gap = straggler EP rank (router imbalance).
        # Bridge 0.3.x nests timing attrs in multiple candidate locations;
        # try all of them defensively.
        _lvl = int(_timers_level)
        for _tgt_path, _tgt in [
            ("config", config),
            ("config.training", getattr(config, "training", None)),
            ("config.logger", getattr(config, "logger", None)),
            ("config.runtime", getattr(config, "runtime", None)),
            ("config.optimization", getattr(config, "optimization", None)),
            ("config.timers", getattr(config, "timers", None)),
            ("config.training.timers", getattr(getattr(config, "training", None), "timers", None) if getattr(config, "training", None) else None),
        ]:
            if _tgt is None:
                continue
            _try_setattr(_tgt, "timing_log_level", _lvl, f"{_tgt_path}.timing_log_level={_lvl}")
            _try_setattr(_tgt, "timing_log_option", "minmax", f"{_tgt_path}.timing_log_option=minmax")
            _try_setattr(_tgt, "log_timers_to_tensorboard", True, f"{_tgt_path}.log_timers_to_tensorboard")
            _try_setattr(_tgt, "barrier_with_L1_time", True, f"{_tgt_path}.barrier_with_L1_time")

    # Verify MoE dispatcher backend — Avinash flagged possible silent fallback
    # to naive dispatcher (moe_flex_dispatcher_backend=deepep + moe_enable_deepep=false).
    # Log which is actually configured so we know at rank0 print.
    if rank0:
        _moe_backend = getattr(config.model, "moe_flex_dispatcher_backend", "<unset>")
        _moe_enable_deepep = getattr(config.model, "moe_enable_deepep", "<unset>")
        _moe_grouped = getattr(config.model, "moe_grouped_gemm", "<unset>")
        print(f"[moe-config] flex_dispatcher_backend={_moe_backend!r}  "
              f"enable_deepep={_moe_enable_deepep!r}  "
              f"grouped_gemm={_moe_grouped!r}", flush=True)

    def _parse_profile_ranks():
        """Global ranks to record Chrome traces. None → all ranks."""
        if run.profile_all_ranks:
            return None
        raw = run.profile_ranks
        return {int(r.strip()) for r in raw.split(",") if r.strip()}

    # ── PyTorch profiler enablement (env-gated) ─────────────────────────────
    # TORCH_PROFILE=1 enables per-op timing capture. Attempts Bridge/Megatron
    # config-based first (cleanest, no monkey-patch); falls back to wrapping
    # pretrain() in torch.profiler.profile() context (see block below at pretrain
    # call site).
    #
    # Overhead: 10-20% runtime cost. Trace file size scales with iters.
    # PROFILE_STEP_START/END bound the capture window — set to skip warmup.
    # PROFILE_RANKS=0,8 records selected ranks (default: 0). PROFILE_ALL_RANKS=1
    # records every rank (large trace volume on 128-GPU jobs).
    #
    # Output: Chrome trace JSON at PROFILE_OUTPUT_DIR/trace_rank<N>.json
    #   View with: chrome://tracing OR perfetto.dev (better UI)
    if run.torch_profile:
        _profile_start = run.profile_step_start
        _profile_end   = run.profile_step_end
        _profile_ranks_set = _parse_profile_ranks()
        _profile_ranks_list = (
            sorted(_profile_ranks_set) if _profile_ranks_set is not None else [0]
        )
        # Try Bridge/Megatron config attrs first (config-based path is
        # cleanest — Megatron handles trace file naming/rank scoping).
        _config_profile_ok = False
        for _tgt in (getattr(config, "training", None), config):
            if _tgt is None:
                continue
            if hasattr(_tgt, "profile"):
                _try_setattr(_tgt, "profile", True, f"config.<...>.profile")
                _try_setattr(_tgt, "use_pytorch_profiler", True,
                             f"config.<...>.use_pytorch_profiler")
                _try_setattr(_tgt, "profile_step_start", _profile_start,
                             f"config.<...>.profile_step_start={_profile_start}")
                _try_setattr(_tgt, "profile_step_end", _profile_end,
                             f"config.<...>.profile_step_end={_profile_end}")
                _try_setattr(_tgt, "profile_ranks", _profile_ranks_list,
                             f"config.<...>.profile_ranks={_profile_ranks_list}")
                _config_profile_ok = True
                break
        if rank0:
            if _config_profile_ok:
                _ranks_msg = (
                    "all ranks" if _profile_ranks_set is None
                    else f"ranks {_profile_ranks_list}"
                )
                print(f"[profile] Bridge config-based profiler ENABLED "
                      f"(steps {_profile_start}-{_profile_end}, {_ranks_msg})",
                      flush=True)
            else:
                print(f"[profile] Bridge config attrs absent; will use "
                      f"torch.profiler.profile() context wrapping pretrain()",
                      flush=True)

    # Tokenizer staging (§ 5).
    staged_tokenizer = stage_tokenizer_for_megatron(
        run.phase1_modified, Path(results_dir),
    )
    config.tokenizer.tokenizer_type  = "HuggingFaceTokenizer"
    config.tokenizer.tokenizer_model = str(staged_tokenizer)

    # Data (§ 10 — wire_dataset_for_training asserts mock=False + split_matrix).
    wire_dataset_for_training(
        config.dataset,
        data_root=run.data_root,
        cache_dir=run.cache_dir,
        seq_length=run.seq_length,
    )
    config.dataset.num_workers = run.dataset_num_workers
    if rank0:
        print(f"[data] dataset.num_workers={run.dataset_num_workers}", flush=True)

    # Training
    config.train.train_iters       = target_iters
    config.train.global_batch_size = run.global_batch_size
    config.train.micro_batch_size  = run.micro_batch_size

    # LR schedule
    peak_lr = run.lr
    min_lr  = run.min_lr
    if run.lr_decay_style == "wsd":
        applied = _configure_wsd(config, target_iters, run.warmup_iters,
                                 peak_lr, min_lr, run.wsd_stable_frac)
        if rank0:
            print(f"[sched] applied: {applied}")
    else:
        config.optimizer.lr              = peak_lr
        config.optimizer.min_lr          = min_lr
        config.scheduler.lr_warmup_iters = run.warmup_iters
        config.scheduler.lr_decay_iters  = target_iters
        config.scheduler.lr_decay_style  = "cosine"
        if rank0:
            print(f"[sched] applied: cosine")

    # Post-configure LR assertion (§ 16).
    actual_lr = config.optimizer.lr
    if abs(actual_lr - run.lr) / run.lr > 0.01:
        raise RuntimeError(
            f"LR wiring failed: requested {run.lr}, "
            f"config.optimizer.lr={actual_lr}. See GOTCHAS § 16."
        )
    if rank0:
        print(f"[lr] peak={config.optimizer.lr}, min={config.optimizer.min_lr}, "
              f"warmup_iters={config.scheduler.lr_warmup_iters}, "
              f"decay_style={config.scheduler.lr_decay_style}", flush=True)

    # Eval (§ 12 — Nemotron-H eval on config.train).
    config.train.eval_interval = run.eval_interval
    config.train.eval_iters    = run.eval_iters

    # ── Resume from Step 0 with finetune=True (§ 1) ─────────────────────
    # LOAD-BEARING: without finetune=True, Bridge restores Step 0's iter
    # counter, optimizer state, scheduler state, and RNG — arm would silently
    # continue Step 0 at the new LR.
    config.checkpoint.save          = results_dir
    if run.init_from_hf:
        config.checkpoint.load = None
        config.checkpoint.finetune = False
        if rank0:
            print("[ckpt] init_from_hf=true — fresh train from HF weights (no Step0 load)")
    else:
        config.checkpoint.load = run.step0_megatron_dir
        config.checkpoint.finetune = run.finetune
    config.checkpoint.save_interval = run.save_interval

    config.logger.tensorboard_dir = tb_dir
    # log_interval already applied above from run.log_interval / LOG_INTERVAL.

    # ── WandB integration (runbook §7) ──────────────────────────────────
    # Wired via env vars set by the launcher (phase2_lr_sweep_launcher_mn.sh)
    # and .env (WANDB_API_KEY). Layered per §20 env-init parity: shell sets
    # keys/dirs; Python translates to config.logger.* fields Bridge respects.
    #
    # Defensive gate: only wire wandb_project if API key is present. Missing
    # key means the user launched an ad-hoc arm without .env — TensorBoard
    # is still active as a fallback logger, so no data is lost.
    _wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
    _wandb_proj = os.environ.get("WANDB_PROJECT", "").strip()
    if _wandb_key and _wandb_proj:
        config.logger.wandb_project  = _wandb_proj
        # wandb_exp_name is Bridge's field for the individual run name.
        _run_name = os.environ.get("WANDB_NAME", "").strip()
        if _run_name:
            config.logger.wandb_exp_name = _run_name
        _entity = os.environ.get("WANDB_ENTITY", "").strip()
        if _entity:
            config.logger.wandb_entity = _entity
        _wandb_dir = os.environ.get("WANDB_DIR", "").strip()
        if _wandb_dir:
            config.logger.wandb_save_dir = _wandb_dir
        if rank0:
            print(
                f"[wandb] wired project={_wandb_proj!r} run={_run_name!r} "
                f"entity={_entity or '(default)'!r} mode="
                f"{os.environ.get('WANDB_MODE','online')} dir={_wandb_dir or '(default)'}",
                flush=True,
            )
    elif rank0:
        print(
            f"[wandb] SKIPPED — "
            f"WANDB_API_KEY {'set' if _wandb_key else 'MISSING'}, "
            f"WANDB_PROJECT {'set' if _wandb_proj else 'MISSING'}. "
            f"TensorBoard still active at {tb_dir}.",
            flush=True,
        )

    # ── Optional: PRINT_PARAMS diagnostic ────────────────────────────────
    if run.print_params_then_exit:
        def _print_and_exit(model_chunks):
            rank  = dist.get_rank() if dist.is_initialized() else 0
            world = dist.get_world_size() if dist.is_initialized() else 1
            for chunk in (model_chunks if isinstance(model_chunks, list) else [model_chunks]):
                for n, p in chunk.named_parameters():
                    tp_flag = getattr(p, "tensor_model_parallel", None)
                    partition_dim = getattr(p, "partition_dim", None)
                    print(
                        f"[rank {rank}/{world}]  {n:80s}  "
                        f"shape={tuple(p.shape)}  "
                        f"tp_parallel={tp_flag}  partition_dim={partition_dim}",
                        flush=True,
                    )
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                print("\n[--print-params-then-exit] set; all ranks exiting.")
            sys.exit(0)

        config.model.register_post_wrap_hook(_print_and_exit)

    # ── Run ──────────────────────────────────────────────────────────────
    t0 = time.time()

    _profile_env = run.torch_profile
    _profile_fallback_needed = _profile_env and not locals().get("_config_profile_ok", False)

    if _profile_fallback_needed:
        # Fallback path: Bridge config didn't accept profile attrs. Wrap
        # pretrain() with torch.profiler.profile(schedule=...) and drive the
        # schedule from a forward_step wrapper.
        #
        # HOW THE BOUNDED WINDOW WORKS
        #   The `schedule(wait, warmup, active, repeat=1)` state machine
        #   advances one step per profiler.step() call. Since Bridge doesn't
        #   expose a "post-iter" hook, we wrap `forward_step` and call
        #   profiler.step() once per (micro_per_iter) micro-batches, i.e.
        #   once per training iter. This gives us:
        #     - iters [0, wait):                  no capture (warm cluster)
        #     - iter  [wait, wait+warmup):        stage-in, no trace
        #     - iters [wait+warmup, wait+warmup+active): TRACED
        #     - iters after that:                 no capture (idle)
        #
        # WINDOW MAPPING FROM PROFILE_STEP_START/END
        #   PROFILE_STEP_START=3, PROFILE_STEP_END=7 →
        #     wait=2, warmup=1, active=5 → traces iters 3..7 (inclusive).
        #
        # PP CAVEAT
        #   Assumes PP=1 → forward_step called once per micro-batch.
        #   For PP>1 or interleaved 1F1B, the call-count per iter changes;
        #   REPLICA_MP=(1,8,1) here matches the PP=1 assumption. When we
        #   move to PP>1, extend _micro_per_iter accordingly.
        import torch.profiler as _tprof
        _profile_out = os.environ.get(
            "PROFILE_OUTPUT_DIR",
            profile_output_dir,
        )
        _profile_rank = int(os.environ.get("RANK", "0"))
        _profile_ranks_set = _parse_profile_ranks()
        if _profile_ranks_set is None:
            _record_this_rank = True
        else:
            _record_this_rank = _profile_rank in _profile_ranks_set

        if _record_this_rank:
            os.makedirs(_profile_out, exist_ok=True)
            _trace_path = os.path.join(_profile_out, f"trace_rank{_profile_rank}.json")

            # Derive micro-batches per iter (used to advance schedule
            # once per iter, not once per micro-batch).
            _world = int(os.environ.get("WORLD_SIZE", "1"))
            _tp_ep_pp = os.environ.get("REPLICA_MP", "1,1,1").split(",")
            _tp = int(_tp_ep_pp[0]) if len(_tp_ep_pp) >= 1 else 1
            _pp = int(_tp_ep_pp[2]) if len(_tp_ep_pp) >= 3 else 1
            _dp = max(1, _world // max(1, _tp * _pp))
            _micro_per_iter = max(
                1,
                run.global_batch_size // max(1, run.micro_batch_size * _dp),
            )

            _wait   = max(0, _profile_start - 1)
            _warmup = 1
            _active = max(1, _profile_end - _profile_start + 1)

            if rank0:
                print(f"[profile] Bounded torch.profiler schedule "
                      f"wait={_wait} warmup={_warmup} active={_active} repeat=1 "
                      f"→ traces iters {_profile_start}..{_profile_end}",
                      flush=True)
                print(f"[profile] micro_per_iter={_micro_per_iter} "
                      f"(GBS={run.global_batch_size} MBS={run.micro_batch_size} "
                      f"DP={_dp} TP={_tp} PP={_pp})", flush=True)
                _ranks_msg = (
                    "all ranks" if _profile_ranks_set is None
                    else f"ranks {sorted(_profile_ranks_set)}"
                )
                print(f"[profile] Recording {_ranks_msg}. "
                      f"Overhead ~10-20% (bounded window).", flush=True)
                if _pp > 1:
                    print(f"[profile] WARN: PP={_pp} > 1 detected. "
                          f"_micro_per_iter assumes 1 forward_step call per "
                          f"micro-batch; interleaved 1F1B may skew the "
                          f"schedule. Verify captured iters manually.",
                          flush=True)
            print(f"[profile] rank {_profile_rank}: trace → {_trace_path}",
                  flush=True)

            _with_stack = os.environ.get("PROFILE_WITH_STACK", "1") not in (
                "0", "false", "False",
            )
            _profile_done = [False]
            _profile_exported = [False]
            _profile_stopped = [False]

            def _export_profile_trace(*, early: bool = False) -> None:
                if _profile_exported[0]:
                    return
                if not _profile_stopped[0]:
                    try:
                        _p.stop()
                    except Exception:
                        pass
                    _profile_stopped[0] = True
                try:
                    _p.export_chrome_trace(_trace_path)
                    _profile_exported[0] = True
                    _sz_mb = (
                        os.path.getsize(_trace_path) / 1e6
                        if os.path.exists(_trace_path) else 0.0
                    )
                    _tag = "early " if early else ""
                    print(
                        f"[profile] rank {_profile_rank}: {_tag}Chrome trace exported "
                        f"({_sz_mb:.1f} MB): {_trace_path}",
                        flush=True,
                    )
                    if (
                        os.environ.get("WANDB_PROFILE_UPLOAD", "0")
                        not in ("0", "false", "False")
                        and os.environ.get("WANDB_API_KEY", "").strip()
                    ):
                        try:
                            import wandb
                            if wandb.run is not None:
                                wandb.save(
                                    _trace_path,
                                    base_path=os.path.dirname(_trace_path),
                                    policy="now",
                                )
                                print(
                                    f"[profile] rank {_profile_rank}: trace uploaded to wandb",
                                    flush=True,
                                )
                        except Exception as _wandb_prof_err:
                            print(
                                f"[profile] rank {_profile_rank}: wandb trace upload "
                                f"skipped: {_wandb_prof_err}",
                                flush=True,
                            )
                except Exception as _exp_err:
                    print(
                        f"[profile] rank {_profile_rank}: export_chrome_trace "
                        f"FAILED: {_exp_err}. Profiler stopped but trace not "
                        f"written.",
                        flush=True,
                    )

            def _maybe_advance_profile() -> None:
                if _profile_done[0]:
                    return
                _mb_counter[0] += 1
                if _mb_counter[0] >= _micro_per_iter:
                    _mb_counter[0] = 0
                    _iter_counter[0] += 1
                    try:
                        _p.step()
                    except Exception:
                        pass
                    if _iter_counter[0] >= _profile_end:
                        _profile_done[0] = True
                        if rank0 or _profile_rank in (_profile_ranks_set or {0}):
                            print(
                                f"[profile] rank {_profile_rank}: window complete "
                                f"at iter {_iter_counter[0]}; stopping profiler "
                                f"before next iter (avoids DeepEP desync)",
                                flush=True,
                            )
                        _export_profile_trace(early=True)

            _p = _tprof.profile(
                activities=[
                    _tprof.ProfilerActivity.CPU,
                    _tprof.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=_with_stack,
                with_flops=False,
                schedule=_tprof.schedule(
                    wait=_wait, warmup=_warmup, active=_active, repeat=1,
                ),
            )

            _mb_counter   = [0]
            _iter_counter = [0]
            _original_fs  = forward_step

            # ── Signature-preserving wrapper (fix for 55920 crash) ──────────
            # Bridge inspects `forward_step_func` at call time to decide the
            # calling convention:
            #   3-arg signature (state, data_iterator, model) → state is
            #     bound and Bridge invokes as `fn(state, data_iterator, model)`.
            #   2-arg signature (data_iterator, model) → Bridge invokes as
            #     `fn(data_iterator, model)`.
            # A vararg wrapper `def w(*a, **kw)` collapses to 2-arg via
            # inspect, so Bridge passes `(data_iterator, model)` but the
            # underlying `gpt_step.forward_step` expects `state` first →
            # TypeError: missing 'model'. See job 55920 post-mortem.
            #
            # Fix: build the wrapper with the ORIGINAL signature so Bridge's
            # inspection sees the right arg count and does the same binding
            # it would for the unwrapped function.
            import functools as _ftools
            import inspect as _inspect

            _orig_sig = None
            try:
                _orig_sig = _inspect.signature(_original_fs)
            except (TypeError, ValueError):
                _orig_sig = None

            if _orig_sig is not None:
                # Reconstruct the exact positional-param names AND their
                # default values, forwarding them into _original_fs.
                #
                # BUG FIX (post-55926): earlier version appended just
                # `_pname` and dropped defaults. When Bridge introspected
                # forward_step and saw e.g. `return_schedule_plan=False`,
                # then called our wrapper with only (state, data_iterator,
                # model), Python raised "missing 1 required positional
                # argument: 'return_schedule_plan'" because our generated
                # wrapper had no default for it.
                #
                # Fix: inject each default value into the exec namespace
                # under a unique name (`_dflt_<param>`) and reference that
                # name in the generated signature. Handles arbitrary default
                # types (bool, None, callables, dataclasses) without repr()
                # trouble.
                _pos_params  = []
                _fwd_pos     = []
                _kw_forward  = []
                _defaults_ns = {}
                _has_var_pos = False
                _has_kw_only = False
                for _pname, _pobj in _orig_sig.parameters.items():
                    if _pobj.kind in (
                        _inspect.Parameter.POSITIONAL_ONLY,
                        _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ):
                        if _pobj.default is not _inspect.Parameter.empty:
                            _dflt = f"_dflt_{_pname}"
                            _defaults_ns[_dflt] = _pobj.default
                            _pos_params.append(f"{_pname}={_dflt}")
                        else:
                            _pos_params.append(_pname)
                        _fwd_pos.append(_pname)
                    elif _pobj.kind == _inspect.Parameter.VAR_POSITIONAL:
                        _pos_params.append(f"*{_pname}")
                        _fwd_pos.append(f"*{_pname}")
                        _has_var_pos = True
                    elif _pobj.kind == _inspect.Parameter.KEYWORD_ONLY:
                        # KEYWORD_ONLY needs a `*` marker before it if there
                        # was no VAR_POSITIONAL. Insert one on first hit.
                        if not _has_var_pos and not _has_kw_only:
                            _pos_params.append("*")
                        _has_kw_only = True
                        if _pobj.default is not _inspect.Parameter.empty:
                            _dflt = f"_dflt_{_pname}"
                            _defaults_ns[_dflt] = _pobj.default
                            _pos_params.append(f"{_pname}={_dflt}")
                        else:
                            _pos_params.append(_pname)
                        _kw_forward.append(f"{_pname}={_pname}")
                    elif _pobj.kind == _inspect.Parameter.VAR_KEYWORD:
                        _pos_params.append(f"**{_pname}")
                        _kw_forward.append(f"**{_pname}")
                _sig_str   = ", ".join(_pos_params)
                _call_args = ", ".join(_fwd_pos + _kw_forward) if (_fwd_pos or _kw_forward) else ""
                _body_lines = [
                    f"def _profiled_forward_step({_sig_str}):",
                    f"    if _profile_done[0]:",
                    f"        return _original_fs({_call_args})",
                    f"    _r = _original_fs({_call_args})",
                    f"    _maybe_advance_profile()",
                    f"    return _r",
                ]
                _wrap_src = "\n".join(_body_lines) + "\n"
                _ns = {
                    "_original_fs":    _original_fs,
                    "_mb_counter":     _mb_counter,
                    "_iter_counter":   _iter_counter,
                    "_micro_per_iter": _micro_per_iter,
                    "_p":              _p,
                    "_profile_done":   _profile_done,
                    "_maybe_advance_profile": _maybe_advance_profile,
                    **_defaults_ns,   # inject preserved defaults as _dflt_<name>
                }
                try:
                    exec(_wrap_src, _ns)
                    _profiled_forward_step = _ns["_profiled_forward_step"]
                    _profiled_forward_step = _ftools.wraps(_original_fs)(_profiled_forward_step)
                    # Preserve original signature explicitly (functools.wraps
                    # doesn't copy __signature__ unless we set it).
                    _profiled_forward_step.__signature__ = _orig_sig
                    _profiled_forward_step.__wrapped__   = _original_fs
                    if rank0:
                        print(f"[profile] Wrapper built with signature "
                              f"{_orig_sig} (matches _original_fs)",
                              flush=True)
                except Exception as _sig_err:
                    if rank0:
                        print(f"[profile] WARN: signature-preserving wrapper "
                              f"build FAILED ({_sig_err}); falling back to "
                              f"varargs wrapper — may re-trigger 55920 crash "
                              f"if Bridge does signature introspection.",
                              flush=True)
                    _orig_sig = None  # trigger fallback

            if _orig_sig is None:
                # Fallback: varargs wrapper (last-resort; known to crash if
                # Bridge binds `state` based on signature — see 55920).
                @_ftools.wraps(_original_fs)
                def _profiled_forward_step(*a, **kw):
                    if _profile_done[0]:
                        return _original_fs(*a, **kw)
                    _r = _original_fs(*a, **kw)
                    _maybe_advance_profile()
                    return _r
                _profiled_forward_step.__wrapped__ = _original_fs

            _p.start()
            try:
                pretrain(config, _profiled_forward_step)
            finally:
                if not _profile_exported[0]:
                    _export_profile_trace()
                if rank0 and _profile_exported[0]:
                    print(f"[profile] forward_step observed "
                          f"{_iter_counter[0]} iter boundaries "
                          f"(schedule expected "
                          f"{_wait + _warmup + _active})", flush=True)
                    print(f"[profile] View with: chrome://tracing OR "
                          f"perfetto.dev (drop file into browser)",
                          flush=True)
        else:
            # Non-recording ranks just run normally
            pretrain(config, forward_step)
    else:
        pretrain(config, forward_step)

    train_seconds = time.time() - t0
    if rank0:
        print(f"[2a-mn:{arm_tag}] training done in {train_seconds:.0f}s")

    # ── End-of-arm HF convert — for run_diff eval ────────────────────────
    skip_hf_convert = run.skip_hf_convert
    if rank0:
        print(f"[2a-mn:{arm_tag}] Megatron ckpt: {results_dir}")
        if skip_hf_convert:
            print(f"[2a-mn:{arm_tag}] CPT_SKIP_HF_CONVERT=1 — skipping Megatron→HF export.")
            print(f"[2a-mn:{arm_tag}] Run phase2_convert_megatron_to_hf.py manually if needed.")
        else:
            print(f"[2a-mn:{arm_tag}] Converting to HF → {final_hf_dir}")
            try:
                from phase2_convert_megatron_to_hf import convert_megatron_to_hf
                convert_megatron_to_hf(
                    megatron_ckpt_dir=results_dir,
                    hf_reference_dir=run.phase1_modified,
                    hf_output_dir=final_hf_dir,
                )
                print(f"[2a-mn:{arm_tag}] HF checkpoint written to {final_hf_dir}")
            except Exception as e:
                print(f"[2a-mn:{arm_tag}] HF conversion FAILED: {e}")
                print(f"[2a-mn:{arm_tag}] Megatron checkpoint at {results_dir} is intact; "
                      f"run phase2_convert_megatron_to_hf.py manually.")

    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
