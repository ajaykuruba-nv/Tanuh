# 16-node / 128-GPU GBS2048 run

This run uses TP1/EP8/PP1, MBS1, GBS2048, sequence length 8192, and 100
iterations in the NeMo 26.06 container.

Pinned nodes:

```text
slinky-[0,5,7-9,12,14-15,18,20,22-25,29,32]
```

The requested nodes `slinky-[1-4,6,10,13,27,28,31]` are absent. Slinky-30 is
also absent because NCCL job 56725 showed a different HCA enumeration. Eleven
selected nodes passed that NCCL run. Slinky-0 and slinky-18 sustained
successful distributed training runs. Slinky-[7,14,22] are unflagged
candidates but have not appeared in a completed NCCL test, so validate them
before using this placement as a clean infrastructure baseline.

Submit:

```bash
cd /mnt/pfs1/avinash/pre-training/ajay/GPU16_Run
sbatch phase2_pretrain_cpt_mn_16n_tp1ep8_gbs2048_selected_nodes.sh
```
