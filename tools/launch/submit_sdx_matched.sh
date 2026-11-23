#!/bin/bash
# Babel: Sudoku-Extreme, pretraining reference at the compute of 5,000 RL steps (one RL step = 8 pretraining steps of
# compute: 256 answers x 4 calls, sampled then recomputed with gradient, vs batch 128 with 2 target calls on half).
# Per seed, from the validation-best pretraining checkpoint (seed 0: 160k; seeds 1, 2: 200k): 40,000 more pretraining
# steps with the pretraining recipe (lr 3e-4, weight average 0.9999), checkpoints every 10,000 steps, each evaluated
# (K = 32, 16, 8, 4, 1; validation and test). Seed 0 also gets RL from 160k, same settings as its RL from 200k.
cd ~/projects/RL-for-discrete-flow-map
for s in 0 1 2; do
  P=sdx_fm_pretrain_d256x6_clamp_s$s
  st=200k; [ $s = 0 ] && st=160k
  n=sdx_fm_pre_lr3e-4_s${s}_n40000_pre$st-s$s
  c=$(sbatch --parsable -J $n -o runs/_slurm/${n}_%j.out --export=ALL,EXTREME=1,CLAMP=1,LR=3e-4,KEEP=10000,CKPT=runs/$P/ckpt_${st%k}000.pt,PRE=$st-s$s tools/launch/babel_job.sh fm pre 4 $s 40000)
  f=$(sbatch --parsable --dependency=afterok:$c -J ev_$n -o runs/_slurm/ev_${n}_%j.out --export=ALL,KS=32_16_8_4_1 tools/launch/eval_sdx.sh $n ckpt_10000.pt ckpt_20000.pt ckpt_30000.pt ckpt_40000.pt)
  echo "seed $s: pretraining $c ($n), its eval $f"
done
n=sdx_fm_rl_K4_zs0_G16_s0_n5000_pre160k-s0
c=$(sbatch --parsable -J $n -o runs/_slurm/${n}_%j.out --export=ALL,EXTREME=1,ZS=0,GS=16,GR=16,MICRO=2,CKPT=runs/sdx_fm_pretrain_d256x6_clamp_s0/ckpt_160000.pt,PRE=160k-s0 tools/launch/babel_job.sh fm rl 4 0 5000)
f=$(sbatch --parsable --dependency=afterok:$c -J ev_$n -o runs/_slurm/ev_${n}_%j.out --export=ALL,KS=32_16_8_4_1 tools/launch/eval_sdx.sh $n last.pt)
echo "seed 0: RL from 160k $c, its eval $f"
