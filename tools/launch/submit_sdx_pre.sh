#!/bin/bash
# Babel: Sudoku-Extreme reference for RL. Per seed: continued pretraining from the 200k checkpoint, 5,000 steps,
# lr 3e-5 (same protocol as the Sudoku references), then its evaluation (K = 32, 16, 8, 4, 1; validation and test).
# Seed 0 waits for its pretraining job ($1).
cd ~/projects/RL-for-discrete-flow-map
for s in 0 1 2; do
  P=sdx_fm_pretrain_d256x6_clamp_s$s
  n=sdx_fm_pre_s${s}_n5000_pre200k-s$s
  DEP=""; [ $s = 0 ] && DEP="--dependency=afterok:$1"
  c=$(sbatch --parsable $DEP -J $n -o runs/_slurm/${n}_%j.out --export=ALL,EXTREME=1,CLAMP=1,CKPT=runs/$P/ckpt_200000.pt,PRE=200k-s$s tools/launch/babel_job.sh fm pre 4 $s 5000)
  f=$(sbatch --parsable --dependency=afterok:$c -J ev_$n -o runs/_slurm/ev_${n}_%j.out --export=ALL,KS=32_16_8_4_1 tools/launch/eval_sdx.sh $n last.pt)
  echo "seed $s: continued pretraining $c, its eval $f"
done
