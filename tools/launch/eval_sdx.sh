#!/bin/bash
# SLURM job on Babel: Sudoku-Extreme evaluation of checkpoints of one run on the 1,000 validation puzzles (to choose the
# checkpoint and number of jumps) and on the 1,000 test puzzles; answer from z = 0 with K jumps, T = 0.1, 2 samples.
# usage: sbatch -J <name> -o <log> --export=ALL,KS=32_8_4_1 tools/launch/eval_sdx.sh <run> <checkpoint file>...
# results: one JSON line per (checkpoint, split, K) in results/sudoku_extreme/<run>.jsonl
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --requeue
cd ~/projects/RL-for-discrete-flow-map
R=$1; shift
mkdir -p results/sudoku_extreme
nvidia-smi --query-gpu=name --format=csv,noheader
for c in "$@"; do for split in heldout test; do for k in ${KS//_/ }; do
  .venv/bin/python scripts/eval_passk.py fm runs/$R/$c K=$k zs=0 T=0.1 n=1000 S=2 split=$split >> results/sudoku_extreme/$R.jsonl
done; done; done
