#!/bin/bash
# SLURM job on Babel for tools/launch/launch_sudoku.sh. Requeue-safe: pretraining resumes from RUNDIR/last.pt
# (same folder, step numbers continue); rl and sft resume from runs/<run>/state.pt (saved at every evaluation).
# usage: sbatch -J <name> -o <log> [--gres=gpu:L40S:1] --export=ALL,RUNDIR=runs/<run>,CLAMP=1,... tools/launch/babel_job.sh fm pretrain 4 0
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
cd ~/projects/RL-for-discrete-flow-map
if [ -n "$RUNDIR" ] && [ -f "$RUNDIR/last.pt" ] && [ "$2" = pretrain ]; then
  STEP=$(.venv/bin/python -c "import torch, sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step'])" "$RUNDIR/last.pt")
  [ -f "$RUNDIR/ckpt_$STEP.pt" ] || cp "$RUNDIR/last.pt" "$RUNDIR/ckpt_$STEP.pt"
  export RESUME="$RUNDIR/ckpt_$STEP.pt"
fi
[ "$2" = rl ] || [ "$2" = sft ] && export RLRESUME=1  # RL and fine-tuning go on from runs/<run>/state.pt after a requeue
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
FG=1 bash tools/launch/launch_sudoku.sh "$@"
