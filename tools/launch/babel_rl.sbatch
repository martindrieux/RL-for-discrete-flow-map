#!/bin/bash
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o runs/slurm/%x_%j.out   # create runs/slurm first
# usage: sbatch -J name runs/slurm/rl.sbatch <launch script> <args...>; runs the script with its tmux call in the foreground
cd ~/projects/RL-for-discrete-flow-map
export PATH=$HOME/miniconda3/bin:$PATH
tmux() { bash -c "$5"; }
script=$1; shift
source "$script" "$@"
