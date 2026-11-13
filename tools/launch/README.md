# Launch scripts

Each `launch_*.sh` starts one RL run in a detached tmux session on a lab machine, from the repository root on the
shared home:

    ssh <host> "bash -s <args>" < tools/launch/launch_chain.sh          # args: variant (det|stoch|rwr) jumps
    ssh <host> "bash -s <args>" < tools/launch/launch_chain_T.sh        # args: variant jumps temperature

`launch_rlmul*.sh` (args: jumps) are the 4-digit multiplication RL runs from the 100k checkpoint with the exact-match
reward, the per-digit reward and one advantage per digit. `launch_chain*.sh` start from the 200k checkpoint
(`runs/ckpt_mul4_200k.pt`).

On the Babel SLURM cluster, `babel_rl.sbatch` runs any of these scripts in the foreground (it replaces the tmux call):

    sbatch -J <name> tools/launch/babel_rl.sbatch tools/launch/launch_chain_T.sh stoch 4 2
