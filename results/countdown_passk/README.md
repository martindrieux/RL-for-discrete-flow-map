# Countdown: pass@k, distinct solutions and decodings

Output of `scripts/eval_passk.py` (one JSON line per checkpoint and decoding), on 1000 held-out questions
(`task.sample_questions(1000, heldout=True)` with generator seed 4321), 64 sampled answers per question unless `S` says otherwise.

- `pass@k`: unbiased estimate $1 - \binom{S-c}{k} / \binom{S}{k}$ averaged over questions, $c$ correct answers out of $S$.
- `distinct_correct`: distinct correct answer strings per question; `distinct_canonical`: distinct correct expressions after
  flattening and sorting $+$ and $\times$ chains and identifying equal numbers.
- `zs`: the flow map's noise $z \sim \mathcal{N}(0, \mathrm{zs}^2 I)$; `K`: number of jumps; `T`: sampling temperature.
- `greedy`, `beamW`: accuracy of the autoregressive model's greedy answer and of the best of `W` beams.

`ckpt` names the checkpoint: `runs/ckpt_cd_{fm,ar}_{N}k.pt` are pretraining snapshots after N thousand steps; `runs/<run>/last.pt`
are RL runs launched by `tools/launch/launch_countdown.sh` (their experiment numbers are in the tracker, subproject rl-countdown).
`results.jsonl` predates the canonical count; `results2.jsonl` re-evaluates the same checkpoints with it.
