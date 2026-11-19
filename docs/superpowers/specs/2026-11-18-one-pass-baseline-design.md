# One-pass transformer baseline for Sudoku

## Why
The autoregressive transformer writes the 81 cells left to right, which is a poor fit for Sudoku and makes the
comparison with the flow map unfair. A fairer baseline sees the whole puzzle at once and predicts every cell in
one network call, reading the answer off the logits (most likely digit per cell).

## Approaches considered
- **A. Same network as the flow map (chosen by Martin, 2026-11-18).** The flow map's network, fed the puzzle and
  the zero state (clue cells filled), times fixed to s = 0, t = 1, trained by cross-entropy on the solution.
  Same parameter count (7.50M at width 256, depth 6; 22.08M at width 384, depth 8), same input as the flow map's
  first jump from z = 0. The only difference from the flow map is the training loss (and the single call).
  RL and evaluation reuse the flow map code with K = 1 jump from z = 0.
- B. Plain bidirectional transformer encoder over the 81 cells, no time conditioning, size matched by width and
  depth. Closer to a textbook baseline, but changes architecture and input as well as the loss, and needs new
  RL and evaluation code. Possible later check.

## Design
- Loss (`rlfm.flowmap.onepass_loss`): $L = -\frac{1}{81}\sum_i x_1^i \cdot \log \mathrm{softmax}(f_\theta(0, 0, 1 \mid c)_i)$,
  clue cells fixed (`cond.clamp=true`), so only blank cells carry loss.
- `scripts/pretrain.py train.objective=onepass`; in-training evaluation with `eval.Ks=[1] eval.zs=0`.
- Launcher: `ONEPASS=1 tools/launch/launch_sudoku.sh fm pretrain|rl|pre ...` (RL with K = 1); runs named `sd_onepass_*`;
  tracker algorithm `onepass-ce` (pseudocode in `pseudocode/onepass-ce.txt`).
- Same budget as the flow map: 100k steps, batch 256, lr 3e-4, same puzzles (22-51 clues), 1,000 held-out grids.
- Evaluation: the same 2,000 held-out puzzles, whole grid exactly right, answer = most likely digit per cell.
- RL: same as the flow map runs (#215-#217): REINFORCE with leave-one-out advantages, 16 answers per puzzle with
  each cell drawn from its logits, 16 puzzles per step, puzzles with >= 45 blanks, lr 3e-5, 5,000 steps, from the
  50k-step checkpoint.

## Left to the builder
Seeds beyond seed 0, continued-pretraining reference for the one-pass model, approach B.
