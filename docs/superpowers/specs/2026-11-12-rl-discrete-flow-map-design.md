# RL for discrete flow maps — design

Date: 2026-11-12. Repository: `RL-for-discrete-flow-map` (empty at the time of writing; code starts from scratch).
Method note: Overleaf project cloned at `~/Documents/overleaf-flowmaps/main.tex` ("RL for Continuous Flow Language Models").
Reference method: Lee et al. 2026, *Flow Map Language Models* (arXiv 2602.16813, code `github.com/david3684/flm`).

Items marked **[my call]** were chosen by Claude, not by Martin; everything else was decided by Martin.

## 1. Intent

Show, at small scale, that a flow map over one-hot token states can be fine-tuned with RL using the
**exact categorical likelihood of the sampled tokens given the noise**, `log π_θ(y | z)`, and compare RL
algorithms on that signal. Two stages:

1. **Pretrain** a flow map from data with the FMLM loss, trained directly (no teacher, no distillation)
   and without FMLM's extra training tricks.
2. **RL**: define rewards, turn the generator into a stochastic policy through categorical decoding of its
   final output, and try several RL algorithms on it.

## 2. Goals, non-goals, anti-goals

Goals
- A clean, small codebase that pretrains a flow map on small discrete datasets and runs RL on it.
- At least three RL algorithms sharing one rollout code path (Section 5).
- Policy **option A** first (Section 4); **option B** is the planned second step and the code should not
  make it hard to add.
- Every run recorded on the experiment tracker (Section 8).

Non-goals (for this first version)
- Large vocabularies (BERT / GPT-2 tokenisers), OWT/LM1B scale, released FMLM checkpoints.
- Prompt-conditioned generation (Sudoku, question answering). All tasks are unconditional: the prompt `c`
  of the note is empty.
- FMLM extras other than the time reparameterisation: boundary sampling of `(s,t)=(0,1)`,
  self-conditioning, posterior refinement, distillation from an FLM teacher, autoguidance.
  (The time reparameterisation `τ(t)` **is used**: Section 3.)

Anti-goals (would make the result worthless even if it "works")
- A reward gain that comes only from collapse (one sample repeated) or reward hacking. Diversity and the
  KL to the pretrained model are always reported next to the reward.
- Comparisons between runs that differ in more than one setting.

## 3. Stage 1: pretraining (FMLM, direct training)

Notation: sequence length `L`, vocabulary size `V`, data `x_1 ∈ {0,1}^{L×V}` (one-hot rows),
noise `x_0 ~ N(0, I_{L×V})`, interpolant `I_t = (1−t) x_0 + t x_1`, `t ∈ [0,1]` (0 = noise, 1 = data).

**Network.** One network outputs the *two-time denoiser* `δ̂_{s,t}(x) ∈ Δ^{V−1}` for each of the `L`
positions, `0 ≤ s ≤ t ≤ 1`:
- input projection: `x ∈ R^{L×V} ↦ x E`, `E ∈ R^{V×d}` (a linear map of the continuous state, no lookup);
- a DiT-style transformer (bidirectional attention, RoPE), conditioned on `(s, t)` through sinusoidal
  embeddings of the reparameterised times `τ(s)` and `τ(t)` (uniform in `[0,1]` under the training law) **[my call]**, summed after an MLP, fed to AdaLN in every block;
- output head: linear `d → V`, then a softmax per position. The pre-softmax values are the **logits**
  `ℓ_{s,t}(x) ∈ R^{L×V}`; `δ̂_{s,t}(x) = softmax(ℓ_{s,t}(x))`.

**Flow map** from the denoiser (Lee et al., Prop. 3.3 (i)):
`X̂_{s,t}(x) = ((1−t)/(1−s)) x + ((t−s)/(1−s)) δ̂_{s,t}(x)`, defined for `s < 1`.
Note `X̂_{s,1}(x) = δ̂_{s,1}(x)`: the final state is the softmax output itself.

**Time reparameterisation** (Lee et al., Eq. 25; used for training and sampling). Times are sampled and
gridded uniformly in `τ ∈ [0,1]` and mapped to `t = t(τ)`, the inverse of
`τ(t) = 1 − (V/(V−1)) P_e(t)`, where `P_e(t)` is the probability that `argmax` of a row of `I_t` differs
from the data token. For this interpolant `P_e` does not depend on the data and has a closed form
**[derived by Claude; the paper does not state how it computes `P_e`]**: with `a = t/(1−t)`,
`1 − P_e(t) = ∫ φ(ξ) Φ(ξ + a)^{V−1} dξ` (`φ`, `Φ` the standard normal density and CDF).
Compute it once per `V` by 1-D quadrature on a fine grid of `t` and invert by monotone interpolation.
For scale, the `t` values at `τ = 0.1, 0.5, 0.9` are `0.34, 0.63, 0.76` for `V = 12` and
`0.43, 0.67, 0.78` for `V = 27`: even at small `V` most of the decoding happens in a band of `t`.

**Loss** `L = L_diag + L_off`, each term on its own half of the batch **[my call]**, averaged over positions.
- Diagonal: `τ ~ U[0,1]`, `t = t(τ)`, `L_diag = −(1/L) Σ_l x_1^l · log δ̂_{t,t}(I_t)^l`.
- Off-diagonal (semigroup, self-distilled from the current weights), times drawn in `τ`:
  `h ~ U[0,1]`, `τ_s ~ U[0, 1−h]`, `τ_t = τ_s + h`, `τ_u = (τ_s + τ_t)/2`, then `s, u, t = t(τ_s), t(τ_u), t(τ_t)`,
  `γ = (1−t)(u−s) / ((1−u)(t−s))`,
  `δ̄ = sg( γ δ̂_{s,u}(I_s) + (1−γ) δ̂_{u,t}(X̂_{s,u}(I_s)) )`,
  `L_off = −(1/L) Σ_l δ̄^l · log δ̂_{s,t}(I_s)^l`.
  `sg` is stop-gradient; the target uses the current weights, not an EMA **[my call]**.
  Guard `t − s ≥ 1e−4` and `s ≤ 1 − 1e−4` numerically.
- Optimiser AdamW, linear warm-up then constant learning rate, EMA of the weights (decay 0.999 for these
  short runs) used for evaluation and as the RL initialisation **[my call]**.

**Sampling** with `K` jumps on the grid `t_k = t(k/K)` (uniform in `τ`): `x_0 ~ N(0, I)`,
`x_{k+1} = X̂_{t_k, t_{k+1}}(x_k)`; decode `y_i = argmax_v x_K[i, v]`. Evaluate at `K ∈ {1, 2, 4, 8}`.

## 4. Stage 2: the policy

The whole sequence is one action; the noise `z = x_0` is part of the sample.

**Option A (first).** Run the first `K−1` jumps as in sampling; the last jump `t_{K−1} → 1` gives logits
`ℓ = ℓ_{t_{K−1},1}(x_{K−1})`. With sampling temperature `T > 0` (named `T`, not `τ`, which is the reparameterised time):
`p_{θ,i}(v | z) = softmax(ℓ_i / T)_v`, `y_i ~ Cat(p_{θ,i}(· | z))` independently,
`log π_θ(y | z) = Σ_i log p_{θ,i}(y_i | z)`.
Gradients flow through **all** `K` jumps (`x_{K−1}` depends on `θ`).

Known risk: for an accurate flow map `δ̂_{s,1}` is close to one-hot, so `p_θ` has little entropy and the
score `∇ log π` becomes small. Track the policy entropy `(1/L) Σ_i H(p_{θ,i})` in every run; `T` is a swept
setting.

**Option B (second).** Stop at `t* = t(τ*) < 1`, `τ* ∈ {0.8, 0.9, 0.95}` swept **[my call]**: run jumps on
the grid `t_k = t(k τ*/K)`, `k = 0…K` (uniform in `τ` on `[0, τ*]`), then
`p_{θ,i}(v | z) = softmax(ℓ_{t*,t*}(x_K)_i / T)_v` (the diagonal head, i.e. the token posterior).
Same sampling and same RL code; only the logits source changes. Design the rollout so the logits source is
one switch.

## 5. Stage 2: RL algorithms

Shared rollout: for each update draw `B` groups of `G ≥ 2` samples `(z_g, y_g)` from the current policy,
compute rewards `r_g = R(y_g)` (black box, no gradient), store `z_g`, `y_g` and the **per-token** old
log-probabilities `log p_{old,i}(y_{g,i} | z_g)` (needed by the GRPO ratio).
Because `z_g` is stored, `log π_θ(y_g | z_g)` can be recomputed exactly for any `θ`.

**Making the algorithms comparable [my call].** Every loss below is normalised the same way, as a mean
over the `G·L` tokens of a group (so REINFORCE-LOO is written with `1/(G L)` instead of the note's `1/G`;
this rescales the gradient by the constant `1/L`, which only shifts the learning rate). The KL term is
normalised the same way, so `β` means the same thing in every algorithm. The learning rate is tuned per
algorithm with a small sweep on the toy task and then frozen. GRPO differs from REINFORCE-LOO in three
pieces, which are added one at a time as an ablation ladder:
REINFORCE-LOO → `+` std-normalised group advantage → `+` `μ > 1` steps with clipping (= GRPO).

1. **REINFORCE with leave-one-out baseline** (the Overleaf note):
   `A_g = r_g − (1/(G−1)) Σ_{j≠g} r_j`,
   `L(θ) = −(1/(G L)) Σ_g sg(A_g) Σ_i log p_{θ,i}(y_{g,i} | z_g)`, one gradient step per batch.
2. **GRPO / PPO-clip**: `A_g = (r_g − mean_j r_j) / (std_j r_j + ε)`, per-token ratio
   `ρ_{g,i} = p_{θ,i}(y_{g,i} | z_g) / p_{old,i}(y_{g,i} | z_g)`,
   `L(θ) = −(1/(G L)) Σ_{g,i} min(ρ_{g,i} A_g, clip(ρ_{g,i}, 1−ε_c, 1+ε_c) A_g)`, `μ` gradient steps per batch.
3. **KL-regularised** version of either: add `β (1/(G L)) Σ_g Σ_i KL(p_{θ,i}(· | z_g) ‖ p_{ref,i}(· | z_g))`,
   computed exactly over the vocabulary, where `p_ref` is the frozen pretrained policy run from the same
   `z_g` along its own trajectory. This is `E_z KL(π_θ(·|z) ‖ π_ref(·|z))`, which upper-bounds the KL between
   the two marginal distributions of `y`.

Baselines every RL result is compared with: the pretrained model with no RL (same `K`, `T`), and
best-of-`G` selection from it: for each reported sample, draw `G` samples from the pretrained policy and
keep the one with the highest reward, `G` being the RL group size. This measures what selection alone buys
at the same per-sample reward cost **[my call]**.

## 6. Datasets and rewards (in this order)

All unconditional, fixed length `L` with a pad token where needed; the pad token is an ordinary token in
the loss **[my call]**.

1. **Toy arithmetic** (debugging; known optimum) **[my call for the exact construction]**:
   strings `ab+cd=efg` (two 2-digit numbers, sum written with 3 digits), `L = 9`, `V = 12`.
   Pretraining data: 50% correct equations, 50% with a uniformly random wrong sum.
   Reward: `1` if the decoded string is a well-formed, correct equation, else `0`. Expected pretrained
   reward ≈ 0.5; RL should push it towards 1 without collapsing to a few equations.
2. **QM9 SMILES** (main proof of concept): ~134k molecules, character-level tokens, `L` = the maximum
   canonical SMILES length in the data (≈ 32–40) with padding, `V` ≈ 30–40.
   Reward: `QED(mol)` if RDKit parses the string, else `0`. Report validity, uniqueness, novelty (vs the
   training set) next to the reward. A second reward (ring count, as in Schiff et al. 2024, *Simple Guidance
   Mechanisms for Discrete Diffusion Models*) may be added later.
3. **text8** (second proof of concept): character level, `V = 27`, random crops of `L = 64`.
   Reward: fraction of space-separated words found in a fixed dictionary (text8 words of frequency ≥ 5).
   This reward is easy to hack (repeating "the"), so it is the test of whether the KL term matters.

## 7. Model sizes and evaluation **[my call]**

| dataset | width `d` | depth | heads | approx. params |
|---|---|---|---|---|
| toy | 128 | 4 | 4 | ~1M |
| QM9 | 256 | 6 | 8 | ~5M |
| text8 | 384 | 8 | 6 | ~15M |

Pretraining is judged by: diagonal cross-entropy on held-out data; sample metrics at `K ∈ {1,2,4,8}`
(toy: fraction of correct equations; QM9: validity / uniqueness / novelty / mean QED; text8: dictionary-word
fraction and unigram entropy).
RL is judged by: mean reward vs RL step, policy entropy, KL to `π_ref`, and the fraction of unique samples
among 1000 draws; reported under categorical decoding at temperature `T` **and** under argmax decoding (the note warns
that argmax is not guaranteed to improve). Fixed evaluation seed and sample count across runs.

## 8. Compute, storage and tracking

Machines (SSH aliases in `~/.ssh/config`): `bentley`, `bugatti` (RTX 4000 Ada, 20 GB), `coucou`, `dindon`,
`bengali` (RTX A5000, 24 GB). One GPU each, 16–28 CPUs, 62 GB RAM. They are **shared with other users**
(at the time of writing four of the five GPUs were busy): check `nvidia-smi` before launching and run at
most one job per free GPU.

Storage: the home directory is one NFS volume shared by all five machines, with a quota of **30 GB and
300k files** (13 GB and 109k files already used). Therefore:
- code, the Python environment, datasets (all small), and checkpoints live in the shared home;
  keep checkpoints to `last` + `best` per run;
- anything large or temporary (caches, sample dumps) goes on each machine's local `/tmp` (≈ 95 GB free,
  not shared);
- one environment in the project directory created with `uv` (installed at `~/.local/bin/uv`), with
  `UV_CACHE_DIR` on local `/tmp` so the cache does not count against the quota.

Launching: from the laptop over SSH, inside `tmux` on the remote machine, one run per GPU.

Tracking: every launch goes through `nb.start(...)` (experiment tracker, lab.loopflowlabs.com), with
`@nb.core` on the loss, sampler, rollout and RL-update functions and pseudocode files for each algorithm.
Suggested subprojects: `pretrain-toy`, `pretrain-qm9`, `rl-toy`, `rl-qm9`, `rl-text8`.

## 9. Code layout and tests **[my call]**

```
rlfm/model.py        DiT with (s,t) conditioning, linear input projection, softmax head
rlfm/flowmap.py      X̂ from δ̂, diagonal and semigroup losses, K-step sampler
rlfm/policy.py       rollout (option A, option B switch), log π(y|z), exact per-token KL
rlfm/rl.py           REINFORCE-LOO, GRPO/PPO-clip, KL term
rlfm/data/           toy arithmetic, QM9 SMILES, text8
rlfm/rewards/        arithmetic correctness, RDKit QED/validity, dictionary-word fraction
scripts/pretrain.py, scripts/rl.py, scripts/eval.py
configs/*.yaml
tests/
```

Tests: `τ(0) = 0`, `τ(1) = 1`, `τ` increasing, and the closed-form `P_e(t)` matches a Monte Carlo estimate;
`X̂_{s,s}(x) = x` and `X̂_{s,1}(x) = δ̂_{s,1}(x)`; `γ ∈ [0,1]`; the semigroup target equals
`δ̂_{s,t}` for an analytically consistent toy denoiser; `log π(y|z)` matches a brute-force product; the
REINFORCE-LOO gradient matches the exact gradient on a tiny bandit (small `L`, `V`, enumerable `y`);
a few pretraining and RL steps run on CPU.

## 10. Left to the builder

Exact learning rates, batch sizes, number of steps, `G`, `μ`, `ε_c`, `β`, `T` grid, and the QM9 data source
(any standard copy of the QM9 SMILES, e.g. MoleculeNet), as long as the choices are recorded in configs
and on the tracker.

## 11. First experiment grid (arithmetic only; agreed 2026-11-12)

Defaults set by Claude for the questions the builder check raised; Martin can override.
- RL uses `K = 1` (one-step generation) first; the best configuration is then rerun at `K = 4`.
- 3 seeds per RL configuration; 1 seed for the learning-rate sweep.
- Gate before RL: at `K = 1`, ≥ 95% of samples are well-formed equations and ≈ 50% are correct.
  If it fails, train longer before re-enabling any FMLM extra.
- If option A's entropy collapses even at `T = 2`, move to option B (no entropy bonus).

| step | run | change vs previous row |
|---|---|---|
| 0 | pretrain flow map | — (evaluate at `K ∈ {1,2,4,8}`) |
| 1a | REINFORCE-LOO, `T = 1`, lr ∈ 3 values | learning-rate sweep, picks one lr for all rows |
| 1.1 | REINFORCE-LOO, `T = 1` | RL baseline |
| 1.2 | REINFORCE-LOO, `T = 2` | temperature |
| 1.3 | + std-normalised advantage | advantage scaling |
| 1.4 | GRPO: + `μ = 4` steps with clipping `ε_c = 0.2` | sample reuse |
| 1.5 | REINFORCE-LOO + KL, `β = 0.1` | KL to the pretrained policy |
| 2 | best of 1.1–1.5 at `K = 4` | number of jumps |

Reference points (no training): the pretrained policy, and best-of-`G` selection from it.

## 12. Held-out sets (added 2026-11-12, required)

Every dataset keeps a held-out split never used for pretraining or RL updates, and every evaluation
reports its metrics on it separately. Arithmetic: 10% of operand pairs `(a, b)` (1000 of 10^4),
chosen by `task.holdout_seed`, never appear in pretraining with any sum; `heldout_correct` is the
fraction of correct equations among samples whose operands are held out. The scripts refuse to run
without a held-out split unless `allow_no_holdout=true` (debugging only, not reportable).
