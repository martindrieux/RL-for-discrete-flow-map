<!-- nb:start -->
## Experiments

Experiments in this repo are recorded with the `nb` experiment tracker (see `nb.json`).
Before launching, changing or reporting any experiment, and at the start of each
session, read the rules: use the `experiment-tracker` skill if your tool has skills,
otherwise run `nb guide` and follow what it prints. Explain with short formulas
rather than words wherever a formula exists. `nb doctor` lists what is still missing.
<!-- nb:end -->

## Held-out set (required)

Every dataset in this repo keeps a held-out set that is never used for training, and every
evaluation reports metrics on it separately (e.g. `heldout_correct`). For arithmetic this is
`task.holdout_frac` (default 0.1 in `configs/arith_pretrain.yaml`): a fraction of operand pairs
(a, b) never shown in pretraining, with any sum. `scripts/pretrain.py` and `scripts/rl.py` refuse
to run without one; `allow_no_holdout=true` exists only for debugging and such runs must not be
reported as results. New datasets (QM9, text8, ...) must define their held-out split before their
first training run.
