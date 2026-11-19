import csv

import numpy as np
import pytest
import torch

import rlfm.data.sudoku as S
import rlfm.data.sudoku_extreme as X
from rlfm.rewards import sudoku as R


def _write_csv(path, n, seed):
    """n puzzles with unique solutions in the released format: source, question ('.' = blank), answer, rating."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "question", "answer", "rating"])
        for k in range(n):
            sol, rem = S.make_grid(seed + k)
            w.writerow(["synthetic", "".join("." if r else str(v) for v, r in zip(sol, rem)), "".join(map(str, sol)), k % 37])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("extreme")
    _write_csv(d / "train.csv", 70, 0)
    _write_csv(d / "test.csv", 25, 10_000)
    S.EXTREME_PATH = str(d / "extreme.npz")
    X.build(str(d / "train.csv"), str(d / "test.csv"), out=S.EXTREME_PATH, n=20, n_subsets=2)
    return np.load(S.EXTREME_PATH)


def test_stratified_takes_the_same_number_from_every_difficulty_part():
    rng = np.random.default_rng(0)
    ratings = np.random.default_rng(1).permutation(1000)
    exclude = np.flatnonzero(ratings < 50)[:20]
    idx = X.stratified(ratings, 100, rng, exclude)
    assert len(set(idx.tolist())) == 100 and not set(idx.tolist()) & set(exclude.tolist())
    assert (np.bincount(ratings[idx] // 100, minlength=10) == 10).all()


def test_build_gives_disjoint_valid_splits(built):
    assert built["train_puzzle"].shape == (2, 20, 81) and built["val_puzzle"].shape == (20, 81)
    assert built["test_puzzle"].shape == (20, 81)
    assert not set(built["val_index"].tolist()) & set(built["train_index"].ravel().tolist())
    for q, s in ((built["val_puzzle"], built["val_solution"]), (built["test_puzzle"], built["test_solution"]),
                 (built["train_puzzle"][1], built["train_solution"][1])):
        strings = ["".join(map(str, a.tolist())) + "".join(map(str, b.tolist())) for a, b in zip(q, s)]
        assert R.correct(strings).all()


def test_task_relabels_training_puzzles_and_keeps_heldout_ones_as_released(built):
    task = S.SudokuTask(n_heldout=20, source="extreme", subset=1)
    ids = task.sample(32, generator=torch.Generator().manual_seed(0))
    assert R.correct(task.decode(ids)).all()
    clues = {int((row != 0).sum()) for row in built["train_puzzle"][1]}
    assert {int((q != 0).sum()) for q in ids[:, :81]} <= clues   # a symmetry keeps the number of clues
    as_set = lambda a: {tuple(r) for r in np.asarray(a).tolist()}
    held = task.sample_questions(20, heldout=True, generator=torch.Generator().manual_seed(1))
    assert as_set(held) == as_set(built["val_puzzle"])           # every validation puzzle once, unchanged
    task.eval_split = "test"
    assert as_set(task.sample_questions(20, heldout=True)) == as_set(built["test_puzzle"])
