import pytest
import torch

import rlfm.data.sudoku as S
from rlfm.rewards import sudoku as R

# A puzzle with a unique solution (Wikipedia's example)
PUZZLE = ("530070000600195000098000060800060003400803001700020006060000280000419005000080079")
SOLUTION = ("534678912672195348198342567859761423426853791713924856961537284287419635345286179")


def test_solver_counts_solutions():
    n, sol = S.solve([int(c) for c in PUZZLE])
    assert n == 1 and "".join(map(str, sol)) == SOLUTION
    assert S.solve([0] * 81, limit=2)[0] == 2
    bad = [int(c) for c in PUZZLE]
    bad[2] = 5  # two 5s in the first row
    assert S.solve(bad)[0] == 0


def test_generated_grid_is_valid_and_every_subset_of_R_is_unique():
    sol, rem = S.make_grid(3)
    assert S.is_solution(sol)
    assert S.solve([0 if r else v for v, r in zip(sol, rem)])[0] == 1
    assert 40 <= sum(rem) <= 64


@pytest.fixture(scope="module")
def task(tmp_path_factory):
    S.POOL_DIR = str(tmp_path_factory.mktemp("sudoku"))
    S.build_pool("train", 12, workers=2)
    S.build_pool("heldout", 6, workers=2)
    return S.SudokuTask(n_train=12, n_heldout=6, min_blanks=30)


def test_samples_are_unique_puzzles_of_their_solutions(task):
    g = torch.Generator().manual_seed(0)
    ids = task.sample(16, generator=g)
    assert ids.shape == (16, 162)
    s = task.decode(ids)
    assert R.correct(s).all()
    q = ids[:, :81]
    assert ((q == 0).sum(1) >= 30).all()
    for row in q[:4].tolist():
        assert S.solve(row)[0] == 1


def test_heldout_puzzles_come_from_other_grids(task):
    g = torch.Generator().manual_seed(1)
    held = task.sample_questions(6, heldout=True, generator=g)
    assert held.shape == (6, 81)
    train_sols = {tuple(S.solve(r)[1]) for r in task.sample(40, generator=g)[:, :81].tolist()}
    held_sols = {tuple(S.solve(r)[1]) for r in held.tolist()}
    # symmetries make grids differ anyway; the check is that no held-out solution equals a training one
    assert not train_sols & held_sols


def test_reward_checks_rules_and_givens():
    ok = PUZZLE + SOLUTION
    swapped = PUZZLE + SOLUTION[:2] + SOLUTION[3] + SOLUTION[2] + SOLUTION[4:]  # two blank cells of row 1: columns break
    other_givens = "1" + PUZZLE[1:] + SOLUTION                                  # answer disagrees with a given
    assert R.correct([ok, swapped, other_givens]).tolist() == [True, False, False]
    assert R.well_formed([ok, other_givens, PUZZLE + "0" + SOLUTION[1:]]).tolist() == [True, False, False]
    assert R.reward([ok]).tolist() == [1.0]


def test_training_task_keeps_hard_puzzles_only(task):
    """train_min_blanks = 45: training draws have at least 45 blanks; the evaluation task is unchanged."""
    from rlfm.common import training_task
    hard = training_task(task, {"train_min_blanks": 45})
    assert training_task(task, {}) is task and task.min_blanks == 30 and hard.min_blanks == 45
    q = hard.sample_questions(200, heldout=False, generator=torch.Generator().manual_seed(0))
    assert int((q == 0).sum(1).min()) >= 45
    q_all = task.sample_questions(200, heldout=False, generator=torch.Generator().manual_seed(0))
    assert int((q_all == 0).sum(1).min()) < 45
