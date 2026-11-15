import torch

from rlfm.data.countdown import CountdownTask, evaluate
from rlfm.rewards import countdown as R


def test_evaluate_postfix():
    assert evaluate("AB+C*D-", [1, 2, 3, 4]) == 5          # ((1+2)*3)-4
    assert evaluate("ABCD+++", [1, 2, 3, 4]) == 10
    assert evaluate("AB/CD++", [7, 2, 3, 4]) is None       # 7/2 is not exact
    assert evaluate("AA+C*D-", [1, 2, 3, 4]) is None       # A twice, B missing
    assert evaluate("A+BC*D-", [1, 2, 3, 4]) is None       # operator without two operands


def test_generated_answers_are_correct_and_heldout_is_disjoint():
    task = CountdownTask(holdout_frac=0.1)
    g = torch.Generator().manual_seed(0)
    ids = task.sample(200, generator=g)
    s = task.decode(ids)
    assert ids.shape == (200, 18) and R.correct(s).all()
    train_sets = {tuple(sorted(R.parse(x)[0])) for x in s}
    held = task.decode(task.sample_questions(200, heldout=True, generator=g))
    held_sets = {tuple(sorted(int(q[i:i + 2]) for i in range(0, 8, 2))) for q in held}
    assert not train_sets & held_sets
    assert all(task._held(list(h)) for h in held_sets)


def test_reward_accepts_any_valid_solution():
    task = CountdownTask()
    q = "01020304" + "010"                                   # target 10 from 1, 2, 3, 4
    assert R.reward([q + "ABCD+++", q + "DC+B+A+", q + "AB+C+D+"]).tolist() == [1.0, 1.0, 1.0]
    assert R.reward([q + "AB+C*D-"]).tolist() == [0.0]       # = 5
    assert R.well_formed([q + "AB+C*D-", q + "AAAA+++"]).tolist() == [True, False]
