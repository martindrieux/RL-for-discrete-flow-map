import torch

from rlfm.data.arithmetic import ArithmeticTask
from rlfm.rewards.arithmetic import correct, distinct_correct, parse, reward, well_formed


def test_encode_decode_roundtrip():
    task = ArithmeticTask()
    for s in ["07+95=102", "12+34=046", "00+00=000"]:
        ids = task.encode(s)
        assert ids.shape == (9,) and ids.dtype == torch.long
        assert task.decode(ids) == s
    batch = torch.stack([task.encode("07+95=102"), task.encode("12+34=046")])
    assert task.decode(batch) == ["07+95=102", "12+34=046"]


def test_decode_accepts_malformed_ids():
    task = ArithmeticTask()
    assert task.decode(torch.full((9,), 11)) == "========="


def test_sample_shapes_and_vocab():
    task = ArithmeticTask()
    x = task.sample(64)
    assert x.shape == (64, task.seq_len) == (64, 9)
    assert x.dtype == torch.long
    assert x.min() >= 0 and x.max() < task.vocab_size == 12
    assert well_formed(task.decode(x)).all()


def test_fraction_correct():
    for frac in [0.5, 0.2]:
        task = ArithmeticTask(frac_correct=frac)
        g = torch.Generator().manual_seed(0)
        r = reward(task.decode(task.sample(20_000, generator=g)))
        assert r.dtype == torch.float32
        assert abs(r.mean().item() - frac) < 0.02


def test_wrong_sums_never_equal_true_sum():
    task = ArithmeticTask(frac_correct=0.0)
    strings = task.decode(task.sample(20_000, generator=torch.Generator().manual_seed(1)))
    assert not correct(strings).any()
    assert all(0 <= parse(s)[2] < 199 for s in strings)


def test_reward_zero_on_malformed():
    bad = ["12+34=4+6", "1++34=046", "12+34=46", "", "ab+cd=efg", "12+34=0460"]
    assert reward(bad).tolist() == [0.0] * len(bad)
    assert not well_formed(bad).any()
    assert reward(["12+34=046", "07+95=102", "07+95=103"]).tolist() == [1.0, 1.0, 0.0]


def test_distinct_correct():
    assert distinct_correct(["12+34=046", "12+34=046", "07+95=102", "07+95=103", "1++34=046"]) == 2


def test_determinism():
    task = ArithmeticTask()
    x1 = task.sample(128, generator=torch.Generator().manual_seed(42))
    x2 = task.sample(128, generator=torch.Generator().manual_seed(42))
    x3 = task.sample(128, generator=torch.Generator().manual_seed(43))
    assert torch.equal(x1, x2)
    assert not torch.equal(x1, x3)


def test_one_digit():
    task = ArithmeticTask(n_digits=1)
    assert task.seq_len == 6
    assert task.decode(task.encode("7+5=12")) == "7+5=12"
    assert parse("7+5=12") == (7, 5, 12)
    x = task.sample(20_000, generator=torch.Generator().manual_seed(0))
    assert x.shape == (20_000, 6)
    strings = task.decode(x)
    assert well_formed(strings).all()
    assert abs(reward(strings).mean().item() - 0.5) < 0.02


def test_holdout_pairs_never_sampled():
    from rlfm.rewards.arithmetic import parse
    task = ArithmeticTask(n_digits=2, frac_correct=1.0, holdout_frac=0.1, holdout_seed=0)
    assert int(task.heldout_mask.sum()) == 1000
    s = task.decode(task.sample(20000, torch.Generator().manual_seed(0)))
    assert not any(task.is_heldout(*parse(x)[:2]) for x in s)


def test_cli_float_override():
    import tempfile, yaml
    from rlfm.common import load_config
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"train": {"lr": 0.1}}, f)
    assert load_config([f.name, "train.lr=1e-5"])["train"]["lr"] == 1e-5


def test_many_digits_hashed_holdout():
    from rlfm.rewards.arithmetic import parse
    for n in (4, 8):
        task = ArithmeticTask(n_digits=n, frac_correct=1.0, holdout_frac=0.1, holdout_seed=0)
        assert task.seq_len == 3 * n + 3
        s = task.decode(task.sample(4000, torch.Generator().manual_seed(1)))
        assert all(parse(x) is not None and parse(x)[0] + parse(x)[1] == parse(x)[2] for x in s)
        assert not any(task.is_heldout(*parse(x)[:2]) for x in s)
        q = task.sample_questions(500, heldout=True, generator=torch.Generator().manual_seed(2))
        assert all(task.is_heldout(int("".join(map(str, r[:n].tolist()))), int("".join(map(str, r[n + 1:2 * n + 1].tolist()))))
                   for r in q)
        a = torch.randint(0, 10**n, (200_000,)); b = torch.randint(0, 10**n, (200_000,))
        assert abs(task.held_out(a, b).float().mean().item() - 0.1) < 0.005


def test_true_answer():
    task = ArithmeticTask(n_digits=8, holdout_frac=0.1)
    q = task.question_tokens(torch.tensor([99999999, 12345678]), torch.tensor([1, 87654321]))
    assert task.decode(task.true_answer(q)) == ["100000000", "099999999"]


def test_multiplication():
    from rlfm.rewards.arithmetic import correct, parse
    for n in (2, 3, 4):
        task = ArithmeticTask(n_digits=n, frac_correct=1.0, holdout_frac=0.1, op="mul")
        assert task.seq_len == 4 * n + 2 and task.answer_len == 2 * n
        s = task.decode(task.sample(2000, torch.Generator().manual_seed(0)))
        assert "*" in s[0] and correct(s).all()
        assert not any(task.is_heldout(*parse(x)[:2]) for x in s)
        q = task.sample_questions(50)
        full = task.decode(torch.cat([q, task.true_answer(q)], 1))
        assert correct(full).all()
    half = ArithmeticTask(n_digits=2, frac_correct=0.5, holdout_frac=0.1, op="mul")
    assert abs(correct(half.decode(half.sample(20000))).float().mean().item() - 0.5) < 0.02
    assert not correct(["12*34=0400", "12*34=408", "12+34=0408"]).any()
    assert correct(["12*34=0408", "12+34=046"]).all()


def test_cli_text_with_colon_stays_text():
    import tempfile, yaml
    from rlfm.common import load_config
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"track": {}}, f)
    cfg = load_config([f.name, "track.formula=$a$, $b$: $h = 1$", "eval.Ks=[1,2]"])
    assert cfg["track"]["formula"] == "$a$, $b$: $h = 1$" and cfg["eval"]["Ks"] == [1, 2]


def test_digit_fraction_reward():
    from rlfm.rewards.arithmetic import digit_fraction
    task = ArithmeticTask(n_digits=2, holdout_frac=0.1, op="mul")
    q = task.question_tokens(torch.tensor([12, 99]), torch.tensor([34, 99]))
    y = torch.stack([task.encode("0408"), task.encode("9800")])  # 99*99 = 9801
    assert digit_fraction(y, task.true_answer(q)).tolist() == [1.0, 0.75]
