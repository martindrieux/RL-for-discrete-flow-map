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
