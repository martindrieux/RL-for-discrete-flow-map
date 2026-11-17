import itertools

import torch

from rlfm.flowmap import FlowMap, jump, pretrain_loss, sample
from rlfm.model import DiT
from rlfm.policy import policy_logits, token_logp
from rlfm.rl import RLConfig, Rollout, advantages, pg_loss, rl_update
from rlfm.timewarp import TimeWarp, prob_correct


def tiny(V=12, L=9):
    torch.manual_seed(0)
    return FlowMap(DiT(V, L, d=32, depth=1, heads=2), TimeWarp(V))


def test_timewarp_endpoints_monotone_inverse():
    w = TimeWarp(12)
    t = torch.linspace(0, 1, 101)
    tau = w.tau(t)
    assert abs(tau[0].item()) < 1e-6 and abs(tau[-1].item() - 1) < 1e-6
    assert (tau[1:] >= tau[:-1]).all()
    ti = torch.linspace(0.2, 0.8, 13)
    assert torch.allclose(w.t_of(w.tau(ti)), ti, atol=2e-3)


def test_prob_correct_matches_monte_carlo():
    V, N = 12, 200_000
    g = torch.Generator().manual_seed(0)
    for t in (0.3, 0.5, 0.7):
        x = (1 - t) * torch.randn(N, V, generator=g)
        x[:, 0] += t
        mc = (x.argmax(-1) == 0).float().mean().item()
        assert abs(mc - prob_correct(torch.tensor([t]), V).item()) < 0.01


def test_jump_boundaries():
    x, d = torch.randn(4, 9, 12), torch.softmax(torch.randn(4, 9, 12), -1)
    s = torch.full((4,), 0.3)
    assert torch.allclose(jump(x, s, s, d), x)
    assert torch.allclose(jump(x, s, torch.ones(4), d), d)


def test_semigroup_weight_gamma():
    """delta_{s,t} recovered from X_{u,t} o X_{s,u} equals gamma delta_{s,u} + (1-gamma) delta_{u,t}."""
    x, d1, d2 = torch.randn(3, 5, 7), torch.randn(3, 5, 7), torch.randn(3, 5, 7)
    s, u, t = torch.full((3,), 0.2), torch.full((3,), 0.5), torch.full((3,), 0.9)
    Xst = jump(jump(x, s, u, d1), u, t, d2)
    d_st = (Xst - ((1 - t) / (1 - s))[:, None, None] * x) * ((1 - s) / (t - s))[:, None, None]
    gamma = ((1 - t) * (u - s) / ((1 - u) * (t - s)))[:, None, None]
    assert torch.allclose(d_st, gamma * d1 + (1 - gamma) * d2, atol=1e-5)


def test_pretrain_loss_and_sampler_run():
    fm = tiny()
    x1 = torch.nn.functional.one_hot(torch.randint(0, 12, (8, 9)), 12).float()
    out = pretrain_loss(fm, x1)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert sample(fm, torch.randn(2, 9, 12), K=3).shape == (2, 9, 12)


def test_logp_matches_bruteforce():
    logits, y = torch.randn(2, 3, 4), torch.randint(0, 4, (2, 3))
    p = torch.softmax(logits / 2.0, -1)
    brute = torch.log(torch.stack([p[b, torch.arange(3), y[b]].prod() for b in range(2)]))
    assert torch.allclose(token_logp(logits, y, 2.0).sum(-1), brute, atol=1e-5)


def test_policy_logits_options_have_gradients():
    fm = tiny()
    z = torch.randn(2, 9, 12)
    for opt, K in [("A", 1), ("A", 3), ("B", 2)]:
        fm.zero_grad()
        policy_logits(fm, z, K, opt, 0.9).sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fm.parameters())


def test_reinforce_loo_is_unbiased_on_a_bandit():
    """Policy prod_i softmax(theta_i); mean LOO gradient matches the exact gradient of E[r]."""
    torch.manual_seed(0)
    L, V, G, groups = 2, 3, 4, 40_000
    theta = torch.randn(L, V, requires_grad=True)
    table = torch.rand(V**L)
    ys = torch.tensor(list(itertools.product(range(V), repeat=L)))
    logp_all = torch.log_softmax(theta, -1)[torch.arange(L), ys].sum(-1)
    exact = torch.autograd.grad((logp_all.exp() * table).sum(), theta)[0]
    p = torch.softmax(theta.detach(), -1)
    y = torch.stack([torch.multinomial(p[i], groups * G, replacement=True) for i in range(L)], -1)
    r = table[(y * torch.tensor([V, 1])).sum(-1)]
    logp = torch.log_softmax(theta, -1)[torch.arange(L), y]          # [N, L]
    loss = pg_loss(logp, logp.detach(), advantages(r, G, "loo"), None) * L  # undo the 1/L token mean
    est = torch.autograd.grad(-loss, theta)[0]
    assert torch.allclose(est, exact, atol=0.02), (est, exact)


def test_rl_update_runs_with_clip_and_kl():
    fm, ref = tiny(), tiny()
    cfg = RLConfig(K=2, groups=2, group_size=4, advantage="std", mu=2, clip=0.2, beta=0.1)
    z = torch.randn(8, 9, 12)
    with torch.no_grad():
        lg = policy_logits(fm, z, 2)
        y = lg.argmax(-1)
        ro = Rollout(z, y, token_logp(lg, y, 1.0), torch.rand(8), policy_logits(ref, z, 2))
    opt = torch.optim.SGD(fm.parameters(), lr=1e-3)
    d = rl_update(fm, opt, ro, cfg)
    assert all(map(lambda v: v == v, d.values()))


def test_conditional_modes_pretrain_policy_and_rl():
    from rlfm.common import make_flowmap, make_task, one_hot, split
    for mode in ("state", "encoder"):
        cfg = {"task": {"name": "arithmetic", "frac_correct": 1.0, "holdout_frac": 0.1},
               "cond": {"mode": mode, "enc_depth": 1}, "model": {"dim": 32, "depth": 1, "heads": 2}}
        task, rm = make_task(cfg)
        fm = make_flowmap(cfg, task.vocab_size, task.seq_len, task)
        q, ans = split(task.sample(8), task)
        out = pretrain_loss(fm, one_hot(ans, 12), q=q)
        out["loss"].backward()
        assert torch.isfinite(out["loss"])
        z = torch.randn(8, 3, 12)
        assert sample(fm, z, 2, q).shape == (8, 3, 12)
        assert policy_logits(fm, z, 1, "A", 0.9, q).shape == (8, 3, 12)
        # changing the question changes the output: the network reads it (output layer starts at zero)
        for mod in fm.net.modules():  # blocks start as the identity (zero gates); perturb them
            if isinstance(mod, torch.nn.Linear) and mod.out_features in (12, 6 * 32, 2 * 32):
                torch.nn.init.normal_(mod.weight, std=0.1)
        q2 = task.sample_questions(8)
        assert not torch.allclose(fm.logits(z, torch.zeros(8), torch.ones(8), q),
                                  fm.logits(z, torch.zeros(8), torch.ones(8), q2))
        from rlfm.rl import collect, rl_update
        cfg_rl = RLConfig(groups=2, group_size=4)
        ro = collect(fm, None, lambda y, qq: rm.reward(task.decode(torch.cat([qq, y], 1))), cfg_rl, 3, 12, "cpu",
                     question_fn=lambda B: task.sample_questions(B))
        assert (ro.q[0] == ro.q[3]).all() and ro.y.shape == (8, 3)
        rl_update(fm, torch.optim.SGD(fm.parameters(), lr=1e-3), ro, cfg_rl)


def test_per_token_advantages():
    from rlfm.rl import advantages, pg_loss
    r = torch.tensor([[1., 0.], [0., 0.], [1., 1.], [0., 1.]])  # 2 groups of G=2, L=2
    A = advantages(r, 2, "loo")
    assert torch.equal(A, torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]]))
    assert torch.equal(advantages(r.mean(1), 2, "loo"), torch.tensor([0.5, -0.5, 0.5, -0.5]))
    logp = torch.full((4, 2), -1.0, requires_grad=True)
    pg_loss(logp, logp.detach(), A, None).backward()
    assert torch.equal(logp.grad, -A / 8)


def _perturbed_tiny():
    fm = tiny()
    for mod in fm.net.modules():  # zero-initialised layers would make every jump output uniform logits
        if isinstance(mod, torch.nn.Linear) and mod.out_features in (12, 6 * 32, 2 * 32):
            torch.nn.init.normal_(mod.weight, std=0.1)
    return fm


def test_stochastic_chain_one_jump_is_the_current_policy():
    from rlfm.policy import chain_logits, sample_tokens, stochastic_chain
    fm, z = _perturbed_tiny(), torch.randn(6, 9, 12)
    xs, ys, ls = stochastic_chain(fm, z, 1, 1.0, gen=torch.Generator().manual_seed(3))
    lg = policy_logits(fm, z, 1)
    assert torch.allclose(ls[:, 0], lg, atol=1e-5)
    assert torch.equal(ys[:, 0], sample_tokens(lg, 1.0, torch.Generator().manual_seed(3)))
    xs, ys, ls = stochastic_chain(fm, z, 3, 1.0)
    assert xs.shape == (6, 3, 9, 12) and ys.shape == (6, 3, 9)
    assert torch.allclose(chain_logits(fm, xs, 3), ls, atol=1e-5)  # batched recomputation matches the rollout
    # the bridge: x_{k+1} - ((1-t1)/(1-t0)) x_k is (t1-t0)/(1-t0) times the one-hot of y_k
    from rlfm.flowmap import grid
    g = grid(fm, 3)
    r = (xs[:, 1] - (1 - g[1]) / (1 - g[0]) * xs[:, 0]) / ((g[1] - g[0]) / (1 - g[0]))
    assert torch.allclose(r, torch.nn.functional.one_hot(ys[:, 0], 12).float(), atol=1e-4)


def test_stochastic_and_rwr_updates_run():
    from rlfm.rl import collect
    fm = _perturbed_tiny()
    for cfg in (RLConfig(K=3, groups=2, group_size=4, stochastic=True),
                RLConfig(K=3, groups=2, group_size=4, stochastic=True, mu=2, clip=0.2),
                RLConfig(K=3, groups=2, group_size=4, rwr=1.0),
                RLConfig(K=3, groups=2, group_size=4, entropy=0.1),
                RLConfig(K=3, groups=2, group_size=4, stochastic=True, entropy=0.1)):
        ro = collect(fm, None, lambda y: (y == 3).float(), cfg, 9, 12, "cpu")  # per-token reward [N, L]
        assert ro.reward.shape == (8, 9)
        before = [p.clone() for p in fm.parameters()]
        d = rl_update(fm, torch.optim.SGD(fm.parameters(), lr=1e-2), ro, cfg)
        assert all(v == v for v in d.values())
        assert any(not torch.equal(a, b) for a, b in zip(before, fm.parameters()))


def test_entropy_bonus_alone_raises_entropy():
    """With zero advantages, the update is the entropy bonus alone: entropy of the policy goes up."""
    from rlfm.policy import policy_logits, token_entropy
    from rlfm.rl import collect
    fm = _perturbed_tiny()
    cfg = RLConfig(K=2, groups=2, group_size=4, entropy=1.0)
    ro = collect(fm, None, lambda y: torch.zeros(y.shape[0]), cfg, 9, 12, "cpu")
    h0 = token_entropy(policy_logits(fm, ro.z, 2), 1.0).mean()
    opt = torch.optim.SGD(fm.parameters(), lr=1e-1)
    for _ in range(5):
        rl_update(fm, opt, ro, cfg)
    assert token_entropy(policy_logits(fm, ro.z, 2), 1.0).mean() > h0


def test_sft_loss_decreases_on_fixed_targets():
    from rlfm.rl import sft_loss
    fm = _perturbed_tiny()
    cfg = RLConfig(K=2)
    z, y = torch.randn(8, 9, 12), torch.randint(0, 12, (8, 9))
    opt = torch.optim.Adam(fm.parameters(), lr=1e-3)
    l0 = sft_loss(fm, z, y, cfg).item()
    for _ in range(20):
        loss = sft_loss(fm, z, y, cfg)
        opt.zero_grad(); loss.backward(); opt.step()
    assert sft_loss(fm, z, y, cfg).item() < l0


def test_ar_kl_zero_at_reference_and_positive_after_change():
    import copy
    from rlfm.ar import ARNet, ar_kl, ar_sample
    torch.manual_seed(0)
    net = ARNet(18, d=32, depth=1, heads=2)
    ref = copy.deepcopy(net)
    q = torch.randint(0, 10, (4, 11))
    y = ar_sample(net, q, 7)
    assert ar_kl(net, ref, q, y).abs().max().item() < 1e-5
    with torch.no_grad():
        net.out.bias += torch.randn_like(net.out.bias)
    kl = ar_kl(net, ref, q, y)
    assert kl.shape == (4, 7) and (kl > 0).all()


def test_micro_batches_give_the_same_update():
    """rl.micro = 2 sums the gradients of two halves of the batch: the same step as the whole batch at once."""
    import copy
    from rlfm.rl import collect
    base = _perturbed_tiny()
    ref = tiny()
    for kw in (dict(), dict(mu=2, clip=0.2, beta=0.1), dict(stochastic=True), dict(rwr=1.0), dict(entropy=0.1)):
        cfg1 = RLConfig(K=3, groups=2, group_size=4, **kw)
        torch.manual_seed(1)
        ro = collect(base, ref if cfg1.beta > 0 else None, lambda y: (y == 3).float().mean(1), cfg1, 9, 12, "cpu")
        out = []
        for micro in (1, 2):
            fm = copy.deepcopy(base)
            cfg = RLConfig(K=3, groups=2, group_size=4, micro=micro, **kw)
            d = rl_update(fm, torch.optim.SGD(fm.parameters(), lr=1e-1), ro, cfg, grad_clip=1e9)
            out.append((torch.cat([p.flatten() for p in fm.parameters()]), d))
        assert torch.allclose(out[0][0], out[1][0], atol=1e-6), kw
        for k in out[0][1]:
            assert abs(out[0][1][k] - out[1][1][k]) < 1e-4 * (1 + abs(out[0][1][k])), (kw, k)


def test_noise_scale_of_rollouts():
    """rl.zs = s draws z ~ N(0, s^2 I): s = 0 gives z = 0, s = 0.5 halves the noise of the same generator."""
    from rlfm.rl import collect
    fm = _perturbed_tiny()
    zs = {}
    for s in (0.0, 0.5, 1.0):
        cfg = RLConfig(K=2, groups=2, group_size=4, zs=s)
        zs[s] = collect(fm, None, lambda y: torch.zeros(y.shape[0]), cfg, 9, 12, "cpu",
                        gen=torch.Generator().manual_seed(0)).z
    assert torch.all(zs[0.0] == 0)
    assert torch.allclose(zs[0.5], 0.5 * zs[1.0])
