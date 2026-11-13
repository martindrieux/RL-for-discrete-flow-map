"""RL on the categorical likelihood log pi(y | z): REINFORCE-LOO, std-normalised advantages, PPO/GRPO clip, KL.

All losses are means over the G*L tokens of each group (and over groups).
"""
from dataclasses import dataclass
from typing import Callable

import torch

from .flowmap import FlowMap, grid, jump
from .policy import chain_logits, policy_logits, sample_tokens, stochastic_chain, token_entropy, token_kl, token_logp
from .tracking import nb


@dataclass
class RLConfig:
    K: int = 1
    option: str = "A"
    tau_star: float = 0.9
    temp: float = 1.0          # sampling temperature T
    groups: int = 32           # B prompts (here: groups of the unconditional task)
    group_size: int = 8        # G
    advantage: str = "loo"     # "loo" or "std"
    mu: int = 1                # gradient steps per batch
    clip: float | None = None  # epsilon_c; None = no clipping
    beta: float = 0.0          # KL weight
    stochastic: bool = False   # every jump samples tokens (option A); the policy is the chain of K categoricals
    rwr: float = 0.0           # lambda: weight of reward-weighted regression on the intermediate predictions l_{t_k,1}(x_k)


@dataclass
class Rollout:
    z: torch.Tensor          # [N, L, V], N = B * G, groups contiguous
    y: torch.Tensor          # [N, L]
    logp_old: torch.Tensor   # [N, L] per-token old log-probabilities
    reward: torch.Tensor     # [N], or [N, L] for a per-token reward
    logits_ref: torch.Tensor | None  # [N, L, V] reference-policy logits from the same z
    q: torch.Tensor | None = None    # [N, Lq] question of each sample (shared within a group), or None
    xs: torch.Tensor | None = None   # stochastic chain: states [N, K, L, V]
    ys: torch.Tensor | None = None   # stochastic chain: tokens of every jump [N, K, L]; logp_old is then [N, K, L]


@torch.no_grad()
def collect(fm: FlowMap, ref: FlowMap | None, reward_fn: Callable, cfg: RLConfig,
            L: int, V: int, device, gen: torch.Generator | None = None,
            question_fn: Callable[[int], torch.Tensor] | None = None) -> Rollout:
    """Draw B groups of G pairs (z, y) from the current policy and score them.

    Conditional: question_fn(B) gives B questions [B, Lq]; each is shared by the G samples of its group,
    L is then the answer length and reward_fn(y, q) scores the answer to the question.
    """
    N = cfg.groups * cfg.group_size
    q = None if question_fn is None else question_fn(cfg.groups).to(device).repeat_interleave(cfg.group_size, 0)
    z = torch.randn(N, L, V, generator=gen, device=device)
    if cfg.stochastic:
        assert cfg.option == "A" and cfg.beta == 0, "stochastic chain: option A, no KL penalty"
        xs, ys, logits = stochastic_chain(fm, z, cfg.K, cfg.temp, q, gen)
        y = ys[:, -1]
        r = reward_fn(y) if q is None else reward_fn(y, q)
        return Rollout(z, y, token_logp(logits, ys, cfg.temp), r.to(device).float(), None, q, xs, ys)
    logits = policy_logits(fm, z, cfg.K, cfg.option, cfg.tau_star, q)
    y = sample_tokens(logits, cfg.temp, gen)
    logits_ref = policy_logits(ref, z, cfg.K, cfg.option, cfg.tau_star, q) if (ref is not None and cfg.beta > 0) else None
    r = reward_fn(y) if q is None else reward_fn(y, q)
    return Rollout(z, y, token_logp(logits, y, cfg.temp), r.to(device).float(), logits_ref, q)


@nb.core
def advantages(reward: torch.Tensor, G: int, kind: str) -> torch.Tensor:
    """loo: A_g = r_g - mean_{j != g} r_j.  std: A_g = (r_g - mean_j r_j) / (std_j r_j + 1e-6).

    reward [N] gives A [N]; a per-token reward [N, L] gives A [N, L], each token compared within its group.
    """
    r = reward.view(-1, G, *reward.shape[1:])
    if kind == "loo":
        A = r - (r.sum(1, keepdim=True) - r) / (G - 1)
    elif kind == "std":
        A = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    else:
        raise ValueError(kind)
    return A.reshape(reward.shape)


@nb.core
def pg_loss(logp: torch.Tensor, logp_old: torch.Tensor, A: torch.Tensor, clip: float | None) -> torch.Tensor:
    """No clip: -mean_{g,i} sg(A_{g,i}) log p_i.  Clip: -mean_{g,i} min(rho A, clip(rho, 1-eps, 1+eps) A).

    A [N] (one advantage per sample, shared by its tokens) or [N, L] (one per token).
    """
    A = A.detach()
    A = A[:, None] if A.dim() == 1 else A
    if clip is None:
        return -(A * logp).mean()
    rho = torch.exp(logp - logp_old)
    return -torch.minimum(rho * A, rho.clamp(1 - clip, 1 + clip) * A).mean()


@nb.core
def rwr_loss(fm: FlowMap, z: torch.Tensor, y: torch.Tensor, A: torch.Tensor, K: int, temp: float,
             q: torch.Tensor | None = None) -> torch.Tensor:
    """-mean_{g,k<K-1,i} sg(A_{g,i}) log softmax(l_{t_k,1}(x_k)/T)_i(y_{g,i}): the final answer as target of every
    intermediate prediction of the clean answer, on the deterministic states x_k (given, no gradient through them)."""
    if K < 2:
        return torch.zeros((), device=z.device)
    with torch.no_grad():
        g, x, xs, B = grid(fm, K, device=z.device), z, [], z.shape[0]
        for k in range(K - 1):
            xs.append(x)
            s, t = g[k].expand(B), g[k + 1].expand(B)
            x = jump(x, s, t, fm.delta(x, s, t, q))
    xs = torch.stack(xs, 1)
    logits = _logits_to_one(fm, xs, g, q)
    logp = token_logp(logits, y[:, None].expand(-1, K - 1, -1), temp)  # [N, K-1, L]
    A = A.detach()
    A = A[:, None, None] if A.dim() == 1 else A[:, None, :]
    return -(A * logp).mean()


def _logits_to_one(fm: FlowMap, xs: torch.Tensor, g: torch.Tensor, q: torch.Tensor | None) -> torch.Tensor:
    """l_{t_k, 1}(x_k) for stored states xs [B, J, L, V] at times g[0..J-1]."""
    B, J, L, V = xs.shape
    s = g[:J].repeat(B)
    qq = None if q is None else q.repeat_interleave(J, 0)
    return fm.logits(xs.reshape(B * J, L, V), s, torch.ones_like(s), qq).float().view(B, J, L, V)


def rl_update(fm: FlowMap, opt: torch.optim.Optimizer, ro: Rollout, cfg: RLConfig, grad_clip: float = 1.0) -> dict:
    """mu gradient steps on one rollout; returns diagnostics of the last step."""
    A = advantages(ro.reward, cfg.group_size, cfg.advantage)
    for _ in range(cfg.mu):
        if ro.xs is not None:  # stochastic chain: every jump's tokens get the final advantage
            N, K, L = ro.ys.shape
            logits = chain_logits(fm, ro.xs, K, ro.q)
            logp = token_logp(logits, ro.ys, cfg.temp).reshape(N, K * L)
            Ak = A[:, None].expand(N, K * L) if A.dim() == 1 else A.repeat(1, K)
            loss = pg_loss(logp, ro.logp_old.reshape(N, K * L), Ak, cfg.clip)
            logits = logits[:, -1]
        else:
            logits = policy_logits(fm, ro.z, cfg.K, cfg.option, cfg.tau_star, ro.q)
            logp = token_logp(logits, ro.y, cfg.temp)
            loss = pg_loss(logp, ro.logp_old, A, cfg.clip)
        if cfg.rwr > 0:
            loss = loss + cfg.rwr * rwr_loss(fm, ro.z, ro.y, A, cfg.K, cfg.temp, ro.q)
        kl = token_kl(logits, ro.logits_ref, cfg.temp).mean() if ro.logits_ref is not None else torch.zeros((), device=logits.device)
        total = loss + cfg.beta * kl
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(fm.parameters(), grad_clip)
        opt.step()
    return {"pg_loss": loss.item(), "kl_train": kl.item(), "grad_norm": gn.item(),
            "entropy_train": token_entropy(logits.detach(), cfg.temp).mean().item()}
