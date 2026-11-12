"""RL on the categorical likelihood log pi(y | z): REINFORCE-LOO, std-normalised advantages, PPO/GRPO clip, KL.

All losses are means over the G*L tokens of each group (and over groups).
"""
from dataclasses import dataclass
from typing import Callable

import torch

from .flowmap import FlowMap
from .policy import policy_logits, sample_tokens, token_entropy, token_kl, token_logp
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


@dataclass
class Rollout:
    z: torch.Tensor          # [N, L, V], N = B * G, groups contiguous
    y: torch.Tensor          # [N, L]
    logp_old: torch.Tensor   # [N, L] per-token old log-probabilities
    reward: torch.Tensor     # [N]
    logits_ref: torch.Tensor | None  # [N, L, V] reference-policy logits from the same z
    q: torch.Tensor | None = None    # [N, Lq] question of each sample (shared within a group), or None


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
    logits = policy_logits(fm, z, cfg.K, cfg.option, cfg.tau_star, q)
    y = sample_tokens(logits, cfg.temp, gen)
    logits_ref = policy_logits(ref, z, cfg.K, cfg.option, cfg.tau_star, q) if (ref is not None and cfg.beta > 0) else None
    r = reward_fn(y) if q is None else reward_fn(y, q)
    return Rollout(z, y, token_logp(logits, y, cfg.temp), r.to(device).float(), logits_ref, q)


@nb.core
def advantages(reward: torch.Tensor, G: int, kind: str) -> torch.Tensor:
    """loo: A_g = r_g - mean_{j != g} r_j.  std: A_g = (r_g - mean_j r_j) / (std_j r_j + 1e-6)."""
    r = reward.view(-1, G)
    if kind == "loo":
        A = r - (r.sum(1, keepdim=True) - r) / (G - 1)
    elif kind == "std":
        A = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    else:
        raise ValueError(kind)
    return A.reshape(-1)


@nb.core
def pg_loss(logp: torch.Tensor, logp_old: torch.Tensor, A: torch.Tensor, clip: float | None) -> torch.Tensor:
    """No clip: -mean_{g,i} sg(A_g) log p_i.  Clip: -mean_{g,i} min(rho A, clip(rho, 1-eps, 1+eps) A)."""
    A = A.detach()[:, None]
    if clip is None:
        return -(A * logp).mean()
    rho = torch.exp(logp - logp_old)
    return -torch.minimum(rho * A, rho.clamp(1 - clip, 1 + clip) * A).mean()


def rl_update(fm: FlowMap, opt: torch.optim.Optimizer, ro: Rollout, cfg: RLConfig, grad_clip: float = 1.0) -> dict:
    """mu gradient steps on one rollout; returns diagnostics of the last step."""
    A = advantages(ro.reward, cfg.group_size, cfg.advantage)
    for _ in range(cfg.mu):
        logits = policy_logits(fm, ro.z, cfg.K, cfg.option, cfg.tau_star, ro.q)
        logp = token_logp(logits, ro.y, cfg.temp)
        loss = pg_loss(logp, ro.logp_old, A, cfg.clip)
        kl = token_kl(logits, ro.logits_ref, cfg.temp).mean() if ro.logits_ref is not None else torch.zeros((), device=logits.device)
        total = loss + cfg.beta * kl
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(fm.parameters(), grad_clip)
        opt.step()
    return {"pg_loss": loss.item(), "kl_train": kl.item(), "grad_norm": gn.item(),
            "entropy_train": token_entropy(logits.detach(), cfg.temp).mean().item()}
