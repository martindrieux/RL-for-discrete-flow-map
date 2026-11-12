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


@torch.no_grad()
def collect(fm: FlowMap, ref: FlowMap | None, reward_fn: Callable[[torch.Tensor], torch.Tensor], cfg: RLConfig,
            L: int, V: int, device, gen: torch.Generator | None = None) -> Rollout:
    """Draw B groups of G pairs (z, y) from the current policy and score them."""
    N = cfg.groups * cfg.group_size
    z = torch.randn(N, L, V, generator=gen, device=device)
    logits = policy_logits(fm, z, cfg.K, cfg.option, cfg.tau_star)
    y = sample_tokens(logits, cfg.temp, gen)
    logits_ref = policy_logits(ref, z, cfg.K, cfg.option, cfg.tau_star) if (ref is not None and cfg.beta > 0) else None
    return Rollout(z, y, token_logp(logits, y, cfg.temp), reward_fn(y).to(device).float(), logits_ref)


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
        logits = policy_logits(fm, ro.z, cfg.K, cfg.option, cfg.tau_star)
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
