"""The flow map as a stochastic policy: noise z -> logits -> independent categorical tokens.

Option A: logits of the last jump t_{K-1} -> 1 on the grid t_k = t(k/K).
Option B: run K jumps to t* = t(tau_star), logits of the diagonal delta_{t*,t*} (token posterior).
Gradients flow through every jump.
"""
import torch
import torch.nn.functional as F

from .flowmap import FlowMap, grid, jump
from .tracking import nb


@nb.core
def policy_logits(fm: FlowMap, z: torch.Tensor, K: int, option: str = "A", tau_star: float = 0.9) -> torch.Tensor:
    """Logits l(z) in R^{B x L x V}, differentiable in the network weights."""
    B = z.shape[0]
    if option == "A":
        g, n_jumps = grid(fm, K, device=z.device), K - 1
    elif option == "B":
        g, n_jumps = grid(fm, K, tau_end=tau_star, device=z.device), K
    else:
        raise ValueError(option)
    x = z
    for k in range(n_jumps):
        s, t = g[k].expand(B), g[k + 1].expand(B)
        x = jump(x, s, t, fm.delta(x, s, t))
    if option == "A":
        return fm.logits(x, g[K - 1].expand(B), g[K].expand(B)).float()
    return fm.logits(x, g[K].expand(B), g[K].expand(B)).float()


def token_logp(logits: torch.Tensor, y: torch.Tensor, temp: float) -> torch.Tensor:
    """log p_i(y_i | z) with p_i = softmax(l_i / T); shape [B, L]."""
    return F.log_softmax(logits / temp, dim=-1).gather(-1, y[..., None]).squeeze(-1)


def sample_tokens(logits: torch.Tensor, temp: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """y_i ~ Cat(softmax(l_i / T)) independently; shape [B, L]."""
    probs = F.softmax(logits / temp, dim=-1)
    B, L, V = probs.shape
    return torch.multinomial(probs.reshape(B * L, V), 1, generator=gen).view(B, L)


def token_entropy(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """H(p_i), shape [B, L]."""
    logp = F.log_softmax(logits / temp, dim=-1)
    return -(logp.exp() * logp).sum(-1)


def token_kl(logits_p: torch.Tensor, logits_q: torch.Tensor, temp: float) -> torch.Tensor:
    """KL(p_i || q_i) exactly over the vocabulary, shape [B, L]."""
    lp, lq = F.log_softmax(logits_p / temp, dim=-1), F.log_softmax(logits_q / temp, dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1)
