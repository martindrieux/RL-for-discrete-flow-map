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
def policy_logits(fm: FlowMap, z: torch.Tensor, K: int, option: str = "A", tau_star: float = 0.9,
                  q: torch.Tensor | None = None) -> torch.Tensor:
    """Logits l(z, q) in R^{B x L x V} (answer positions only when a question q is given), differentiable."""
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
        x = jump(x, s, t, fm.delta(x, s, t, q))
    if option == "A":
        return fm.logits(x, g[K - 1].expand(B), g[K].expand(B), q).float()
    return fm.logits(x, g[K].expand(B), g[K].expand(B), q).float()


@nb.core
def stochastic_chain(fm: FlowMap, z: torch.Tensor, K: int, temp: float, q: torch.Tensor | None = None,
                     gen: torch.Generator | None = None, greedy: bool = False):
    """Every jump samples tokens: y_k ~ Cat(softmax(l_{t_k,t_{k+1}}(x_k) / T)),
    x_{k+1} = ((1-t_{k+1})/(1-t_k)) x_k + ((t_{k+1}-t_k)/(1-t_k)) onehot(y_k)   (the interpolant's bridge).

    Returns states [B, K, L, V] (x_0 = z, ..., x_{K-1}), tokens [B, K, L] (y_{K-1} is the answer), logits [B, K, L, V].
    greedy: argmax instead of sampling at every jump. No gradient: states are treated as given.
    """
    B, V = z.shape[0], z.shape[-1]
    g, x = grid(fm, K, device=z.device), z
    xs, ys, ls = [], [], []
    with torch.no_grad():
        for k in range(K):
            s, t = g[k].expand(B), g[k + 1].expand(B)
            logits = fm.logits(x, s, t, q).float()
            y = logits.argmax(-1) if greedy else sample_tokens(logits, temp, gen)
            xs.append(x); ys.append(y); ls.append(logits)
            x = jump(x, s, t, F.one_hot(y, V).to(x.dtype))
    return torch.stack(xs, 1), torch.stack(ys, 1), torch.stack(ls, 1)


def chain_logits(fm: FlowMap, xs: torch.Tensor, K: int, q: torch.Tensor | None = None) -> torch.Tensor:
    """Logits l_{t_k, t_{k+1}}(x_k) at stored states xs [B, K, L, V], in one batched call."""
    B, _, L, V = xs.shape
    g = grid(fm, K, device=xs.device)
    s = g[:K].repeat(B)
    t = g[1:].repeat(B)
    qq = None if q is None else q.repeat_interleave(K, 0)
    return fm.logits(xs.reshape(B * K, L, V), s, t, qq).float().view(B, K, L, V)


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
