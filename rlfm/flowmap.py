"""Flow map from the two-time denoiser, FMLM direct-training losses, and the K-jump sampler.

States: x_t in R^{L x V}, I_t = (1-t) x0 + t x1, x0 ~ N(0, I), x1 one-hot; t = 0 noise, t = 1 data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .timewarp import TimeWarp
from .tracking import nb

EPS = 1e-4


class FlowMap(nn.Module):
    """Wraps the logits network; times passed in are raw times t, the network sees tau(t)."""

    def __init__(self, net: nn.Module, warp: TimeWarp):
        super().__init__()
        self.net, self.warp = net, warp

    def logits(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
        """q: question ids [B, Lq] for a conditional network (the state x then holds only the answer)."""
        if q is None:
            return self.net(x, self.warp.tau(s), self.warp.tau(t))
        return self.net(x, self.warp.tau(s), self.warp.tau(t), q)

    def delta(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, q: torch.Tensor | None = None) -> torch.Tensor:
        return F.softmax(self.logits(x, s, t, q).float(), dim=-1)


def _b(v: torch.Tensor) -> torch.Tensor:
    return v[:, None, None]


@nb.core
def jump(x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """X_{s,t}(x) = ((1-t)/(1-s)) x + ((t-s)/(1-s)) delta_{s,t}(x)."""
    one_s = (1 - s).clamp_min(EPS)
    return _b((1 - t) / one_s) * x + _b((t - s) / one_s) * delta


def interpolant(x1: torch.Tensor, t: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    x0 = torch.randn(x1.shape, generator=gen, device=x1.device)
    return _b(1 - t) * x0 + _b(t) * x1


@nb.core
def diag_loss(fm: FlowMap, x1: torch.Tensor, gen: torch.Generator | None = None, q: torch.Tensor | None = None) -> torch.Tensor:
    """tau ~ U[0,1], t = t(tau); L = -(1/L) sum_l x1_l . log delta_{t,t}(I_t)_l, averaged over the batch."""
    B = x1.shape[0]
    t = fm.warp.t_of(torch.rand(B, generator=gen, device=x1.device)).clamp(0, 1 - EPS)
    logp = F.log_softmax(fm.logits(interpolant(x1, t, gen), t, t, q).float(), dim=-1)
    return -(x1 * logp).sum(-1).mean()


@nb.core
def semigroup_loss(fm: FlowMap, x1: torch.Tensor, gen: torch.Generator | None = None, q: torch.Tensor | None = None) -> torch.Tensor:
    """Self-distilled semigroup loss; (s, u, t) drawn uniformly in tau with u the tau-midpoint."""
    B, dev = x1.shape[0], x1.device
    h = torch.rand(B, generator=gen, device=dev)
    tau_s = torch.rand(B, generator=gen, device=dev) * (1 - h)
    tau_t = tau_s + h
    s = fm.warp.t_of(tau_s).clamp(0, 1 - EPS)
    t = torch.maximum(fm.warp.t_of(tau_t), s + EPS).clamp(max=1.0)
    u = fm.warp.t_of(0.5 * (tau_s + tau_t)).clamp(min=s, max=t)
    x_s = interpolant(x1, s, gen)
    with torch.no_grad():
        d_su = fm.delta(x_s, s, u, q)
        x_u = jump(x_s, s, u, d_su)
        d_ut = fm.delta(x_u, u.clamp(max=1 - EPS), t, q)
        gamma = (1 - t) * (u - s) / ((1 - u).clamp_min(EPS) * (t - s).clamp_min(EPS))
        target = _b(gamma.clamp(0, 1)) * d_su + _b(1 - gamma.clamp(0, 1)) * d_ut
    logp = F.log_softmax(fm.logits(x_s, s, t, q).float(), dim=-1)
    return -(target * logp).sum(-1).mean()


def pretrain_loss(fm: FlowMap, x1: torch.Tensor, gen: torch.Generator | None = None, q: torch.Tensor | None = None) -> dict:
    """L = L_diag (first half of the batch) + L_off (second half); x1 is the answer only when q is given."""
    B = x1.shape[0] // 2
    qa, qb = (None, None) if q is None else (q[:B], q[B:])
    l_diag, l_off = diag_loss(fm, x1[:B], gen, qa), semigroup_loss(fm, x1[B:], gen, qb)
    return {"loss": l_diag + l_off, "loss_diag": l_diag.detach(), "loss_off": l_off.detach()}


def grid(fm: FlowMap, K: int, tau_end: float = 1.0, device=None) -> torch.Tensor:
    """t_k = t(k tau_end / K), k = 0..K."""
    g = fm.warp.t_of(torch.linspace(0, tau_end, K + 1, device=device))
    g[0] = 0.0
    if tau_end == 1.0:
        g[-1] = 1.0
    return g


@nb.core
@torch.no_grad()
def sample(fm: FlowMap, z: torch.Tensor, K: int, q: torch.Tensor | None = None) -> torch.Tensor:
    """x_0 = z, x_{k+1} = X_{t_k, t_{k+1}}(x_k) on t_k = t(k/K); returns x_K."""
    g, x, B = grid(fm, K, device=z.device), z, z.shape[0]
    for k in range(K):
        s, t = g[k].expand(B), g[k + 1].expand(B)
        x = jump(x, s, t, fm.delta(x, s, t, q))
    return x
