"""Time reparameterisation tau(t) = 1 - V/(V-1) P_e(t) (Lee et al. 2026, Eq. 25).

For the interpolant I_t = (1-t) x0 + t x1, x0 ~ N(0, I), x1 one-hot, the probability that argmax of a row
of I_t is the data token does not depend on the data:
    1 - P_e(t) = int phi(xi) Phi(xi + a)^(V-1) dxi,   a = t / (1 - t).
"""
import torch


def prob_correct(t: torch.Tensor, V: int, n_quad: int = 4001) -> torch.Tensor:
    """1 - P_e(t) by quadrature on xi in [-10, 10] (float64)."""
    t = t.double()
    xi = torch.linspace(-10.0, 10.0, n_quad, dtype=torch.float64)
    dxi = xi[1] - xi[0]
    phi = torch.exp(-0.5 * xi**2) / (2 * torch.pi) ** 0.5
    a = t / (1 - t).clamp_min(1e-12)
    cdf = torch.special.ndtr(xi[None, :] + a[:, None])
    return (phi[None, :] * cdf.pow(V - 1)).sum(-1) * dxi


def _interp(xq: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of the table (xs increasing, ys) at xq."""
    xq = xq.clamp(xs[0], xs[-1])
    i = torch.searchsorted(xs, xq.contiguous(), right=True).clamp(1, len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    w = ((xq - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
    return y0 + w * (y1 - y0)


class TimeWarp(torch.nn.Module):
    """Maps raw time t <-> reparameterised time tau, both in [0, 1]; `identity=True` disables it."""

    def __init__(self, V: int, n_table: int = 8193, identity: bool = False):
        super().__init__()
        ts = torch.linspace(0.0, 1.0, n_table, dtype=torch.float64)
        if identity:
            taus = ts.clone()
        else:
            pe = 1 - prob_correct(ts, V)
            taus = 1 - V / (V - 1) * pe
            taus[0], taus[-1] = 0.0, 1.0
            taus = torch.cummax(taus.clamp(0, 1), 0).values
            # make the table strictly increasing so the inverse is well defined on flat parts
            taus = taus + torch.linspace(0, 1e-9, n_table, dtype=torch.float64)
            taus = (taus - taus[0]) / (taus[-1] - taus[0])
        self.register_buffer("ts", ts.float())
        self.register_buffer("taus", taus.float())

    def tau(self, t: torch.Tensor) -> torch.Tensor:
        """tau(t)."""
        return _interp(t, self.ts, self.taus)

    def t_of(self, tau: torch.Tensor) -> torch.Tensor:
        """t(tau), the inverse map."""
        return _interp(tau, self.taus, self.ts)
