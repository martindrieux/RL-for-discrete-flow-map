"""Autoregressive baseline: a causal transformer of the same blocks and size as the flow map's network.

pi(y | c) = prod_i softmax(f(c, y_<i)_i / T)(y_i), the answer generated left to right after the question c.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flowmap import CLAMP_LOGIT
from .model import Block
from .policy import token_logp
from .tracking import nb


class ARNet(nn.Module):
    """forward(ids [B, L]) -> next-token logits [B, L, V] (position j predicts token j + 1)."""

    def __init__(self, vocab_size: int, d: int = 256, depth: int = 6, heads: int = 8, clamp_len: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.clamp_len = clamp_len  # Sudoku: question of clamp_len cells; answer cell i is fixed to q_i when q_i != 0
        self.tok = nn.Embedding(vocab_size, d)
        self.c = nn.Parameter(torch.zeros(d))  # constant conditioning of the AdaLN blocks (no time here)
        self.blocks = nn.ModuleList([Block(d, heads, causal=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h, c = self.tok(ids), self.c.expand(ids.shape[0], -1)
        for blk in self.blocks:
            h = blk(h, c)
        logits = self.out(self.norm(h)).float()
        return clamp_answer_logits(logits, ids, self.clamp_len) if self.clamp_len else logits


@nb.core
def clamp_answer_logits(logits: torch.Tensor, ids: torch.Tensor, Lq: int) -> torch.Tensor:
    """Position Lq - 1 + i predicts answer cell i; for q_i != 0 its output is fixed to onehot(q_i) (+C / -C logits)."""
    n = min(ids.shape[1] - Lq + 1, Lq)
    if n <= 0:
        return logits
    q = ids[:, :n]
    fixed = CLAMP_LOGIT * (2 * F.one_hot(q, logits.shape[-1]).float() - 1)
    mid = torch.where((q != 0)[..., None], fixed, logits[:, Lq - 1:Lq - 1 + n])
    return torch.cat([logits[:, :Lq - 1], mid, logits[:, Lq - 1 + n:]], 1)


def answer_logits(net: ARNet, q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Logits of every answer token given the question and the previous answer tokens: [B, La, V]."""
    ids = torch.cat([q, y], 1)
    return net(ids)[:, q.shape[1] - 1:-1]


@nb.core
def ar_logp(net: ARNet, q: torch.Tensor, y: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """log softmax(f(c, y_<i)_i / T)(y_i), shape [B, La]."""
    return token_logp(answer_logits(net, q, y), y, temp)


@nb.core
def ar_kl(net: ARNet, ref: ARNet, q: torch.Tensor, y: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """KL(p_theta(. | c, y_<i) || p_ref(. | c, y_<i)) over the vocabulary at each answer position of y, [B, La]."""
    lp = F.log_softmax(answer_logits(net, q, y) / temp, -1)
    with torch.no_grad():
        lr = F.log_softmax(answer_logits(ref, q, y) / temp, -1)
    return (lp.exp() * (lp - lr)).sum(-1)


@torch.no_grad()
def ar_sample(net: ARNet, q: torch.Tensor, La: int, temp: float = 1.0, greedy: bool = False,
              gen: torch.Generator | None = None) -> torch.Tensor:
    """y_i ~ softmax(f(c, y_<i)_i / T) for i = 1..La (or argmax), [B, La]."""
    ids = q
    for _ in range(La):
        logits = net(ids)[:, -1]
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(F.softmax(logits / temp, -1), 1, generator=gen).squeeze(1)
        ids = torch.cat([ids, nxt[:, None]], 1)
    return ids[:, q.shape[1]:]
