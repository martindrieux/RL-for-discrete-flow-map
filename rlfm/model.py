"""DiT-style denoiser: continuous state x in R^{L x V} and times (s, t) -> logits in R^{L x V}.

delta_{s,t}(x) = softmax(logits) is the two-time denoiser of Lee et al. (2026).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """Sinusoidal embedding of a time in [0, 1], shape [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = scale * t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, d: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(time_embedding(t, self.freq_dim))


def rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on [B, H, L, Dh] (Dh even)."""
    L, dh = x.shape[-2], x.shape[-1]
    half = dh // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
    ang = torch.arange(L, device=x.device)[:, None] * freqs[None, :]
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    """Pre-norm transformer block with AdaLN-zero modulation."""

    def __init__(self, d: int, heads: int, mlp_ratio: int = 4, causal: bool = False):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(approximate="tanh"), nn.Linear(mlp_ratio * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, L, d = h.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c)[:, None, :].chunk(6, dim=-1)
        a = self.norm1(h) * (1 + sc1) + sh1
        q, k, v = self.qkv(a).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=self.causal)
        h = h + g1 * self.proj(a.transpose(1, 2).reshape(B, L, d))
        m = self.norm2(h) * (1 + sc2) + sh2
        return h + g2 * self.mlp(m)


class DiT(nn.Module):
    """Logits network: forward(x [B,L,V], tau_s [B], tau_t [B]) -> logits [B,L,V]."""

    def __init__(self, vocab_size: int, seq_len: int, d: int = 128, depth: int = 4, heads: int = 4):
        super().__init__()
        self.vocab_size, self.seq_len = vocab_size, seq_len
        self.inp = nn.Linear(vocab_size, d, bias=False)  # x E, a linear map of the continuous state
        self.emb_s, self.emb_t = TimeMLP(d), TimeMLP(d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, vocab_size)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, tau_s: torch.Tensor, tau_t: torch.Tensor) -> torch.Tensor:
        c = self.emb_s(tau_s) + self.emb_t(tau_t)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, c)
        sh, sc = self.ada(c)[:, None, :].chunk(2, dim=-1)
        return self.out(self.norm(h) * (1 + sc) + sh)


class CondDiT(nn.Module):
    """Conditional logits network: forward(x [B,La,V], tau_s, tau_t, q [B,Lq]) -> logits [B,La,V] for the answer.

    mode "state":   the question enters as clean one-hot rows, cat(onehot(q), x), through the same input map x E.
    mode "encoder": the question is read by a separate encoder (token embedding + `enc_depth` blocks); its
                    features are prepended to the answer tokens in the main network.
    In both modes a learned segment embedding marks positions as given (question) or to generate (answer).
    """

    def __init__(self, vocab_size: int, prompt_len: int, answer_len: int, mode: str = "state", d: int = 128,
                 depth: int = 4, heads: int = 4, enc_depth: int = 2):
        super().__init__()
        assert mode in ("state", "encoder")
        self.mode, self.vocab_size, self.prompt_len, self.answer_len = mode, vocab_size, prompt_len, answer_len
        self.inp = nn.Linear(vocab_size, d, bias=False)
        self.segment = nn.Embedding(2, d)  # 0 = given (question), 1 = to generate (answer)
        if mode == "encoder":
            self.tok = nn.Embedding(vocab_size, d)
            self.enc = nn.ModuleList([Block(d, heads) for _ in range(enc_depth)])
            for blk in self.enc:  # the encoder is not time-modulated: identity AdaLN, unit gates
                nn.init.zeros_(blk.ada[1].weight)
                with torch.no_grad():
                    blk.ada[1].bias.zero_()
                    blk.ada[1].bias[2 * d:3 * d] = 1.0
                    blk.ada[1].bias[5 * d:6 * d] = 1.0
        self.emb_s, self.emb_t = TimeMLP(d), TimeMLP(d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, vocab_size)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, tau_s: torch.Tensor, tau_t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        B, Lq, La = x.shape[0], q.shape[1], x.shape[1]
        c = self.emb_s(tau_s) + self.emb_t(tau_t)
        if self.mode == "state":
            hq = self.inp(F.one_hot(q, self.vocab_size).to(x.dtype))
        else:
            hq = self.tok(q)
            zero = torch.zeros(B, hq.shape[-1], device=x.device, dtype=hq.dtype)
            for blk in self.enc:
                hq = blk(hq, zero)
        seg = torch.cat([torch.zeros(Lq, dtype=torch.long), torch.ones(La, dtype=torch.long)]).to(x.device)
        h = torch.cat([hq, self.inp(x)], dim=1) + self.segment(seg)[None]
        for blk in self.blocks:
            h = blk(h, c)
        sh, sc = self.ada(c)[:, None, :].chunk(2, dim=-1)
        return self.out(self.norm(h) * (1 + sc) + sh)[:, Lq:]
