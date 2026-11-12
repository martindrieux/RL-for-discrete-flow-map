"""Synthetic addition equations `ab+cd=efg` as fixed-length token sequences."""

import torch

CHARS = "0123456789+="
STOI = {c: i for i, c in enumerate(CHARS)}
PLUS, EQ = STOI["+"], STOI["="]


class ArithmeticTask:
    """Generator of `a+b=c` strings, a mix of correct and incorrect sums."""

    def __init__(self, n_digits: int = 2, frac_correct: float = 0.5) -> None:
        """Store the digit count and fraction of correct equations."""
        self.n_digits = n_digits
        self.frac_correct = frac_correct
        self.vocab_size = len(CHARS)
        self.seq_len = 3 * n_digits + 3

    def encode(self, s: str) -> torch.Tensor:
        """Convert a string of length L into a LongTensor of token ids [L]."""
        return torch.tensor([STOI[c] for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str | list[str]:
        """Convert ids [L] to a string, or ids [B, L] to a list of strings."""
        if ids.dim() == 1:
            return "".join(CHARS[i] for i in ids.tolist())
        return ["".join(CHARS[i] for i in row) for row in ids.tolist()]

    @staticmethod
    def _digits(x: torch.Tensor, k: int) -> torch.Tensor:
        """Zero-padded base-10 digits of x [B], most significant first: [B, k]."""
        powers = 10 ** torch.arange(k - 1, -1, -1, device=x.device)
        return (x.unsqueeze(1) // powers) % 10

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw a batch [B, L] of equations, a fraction `frac_correct` of which are correct."""
        n, hi = self.n_digits, 10**self.n_digits
        a = torch.randint(0, hi, (batch_size,), generator=generator)
        b = torch.randint(0, hi, (batch_size,), generator=generator)
        true = a + b
        # Uniform over [0, 2*hi - 1) minus {true}: draw in [0, 2*hi - 2) and skip over `true`.
        r = torch.randint(0, 2 * hi - 2, (batch_size,), generator=generator)
        wrong = r + (r >= true).long()
        is_correct = torch.rand(batch_size, generator=generator) < self.frac_correct
        c = torch.where(is_correct, true, wrong)
        plus = torch.full((batch_size, 1), PLUS, dtype=torch.long)
        eq = torch.full((batch_size, 1), EQ, dtype=torch.long)
        return torch.cat([self._digits(a, n), plus, self._digits(b, n), eq, self._digits(c, n + 1)], dim=1)
