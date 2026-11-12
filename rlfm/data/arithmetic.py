"""Synthetic equations `ab+cd=efg` (addition) or `ab*cd=efgh` (multiplication) as fixed-length token sequences."""

import torch

CHARS = "0123456789+="          # addition; multiplication uses "*" at the same id as "+"
CHARS_MUL = "0123456789*="
STOI = {c: i for i, c in enumerate(CHARS)}
PLUS, EQ = STOI["+"], STOI["="]  # PLUS is the operator id for both operations


class ArithmeticTask:
    """Generator of `a+b=c` strings, a mix of correct and incorrect sums."""

    def __init__(self, n_digits: int = 2, frac_correct: float = 0.5, holdout_frac: float = 0.0,
                 holdout_seed: int = 0, op: str = "add") -> None:
        """Store the digit count and fraction of correct equations; hold out a fraction of operand pairs.

        A held-out pair (a, b) never appears in training data, with any sum.
        """
        assert n_digits <= 9, "operands must fit the integer hash below"
        self.n_digits = n_digits
        self.frac_correct = frac_correct
        self.holdout_frac, self.holdout_seed = holdout_frac, holdout_seed
        assert op in ("add", "mul")
        self.op = op
        self.chars = CHARS if op == "add" else CHARS_MUL
        self.vocab_size = len(self.chars)
        self.answer_len = n_digits + 1 if op == "add" else 2 * n_digits
        self.seq_len = 2 * n_digits + 2 + self.answer_len
        hi = 10**n_digits
        self.n_results = 2 * hi - 1 if op == "add" else hi * hi  # answers lie in [0, n_results)
        # Up to 3 digits: an explicit list of held-out pairs (exactly holdout_frac of all pairs).
        # From 4 digits: a pair is held out when a fixed hash of (a, b) falls below holdout_frac.
        self.hashed = n_digits >= 4
        if self.hashed:
            self.heldout_mask, self.train_pairs = None, None
        else:
            perm = torch.randperm(hi * hi, generator=torch.Generator().manual_seed(holdout_seed))
            n_out = int(round(holdout_frac * hi * hi))
            self.heldout_mask = torch.zeros(hi * hi, dtype=torch.bool)
            self.heldout_mask[perm[:n_out]] = True
            self.train_pairs = (~self.heldout_mask).nonzero().squeeze(1)

    def held_out(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """BoolTensor [B]: whether each operand pair is held out of training."""
        if not self.hashed:
            return self.heldout_mask[a * 10**self.n_digits + b]
        M = 2_147_483_647
        h = ((a % M) * 48271 + (b % M) * 69621 + self.holdout_seed * 40692 + 12345) % M
        h = (h * 16807 + 2_654_435) % M
        h = (h * 48271 + (h >> 7)) % M
        return (h % 100_000).float() < self.holdout_frac * 100_000

    def _pairs(self, batch_size: int, heldout: bool, generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Operand pairs drawn uniformly from the training pairs (heldout=False) or the held-out pairs."""
        hi = 10**self.n_digits
        if not self.hashed:
            pool = self.heldout_mask.nonzero().squeeze(1) if heldout else self.train_pairs
            if len(pool) == 0:
                raise ValueError("no held-out pairs: set holdout_frac > 0")
            pair = pool[torch.randint(0, len(pool), (batch_size,), generator=generator)]
            return pair // hi, pair % hi
        if heldout and self.holdout_frac <= 0:
            raise ValueError("no held-out pairs: set holdout_frac > 0")
        a_out, b_out, have = [], [], 0
        while have < batch_size:  # rejection sampling
            a = torch.randint(0, hi, (2 * batch_size + 64,), generator=generator)
            b = torch.randint(0, hi, (2 * batch_size + 64,), generator=generator)
            keep = self.held_out(a, b) == heldout
            a_out.append(a[keep]); b_out.append(b[keep]); have += int(keep.sum())
        return torch.cat(a_out)[:batch_size], torch.cat(b_out)[:batch_size]

    @property
    def prompt_len(self) -> int:
        """Length of the question `ab+cd=`; the answer is the last `answer_len` tokens."""
        return 2 * self.n_digits + 2

    def question_tokens(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Question ids [B, prompt_len] for operand tensors a, b [B]."""
        n, B = self.n_digits, a.shape[0]
        plus = torch.full((B, 1), PLUS, dtype=torch.long)
        eq = torch.full((B, 1), EQ, dtype=torch.long)
        return torch.cat([self._digits(a, n), plus, self._digits(b, n), eq], dim=1)

    def sample_questions(self, batch_size: int, heldout: bool = False,
                         generator: torch.Generator | None = None) -> torch.Tensor:
        """Questions [B, prompt_len] with operand pairs drawn uniformly from the training or the held-out pairs."""
        a, b = self._pairs(batch_size, heldout, generator)
        return self.question_tokens(a, b)

    def combine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a + b or a * b."""
        return a + b if self.op == "add" else a * b

    def true_answer(self, q: torch.Tensor) -> torch.Tensor:
        """Correct answer ids [B, answer_len] for questions q [B, prompt_len]."""
        n = self.n_digits
        w = 10 ** torch.arange(n - 1, -1, -1, device=q.device)
        a, b = (q[:, :n] * w).sum(1), (q[:, n + 1:2 * n + 1] * w).sum(1)
        return self._digits(self.combine(a, b), self.answer_len)

    def is_heldout(self, a: int, b: int) -> bool:
        """Whether the operand pair (a, b) is held out of training."""
        return bool(self.held_out(torch.tensor([a]), torch.tensor([b]))[0])

    def encode(self, s: str) -> torch.Tensor:
        """Convert a string of length L into a LongTensor of token ids [L]."""
        return torch.tensor([self.chars.index(c) for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str | list[str]:
        """Convert ids [L] to a string, or ids [B, L] to a list of strings."""
        if ids.dim() == 1:
            return "".join(self.chars[i] for i in ids.tolist())
        return ["".join(self.chars[i] for i in row) for row in ids.tolist()]

    @staticmethod
    def _digits(x: torch.Tensor, k: int) -> torch.Tensor:
        """Zero-padded base-10 digits of x [B], most significant first: [B, k]."""
        powers = 10 ** torch.arange(k - 1, -1, -1, device=x.device)
        return (x.unsqueeze(1) // powers) % 10

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw a batch [B, L] of equations, a fraction `frac_correct` of which are correct."""
        n, hi = self.n_digits, 10**self.n_digits
        a, b = self._pairs(batch_size, False, generator)
        true = self.combine(a, b)
        # Uniform over [0, n_results) minus {true}: draw in [0, n_results - 1) and skip over `true`.
        r = torch.randint(0, self.n_results - 1, (batch_size,), generator=generator)
        wrong = r + (r >= true).long()
        is_correct = torch.rand(batch_size, generator=generator) < self.frac_correct
        c = torch.where(is_correct, true, wrong)
        plus = torch.full((batch_size, 1), PLUS, dtype=torch.long)
        eq = torch.full((batch_size, 1), EQ, dtype=torch.long)
        return torch.cat([self._digits(a, n), plus, self._digits(b, n), eq, self._digits(c, self.answer_len)], dim=1)
