"""Countdown: given four numbers and a target, write an expression that uses each number once and equals the target.

Tokens: question = four 2-digit numbers then a 3-digit target (11 digits); answer = 7 tokens of a postfix expression
over the number slots A, B, C, D (the 1st..4th number) and the operators + - * /, e.g. `AB+C*D-` = ((n1 + n2) * n3) - n4.
Integer arithmetic: a division must be exact. A question is generated from a random expression, so it has at least one
solution; that expression is the pretraining answer (one sample from the solutions, not the only one).
Held-out questions: a fixed hash of the sorted number set falls below holdout_frac; those sets never appear in training.
"""
import random

import torch

CHARS = "0123456789ABCD+-*/"
SLOTS, OPS = "ABCD", "+-*/"
# The five postfix shapes of a binary expression with 4 operands (N = number slot, O = operator)
SHAPES = ["NNONONO", "NNNOONO", "NNONNOO", "NNNONOO", "NNNNOOO"]


def evaluate(expr: str, nums: list[int]) -> int | None:
    """Value of a 7-token postfix expression over slots A..D of nums, or None if malformed, a slot is not used exactly
    once, or a division is not exact."""
    if len(expr) != 7 or sorted(c for c in expr if c in SLOTS) != list(SLOTS):
        return None
    st = []
    for c in expr:
        if c in SLOTS:
            st.append(nums[SLOTS.index(c)])
        elif c in OPS:
            if len(st) < 2:
                return None
            b, a = st.pop(), st.pop()
            if c == "+":
                st.append(a + b)
            elif c == "-":
                st.append(a - b)
            elif c == "*":
                st.append(a * b)
            else:
                if b == 0 or a % b:
                    return None
                st.append(a // b)
        else:
            return None
    return st[0] if len(st) == 1 else None


class CountdownTask:
    def __init__(self, holdout_frac: float = 0.1, holdout_seed: int = 0, max_num: int = 99, max_target: int = 999):
        self.holdout_frac, self.holdout_seed = holdout_frac, holdout_seed
        self.max_num, self.max_target = max_num, max_target
        self.chars = CHARS
        self.vocab_size = len(CHARS)
        self.prompt_len, self.answer_len = 11, 7
        self.seq_len = self.prompt_len + self.answer_len

    def _held(self, nums: list[int]) -> bool:
        a, b, c, d = sorted(nums)
        M = 2_147_483_647
        h = (((a * 101 + b) * 101 + c) * 101 + d + self.holdout_seed * 7919) % M
        h = (h * 48271) % M
        h = (h * 16807 + 2_654_435) % M
        return (h % 100_000) < self.holdout_frac * 100_000

    def _draw(self, rng: random.Random, heldout: bool) -> tuple[list[int], int, str]:
        while True:
            nums = [rng.randint(1, self.max_num) for _ in range(4)]
            if self._held(nums) != heldout:
                continue
            slots = rng.sample(SLOTS, 4)
            shape, it = rng.choice(SHAPES), iter(slots)
            expr = "".join(next(it) if s == "N" else rng.choice(OPS) for s in shape)
            v = evaluate(expr, nums)
            if v is not None and 1 <= v <= self.max_target:
                return nums, v, expr

    def _rng(self, generator: torch.Generator | None) -> random.Random:
        return random.Random(int(torch.randint(0, 2**62, (1,), generator=generator)))

    def _question(self, nums: list[int], target: int) -> str:
        return "".join(f"{n:02d}" for n in nums) + f"{target:03d}"

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Training questions with the expression that generated them: [B, 18]."""
        rng = self._rng(generator)
        rows = []
        for _ in range(batch_size):
            nums, v, expr = self._draw(rng, False)
            rows.append(self.encode(self._question(nums, v) + expr))
        return torch.stack(rows)

    def sample_questions(self, batch_size: int, heldout: bool = False,
                         generator: torch.Generator | None = None) -> torch.Tensor:
        """Questions [B, 11] from training (heldout=False) or held-out number sets."""
        if heldout and self.holdout_frac <= 0:
            raise ValueError("no held-out questions: set holdout_frac > 0")
        rng = self._rng(generator)
        return torch.stack([self.encode(self._question(*self._draw(rng, heldout)[:2])) for _ in range(batch_size)])

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.chars.index(c) for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str | list[str]:
        if ids.dim() == 1:
            return "".join(self.chars[i] for i in ids.tolist())
        return ["".join(self.chars[i] for i in row) for row in ids.tolist()]
