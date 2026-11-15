"""Rewards for Countdown strings: 11 question digits (four 2-digit numbers, 3-digit target) + 7 answer tokens."""
import torch

from ..data.countdown import SLOTS, evaluate


def parse(s: str) -> tuple[list[int], int, str] | None:
    """(numbers, target, expression) if the question part is digits, else None."""
    q, e = s[:11], s[11:]
    if len(s) != 18 or not q.isdigit():
        return None
    return [int(q[i:i + 2]) for i in range(0, 8, 2)], int(q[8:]), e


def _value(s: str) -> int | None:
    p = parse(s)
    return None if p is None else evaluate(p[2], p[0])


def well_formed(strings: list[str]) -> torch.Tensor:
    """BoolTensor [B]: the answer is a valid postfix expression using each number once (any value)."""
    return torch.tensor([_value(s) is not None for s in strings], dtype=torch.bool)


def correct(strings: list[str]) -> torch.Tensor:
    """BoolTensor [B]: well formed and equal to the target."""
    out = []
    for s in strings:
        p = parse(s)
        out.append(p is not None and evaluate(p[2], p[0]) == p[1])
    return torch.tensor(out, dtype=torch.bool)


def reward(strings: list[str]) -> torch.Tensor:
    """FloatTensor [B]: 1.0 for a correct expression, else 0."""
    return correct(strings).float()


def distinct_correct(strings: list[str]) -> int:
    return len({s for s, c in zip(strings, correct(strings).tolist()) if c})
