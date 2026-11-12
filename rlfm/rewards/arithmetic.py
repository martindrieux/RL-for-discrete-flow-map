"""Rewards for generated `a+b=c` strings."""

import re

import torch


def parse(s: str) -> tuple[int, int, int] | None:
    """Return (a, b, c) if s is exactly `n digits + n digits = n+1 digits`, else None."""
    if len(s) < 6 or len(s) % 3 != 0:
        return None
    n = len(s) // 3 - 1
    m = re.fullmatch(rf"([0-9]{{{n}}})\+([0-9]{{{n}}})=([0-9]{{{n + 1}}})", s)
    return None if m is None else (int(m[1]), int(m[2]), int(m[3]))


def well_formed(strings: list[str]) -> torch.Tensor:
    """BoolTensor [B]: whether each string parses."""
    return torch.tensor([parse(s) is not None for s in strings], dtype=torch.bool)


def _is_correct(s: str) -> bool:
    """Whether s parses and its sum is right."""
    p = parse(s)
    return p is not None and p[0] + p[1] == p[2]


def correct(strings: list[str]) -> torch.Tensor:
    """BoolTensor [B]: well formed and a + b == c."""
    return torch.tensor([_is_correct(s) for s in strings], dtype=torch.bool)


def reward(strings: list[str]) -> torch.Tensor:
    """FloatTensor [B]: 1.0 for correct equations, 0.0 otherwise."""
    return correct(strings).float()


def distinct_correct(strings: list[str]) -> int:
    """Number of distinct correct strings."""
    return len({s for s in strings if _is_correct(s)})
