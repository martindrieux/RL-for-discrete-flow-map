"""Rewards for generated `a+b=c` strings."""

import re

import torch


def parse_op(s: str) -> tuple[int, int, int, str] | None:
    """(a, b, c, op) if s is `n digits, + or *, n digits, =, answer` with n + 1 (for +) or 2n (for *) answer digits."""
    m = re.fullmatch(r"([0-9]+)([+*])([0-9]+)=([0-9]+)", s)
    if m is None:
        return None
    n, op = len(m[1]), m[2]
    if len(m[3]) != n or len(m[4]) != (n + 1 if op == "+" else 2 * n):
        return None
    return int(m[1]), int(m[3]), int(m[4]), op


def parse(s: str) -> tuple[int, int, int] | None:
    """(a, b, c) of a well-formed equation, else None."""
    p = parse_op(s)
    return None if p is None else p[:3]


def well_formed(strings: list[str]) -> torch.Tensor:
    """BoolTensor [B]: whether each string parses."""
    return torch.tensor([parse(s) is not None for s in strings], dtype=torch.bool)


def _is_correct(s: str) -> bool:
    """Whether s parses and its sum is right."""
    p = parse_op(s)
    return p is not None and (p[0] + p[1] if p[3] == "+" else p[0] * p[1]) == p[2]


def correct(strings: list[str]) -> torch.Tensor:
    """BoolTensor [B]: well formed and a + b == c."""
    return torch.tensor([_is_correct(s) for s in strings], dtype=torch.bool)


def reward(strings: list[str]) -> torch.Tensor:
    """FloatTensor [B]: 1.0 for correct equations, 0.0 otherwise."""
    return correct(strings).float()


def distinct_correct(strings: list[str]) -> int:
    """Number of distinct correct strings."""
    return len({s for s in strings if _is_correct(s)})


def digit_fraction(y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """FloatTensor [B]: fraction of answer digits equal to the true answer's, y and target [B, L_a]."""
    return digit_correct(y, target).mean(1)


def digit_correct(y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """FloatTensor [B, L_a]: 1.0 where an answer digit equals the true answer's digit."""
    return (y == target).float()
