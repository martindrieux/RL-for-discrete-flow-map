"""Rewards for Sudoku strings: 81 puzzle cells (0 = blank) + 81 answer cells."""
import torch

from ..data.sudoku import UNITS


def parse(s: str) -> tuple[str, str] | None:
    """(puzzle, answer) if the string has the right length, else None."""
    return (s[:81], s[81:]) if len(s) == 162 else None


def _keeps_givens(q: str, a: str) -> bool:
    return all(c in "123456789" for c in a) and all(x == "0" or x == y for x, y in zip(q, a))


def _correct(s: str) -> bool:
    p = parse(s)
    if p is None or not _keeps_givens(*p):
        return False
    a = p[1]
    return all(len({a[i] for i in u}) == 9 for u in UNITS)


def well_formed(strings: list[str]) -> torch.Tensor:
    """BoolTensor [B]: the answer uses digits 1..9 and keeps every given digit of the puzzle (rules not checked)."""
    return torch.tensor([(p := parse(s)) is not None and _keeps_givens(*p) for s in strings], dtype=torch.bool)


def correct(strings: list[str]) -> torch.Tensor:
    """BoolTensor [B]: keeps the givens and every row, column and 3x3 box holds 1..9 (the unique solution)."""
    return torch.tensor([_correct(s) for s in strings], dtype=torch.bool)


def reward(strings: list[str]) -> torch.Tensor:
    """FloatTensor [B]: 1.0 for the solved grid, else 0."""
    return correct(strings).float()


def distinct_correct(strings: list[str]) -> int:
    return len({s for s, c in zip(strings, correct(strings).tolist()) if c})
