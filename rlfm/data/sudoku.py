"""9x9 Sudoku: given a puzzle with blank cells, write the full solved grid.

Tokens: question = the 81 cells row by row, digits 1..9 and 0 for a blank; answer = the 81 cells of the solution.
Every puzzle has a unique solution. Puzzles come from two pools of solved grids built once by
`python -m rlfm.data.sudoku build` (deterministic, one seed per grid): for each grid, cells are blanked one at a time in
a random order, keeping a blank only if the solution stays unique; R = the blanked cells. Any subset of R keeps the
solution unique (more givens only remove solutions), so a puzzle blanks b cells of R, b ~ U{min_blanks, ..., min(
max_blanks, |R|)}. Each drawn grid gets a random symmetry of Sudoku (digit relabelling, row permutations within bands,
band permutations, the same for columns, transpose), which keeps validity and uniqueness.
Held-out puzzles come from a separate pool of grids (other seeds), never used in training.
"""
import os
import random
import sys

import numpy as np
import torch

CHARS = "0123456789"
ROW = [i // 9 for i in range(81)]
COL = [i % 9 for i in range(81)]
BOX = [(i // 27) * 3 + (i % 9) // 3 for i in range(81)]
UNITS = ([[r * 9 + c for c in range(9)] for r in range(9)] + [[r * 9 + c for r in range(9)] for c in range(9)]
         + [[(b // 3) * 27 + (b % 3) * 3 + r * 9 + c for r in range(3) for c in range(3)] for b in range(9)])
FULL = 0b1111111110  # bits 1..9
POOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "sudoku")
SEED_OFFSET = {"train": 0, "heldout": 1_000_000_000}


def solve(grid: list[int], limit: int = 2, rng: random.Random | None = None) -> tuple[int, list[int] | None]:
    """(number of solutions, counted up to limit; the first solution found). Depth-first search on the cell with
    the fewest candidates; candidates tried in random order if rng is given."""
    g = list(grid)
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(g):
        if v:
            bit = 1 << v
            if (rows[ROW[i]] | cols[COL[i]] | boxes[BOX[i]]) & bit:
                return 0, None
            rows[ROW[i]] |= bit
            cols[COL[i]] |= bit
            boxes[BOX[i]] |= bit
    empties = [i for i in range(81) if g[i] == 0]
    found: list = [0, None]

    def rec(k: int) -> bool:
        if k == len(empties):
            found[0] += 1
            if found[1] is None:
                found[1] = list(g)
            return found[0] >= limit
        best_j, best_m, best_n = k, 0, 10
        for j in range(k, len(empties)):
            i = empties[j]
            m = FULL & ~(rows[ROW[i]] | cols[COL[i]] | boxes[BOX[i]])
            n = bin(m).count("1")
            if n < best_n:
                best_j, best_m, best_n = j, m, n
                if n <= 1:
                    break
        if best_n == 0:
            return False
        empties[k], empties[best_j] = empties[best_j], empties[k]
        i = empties[k]
        r, c, b = ROW[i], COL[i], BOX[i]
        digits = [d for d in range(1, 10) if best_m >> d & 1]
        if rng is not None:
            rng.shuffle(digits)
        for d in digits:
            bit = 1 << d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            g[i] = d
            stop = rec(k + 1)
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            g[i] = 0
            if stop:
                return True
        return False

    rec(0)
    return found[0], found[1]


def make_grid(seed: int) -> tuple[list[int], list[bool]]:
    """(random solved grid, removable mask R): blank cells in a random order, keeping a blank if still unique."""
    rng = random.Random(seed)
    _, sol = solve([0] * 81, limit=1, rng=rng)
    g = list(sol)
    for i in rng.sample(range(81), 81):
        v, g[i] = g[i], 0
        if solve(g, limit=2)[0] != 1:
            g[i] = v
    return sol, [x == 0 for x in g]


def is_solution(a: list[int]) -> bool:
    return all(sorted(a[i] for i in u) == list(range(1, 10)) for u in UNITS)


def pool_path(split: str, n: int) -> str:
    return os.path.join(POOL_DIR, f"{split}_{n}.npz")


def build_pool(split: str, n: int, workers: int | None = None) -> str:
    """Grids with seeds SEED_OFFSET[split] + 0..n-1, saved as solution uint8 [n, 81] and removable bool [n, 81]."""
    from multiprocessing import Pool
    seeds = [SEED_OFFSET[split] + k for k in range(n)]
    with Pool(workers or os.cpu_count()) as p:
        out = p.map(make_grid, seeds, chunksize=64)
    os.makedirs(POOL_DIR, exist_ok=True)
    path = pool_path(split, n)
    np.savez_compressed(path, solution=np.array([s for s, _ in out], dtype=np.uint8),
                        removable=np.array([r for _, r in out], dtype=bool))
    return path


def _symmetries(rng: np.random.Generator, B: int) -> tuple[np.ndarray, np.ndarray]:
    """B random Sudoku symmetries: cell map [B, 81] (new cell i takes old cell idx[i]) and digit map [B, 10] (0 -> 0)."""
    def lines():  # [B, 9]: band (or stack) order, then order within each band
        bands = np.argsort(rng.random((B, 3)), 1)
        within = np.argsort(rng.random((B, 3, 3)), 2)
        return (bands[:, :, None] * 3 + np.take_along_axis(within, bands[:, :, None].repeat(3, 2), 1)).reshape(B, 9)
    r, c = lines(), lines()
    idx = r[:, :, None] * 9 + c[:, None, :]                                  # [B, 9, 9]
    t = rng.random(B) < 0.5
    idx[t] = idx[t].transpose(0, 2, 1)
    digits = np.concatenate([np.zeros((B, 1), dtype=np.int64), 1 + np.argsort(rng.random((B, 9)), 1)], 1)
    return idx.reshape(B, 81), digits


class SudokuTask:
    def __init__(self, n_train: int = 50_000, n_heldout: int = 1000, min_blanks: int = 30, max_blanks: int = 81):
        self.n_train, self.n_heldout = n_train, n_heldout
        self.min_blanks, self.max_blanks = min_blanks, max_blanks
        self.chars = CHARS
        self.vocab_size = len(CHARS)
        self.prompt_len, self.answer_len = 81, 81
        self.seq_len = self.prompt_len + self.answer_len
        self._pools = {}

    def pool(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split not in self._pools:
            path = pool_path(split, self.n_train if split == "train" else self.n_heldout)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} missing: run `python -m rlfm.data.sudoku build`")
            d = np.load(path)
            self._pools[split] = (d["solution"].astype(np.int64), d["removable"])
        return self._pools[split]

    def _draw(self, B: int, heldout: bool, generator: torch.Generator | None, distinct: bool = False):
        """(puzzles, solutions), each [B, 81] int64; distinct: different grids when B <= pool size."""
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,), generator=generator)))
        sol, rem = self.pool("heldout" if heldout else "train")
        N = len(sol)
        i = rng.permutation(N)[:B] if distinct and B <= N else rng.integers(0, N, B)
        idx, digits = _symmetries(rng, B)
        s = np.take_along_axis(digits, np.take_along_axis(sol[i], idx, 1), 1)
        r = np.take_along_axis(rem[i], idx, 1)
        hi = np.minimum(self.max_blanks, r.sum(1))
        b = rng.integers(np.minimum(self.min_blanks, hi), hi + 1)          # blanks per puzzle
        score = np.where(r, rng.random((B, 81)), -1.0)
        rank = np.argsort(np.argsort(-score, 1), 1)                          # 0 = first removable cell to blank
        q = np.where(rank < b[:, None], 0, s)
        return torch.from_numpy(q), torch.from_numpy(s)

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Training puzzles with their solutions: [B, 162]."""
        q, s = self._draw(batch_size, False, generator)
        return torch.cat([q, s], 1)

    def sample_questions(self, batch_size: int, heldout: bool = False,
                         generator: torch.Generator | None = None) -> torch.Tensor:
        """Puzzles [B, 81] from training grids (heldout=False) or held-out grids (distinct grids when possible)."""
        return self._draw(batch_size, heldout, generator, distinct=heldout)[0]

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.chars.index(c) for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str | list[str]:
        if ids.dim() == 1:
            return "".join(self.chars[i] for i in ids.tolist())
        return ["".join(self.chars[i] for i in row) for row in ids.tolist()]


if __name__ == "__main__":  # python -m rlfm.data.sudoku build [n_train] [n_heldout]
    if sys.argv[1:2] != ["build"]:
        raise SystemExit(__doc__)
    n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    n_ho = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    for split, n in (("heldout", n_ho), ("train", n_tr)):
        print(build_pool(split, n), flush=True)
