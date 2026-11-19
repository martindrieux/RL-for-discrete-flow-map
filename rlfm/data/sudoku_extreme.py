"""Sudoku-Extreme (released with the Hierarchical Reasoning Model, huggingface.co/datasets/sapientinc/sudoku-extreme)
in the setting of Flow Reasoning Models (arXiv 2606.29150, Appendix A.1): per training seed, 1,000 puzzles of the
official training split; a fixed validation set of 1,000 other training-split puzzles; a fixed 1,000-puzzle subset
of the official test split; score = fraction of test grids fully right.

Training and validation sets are stratified by difficulty (the paper says "difficulty-balanced" without details): the
training split is sorted by rating (ties in random order) and cut into 10 equal parts, and each set takes n / 10
puzzles from every part, so it follows the split's rating distribution. Test puzzles are drawn uniformly.
The raw CSVs (source, question with "." for a blank, answer, rating) stay outside the repository; this writes
data/sudoku/extreme.npz (rlfm.data.sudoku.EXTREME_PATH), read by SudokuTask(source="extreme", subset=k).

python -m rlfm.data.sudoku_extreme ~/datasets/sudoku-extreme/train.csv ~/datasets/sudoku-extreme/test.csv
"""
import csv
import sys

import numpy as np

from . import sudoku as S

N_BINS = 10


def _ratings(path: str) -> np.ndarray:
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        return np.array([int(r[3]) for r in rd], dtype=np.int64)


def _rows(path: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Puzzles and solutions of the file rows idx (distinct, counted after the header): uint8 [len(idx), 81], 0 = blank."""
    pos = {int(i): k for k, i in enumerate(idx)}
    q, s = np.zeros((len(idx), 81), np.uint8), np.zeros((len(idx), 81), np.uint8)
    with open(path) as f:
        rd = csv.reader(f)
        next(rd)
        for j, r in enumerate(rd):
            k = pos.get(j)
            if k is not None:
                q[k] = [0 if c == "." else int(c) for c in r[1]]
                s[k] = [int(c) for c in r[2]]
    return q, s


def stratified(ratings: np.ndarray, n: int, rng: np.random.Generator, exclude=()) -> np.ndarray:
    """n distinct indices outside `exclude`: ratings sorted with random tie-breaking, cut into N_BINS equal parts,
    n / N_BINS drawn from each part."""
    keep = np.ones(len(ratings), bool)
    keep[np.asarray(exclude, dtype=np.int64)] = False
    order = np.lexsort((rng.random(len(ratings)), ratings))
    parts = [p[keep[p]] for p in np.array_split(order, N_BINS)]
    return np.concatenate([rng.choice(p, n // N_BINS, replace=False) for p in parts])


def build(train_csv: str, test_csv: str, out: str | None = None, n: int = 1000, n_subsets: int = 5,
          seed: int = 0) -> str:
    """Validation set, n_subsets training sets (one per training seed, each disjoint from validation) and test set."""
    out = out or S.EXTREME_PATH
    rng = np.random.default_rng(seed)
    r_tr, r_te = _ratings(train_csv), _ratings(test_csv)
    val = stratified(r_tr, n, rng)
    subsets = np.stack([stratified(r_tr, n, rng, exclude=val) for _ in range(n_subsets)])
    test = rng.choice(len(r_te), n, replace=False)
    rows = np.unique(np.concatenate([val, subsets.ravel()]))
    q, s = _rows(train_csv, rows)
    at = np.searchsorted(rows, np.concatenate([val, subsets.ravel()]))
    qv, sv, qs, ss = q[at[:n]], s[at[:n]], q[at[n:]].reshape(n_subsets, n, 81), s[at[n:]].reshape(n_subsets, n, 81)
    qt, st = _rows(test_csv, test)
    for a, b in ((qv, sv), (qs.reshape(-1, 81), ss.reshape(-1, 81)), (qt, st)):  # valid grids that keep the clues
        assert all(S.is_solution(x.tolist()) for x in b) and ((a == 0) | (a == b)).all()
    np.savez_compressed(out, train_puzzle=qs, train_solution=ss, train_rating=r_tr[subsets], train_index=subsets,
                        val_puzzle=qv, val_solution=sv, val_rating=r_tr[val], val_index=val,
                        test_puzzle=qt, test_solution=st, test_rating=r_te[test], test_index=test)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(build(sys.argv[1], sys.argv[2]), flush=True)
