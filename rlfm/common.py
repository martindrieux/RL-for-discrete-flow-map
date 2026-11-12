"""Config loading, model construction, checkpoints and task evaluation shared by the scripts."""
import copy
import sys

import torch
import torch.nn.functional as F
import yaml

from .data.arithmetic import ArithmeticTask
from .flowmap import FlowMap, sample
from .model import DiT
from .rewards import arithmetic as arith_reward
from .timewarp import TimeWarp


def load_config(argv: list[str] | None = None) -> dict:
    """python script.py config.yaml key.sub=value ...  (values parsed as YAML)."""
    argv = sys.argv[1:] if argv is None else argv
    with open(argv[0]) as f:
        cfg = yaml.safe_load(f)
    for item in argv[1:]:
        key, val = item.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        v = yaml.safe_load(val)
        if isinstance(v, str):
            try:
                v = float(v)  # YAML 1.1 reads "1e-5" as a string
            except ValueError:
                pass
        node[last] = v
    return cfg


def make_task(cfg: dict):
    if cfg["task"]["name"] == "arithmetic":
        tc = cfg["task"]
        task = ArithmeticTask(n_digits=tc.get("n_digits", 2), frac_correct=tc.get("frac_correct", 0.5),
                              holdout_frac=tc.get("holdout_frac", 0.0), holdout_seed=tc.get("holdout_seed", 0))
        return task, arith_reward
    raise ValueError(cfg["task"]["name"])


def require_holdout(cfg: dict, allow: bool, what: str) -> None:
    """Every experiment keeps a held-out set; refuse to run without one unless explicitly allowed."""
    if cfg["task"].get("holdout_frac", 0.0) <= 0 and not allow:
        raise SystemExit(f"{what}: task.holdout_frac must be > 0 (a held-out set is required). "
                         "Pass allow_no_holdout=true only for debugging.")


def make_flowmap(cfg: dict, V: int, L: int) -> FlowMap:
    m = cfg["model"]
    net = DiT(V, L, d=m["dim"], depth=m["depth"], heads=m["heads"])
    return FlowMap(net, TimeWarp(V, identity=not m.get("time_warp", True)))


def load_flowmap(path: str, device) -> tuple[FlowMap, dict]:
    """Loads the EMA weights of a pretraining checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    task, _ = make_task(ck["cfg"])
    fm = make_flowmap(ck["cfg"], task.vocab_size, task.seq_len).to(device)
    fm.load_state_dict(ck["ema"])
    return fm, ck["cfg"]


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay, self.model = decay, copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.lerp_(p, 1 - self.decay)


def one_hot(y: torch.Tensor, V: int) -> torch.Tensor:
    return F.one_hot(y, V).float()


def task_metrics(task, reward_mod, y: torch.Tensor, prefix: str) -> dict:
    """Fraction well formed, fraction correct, number of distinct correct strings."""
    s = task.decode(y.cpu())
    m = {f"{prefix}wellformed": reward_mod.well_formed(s).float().mean().item(),
         f"{prefix}correct": reward_mod.correct(s).float().mean().item(),
         f"{prefix}distinct_correct": reward_mod.distinct_correct(s)}
    if getattr(task, "heldout_mask", None) is not None and task.heldout_mask.any():
        parsed = [reward_mod.parse(x) for x in s]
        held = [p for p in parsed if p is not None and task.is_heldout(p[0], p[1])]
        m[f"{prefix}heldout_share"] = len(held) / len(s)       # share of samples whose operands are held out
        if held:
            m[f"{prefix}heldout_correct"] = sum(p[0] + p[1] == p[2] for p in held) / len(held)
    return m


@torch.no_grad()
def eval_sampler(fm: FlowMap, task, reward_mod, Ks: list[int], n: int, seed: int, device) -> dict:
    """Argmax-decoded samples of the K-jump sampler from a fixed set of noises."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, task.seq_len, task.vocab_size, generator=g).to(device)
    out = {}
    for K in Ks:
        y = sample(fm, z, K).argmax(-1)
        out.update(task_metrics(task, reward_mod, y, f"K{K}_"))
    return out
