"""Reference points without training: the pretrained policy, and best-of-G selection from it.

python scripts/eval_baselines.py runs/pretrain/last.pt K=1 temp=1.0 G=8 n=1000 seed=1234
"""
import sys

import torch
import yaml

from rlfm.common import load_flowmap, make_task
from rlfm.policy import policy_logits, sample_tokens, token_entropy


@torch.no_grad()
def main():
    path = sys.argv[1]
    a = {"K": 1, "temp": 1.0, "G": 8, "n": 1000, "seed": 1234, "option": "A", "tau_star": 0.9}
    a.update({k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in sys.argv[2:])})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm, cfg = load_flowmap(path, device)
    task, rm = make_task(cfg)
    V, L, G, n = task.vocab_size, task.seq_len, a["G"], a["n"]
    gz = torch.Generator().manual_seed(a["seed"])
    z = torch.randn(n * G, L, V, generator=gz).to(device)
    logits = policy_logits(fm, z, a["K"], a["option"], a["tau_star"])
    y = sample_tokens(logits, a["temp"], torch.Generator(device=device).manual_seed(a["seed"]))
    s = task.decode(y.cpu())
    r = rm.reward(s).view(n, G)
    first = [s[i * G] for i in range(n)]
    print(yaml.safe_dump({
        "args": a,
        "pretrained_sample_correct": r[:, 0].mean().item(),
        "pretrained_sample_wellformed": rm.well_formed(first).float().mean().item(),
        "pretrained_sample_distinct_correct": rm.distinct_correct(first),
        "pretrained_argmax_correct": rm.correct(task.decode(logits.argmax(-1)[::G].cpu())).float().mean().item(),
        f"best_of_{G}_correct": r.max(1).values.mean().item(),
        "entropy": token_entropy(logits, a["temp"]).mean().item(),
    }, sort_keys=False))


if __name__ == "__main__":
    main()
