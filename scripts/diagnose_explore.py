"""Can RL find the correct digits? Probability the policy gives the true digit where it is wrong.

python scripts/diagnose_explore.py runs/ckpt_mul4_100k.pt [runs/rl_x/last.pt ...] Ks=[1,4] temps=[1,2] n=2000 G=8

For each checkpoint (pretraining or RL), number of jumps K, temperature T, on n held-out questions with fixed noises:
  digit accuracy of the argmax, and per answer position;
  over digits whose argmax is wrong: quantiles of p*, the probability of the true digit, and
  P(true digit appears in at least one of G samples) = 1 - (1 - p*)^G, averaged, plus the fraction of wrong digits
  where it is below 5%;
  the same averaged over all digits.
"""
import sys

import torch
import yaml

from rlfm.common import make_flowmap, make_task
from rlfm.policy import policy_logits


def load_any(path: str, device):
    """EMA weights of a pretraining checkpoint, or the weights of an RL checkpoint (with its pretraining config)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("pre_cfg", ck["cfg"])
    task, _ = make_task(cfg)
    fm = make_flowmap(cfg, task.vocab_size, task.seq_len, task).to(device)
    fm.load_state_dict(ck["ema"])
    return fm.eval(), task


@torch.no_grad()
def main():
    paths = [s for s in sys.argv[1:] if "=" not in s]
    a = {"Ks": [1, 4], "temps": [1.0, 2.0], "n": 2000, "G": 8, "seed": 1234, "heldout": True}
    a.update({k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in sys.argv[1:] if "=" in s)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in paths:
        fm, task = load_any(path, device)
        g = torch.Generator().manual_seed(a["seed"])
        q = task.sample_questions(a["n"], heldout=a["heldout"], generator=g).to(device)
        La, V = task.answer_len, task.vocab_size
        z = torch.randn(a["n"], La, V, generator=g).to(device)
        target = task.true_answer(q)
        print(f"== {path}  ({'held-out' if a['heldout'] else 'training'} questions, n={a['n']}, G={a['G']})")
        for K in a["Ks"]:
            logits = policy_logits(fm, z, K, "A", 0.9, q)
            wrong = logits.argmax(-1) != target
            pos = (~wrong).float().mean(0)
            print(f"K={K}: argmax digit acc {1 - wrong.float().mean().item():.3f}, exact {(~wrong).all(1).float().mean().item():.3f}; "
                  f"per position (most significant first): {' '.join(f'{x:.2f}' for x in pos.tolist())}")
            for T in a["temps"]:
                p = torch.softmax(logits / T, -1).gather(-1, target[..., None]).squeeze(-1)  # p* [n, La]
                hit = 1 - (1 - p) ** a["G"]
                pw, hw = p[wrong], hit[wrong]
                qs = torch.quantile(pw.float().cpu(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])) if pw.numel() else torch.zeros(5)
                print(f"  T={T}: wrong digits {pw.numel()}: p* quantiles 10/25/50/75/90% = "
                      f"{' '.join(f'{x:.1e}' for x in qs.tolist())}; "
                      f"P(true digit in {a['G']} samples) mean {hw.mean().item():.3f}, below 5% for {(hw < 0.05).float().mean().item():.3f} | "
                      f"all digits: mean p* {p.mean().item():.3f}, P(in {a['G']}) {hit.mean().item():.3f}, "
                      f"P(whole answer in {a['G']}) {(1 - (1 - p.prod(1)) ** a['G']).mean().item():.3f}")


if __name__ == "__main__":
    main()
