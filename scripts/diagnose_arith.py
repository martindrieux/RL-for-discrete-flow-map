"""Did pretraining work? Checks a pretrained arithmetic flow map against the data distribution.

python scripts/diagnose_arith.py runs/arith_pretrain/last.pt Ks=[1,8,32,128] n=4000

For each number of jumps K (argmax decoding, fixed noises):
  wellformed, correct (data: q), distinct correct (data with n samples: ~min(n q, 10^4)),
  operand-digit marginal: total variation to uniform,
  sum field given operands: P(correct) should be q; wrong sums should look uniform on [0, 199).
Also the same numbers for n fresh data samples, as the reference row.
"""
import sys

import torch
import yaml

from rlfm.common import load_flowmap, make_task
from rlfm.flowmap import sample
from rlfm.rewards.arithmetic import parse


def stats(strings: list[str], n_digits: int) -> dict:
    parsed = [parse(s) for s in strings]
    ok = [p for p in parsed if p is not None]
    correct = [p for p in ok if p[0] + p[1] == p[2]]
    wrong = [p for p in ok if p[0] + p[1] != p[2]]
    digits = torch.tensor([int(ch) for p in ok for ch in f"{p[0]:0{n_digits}d}{p[1]:0{n_digits}d}"], dtype=torch.long)
    tv_digits = 0.5 * (torch.bincount(digits, minlength=10).float() / max(len(digits), 1) - 0.1).abs().sum().item()
    hi = 2 * 10**n_digits - 1
    wsum = torch.tensor([p[2] for p in wrong]) if wrong else torch.zeros(0, dtype=torch.long)
    hist = torch.bincount(wsum.clamp(max=hi), minlength=hi + 1)[:hi].float()
    tv_wrong = 0.5 * (hist / max(len(wrong), 1) - 1 / hi).abs().sum().item() if wrong else float("nan")
    return {"wellformed": len(ok) / len(strings), "correct": len(correct) / len(strings),
            "distinct_correct": len({p for p in correct}), "tv_operand_digits": round(tv_digits, 4),
            "tv_wrong_sums_vs_uniform": round(tv_wrong, 4), "distinct_strings": len(set(strings))}


@torch.no_grad()
def main():
    a = {"Ks": [1, 8, 32, 128], "n": 4000, "seed": 1234}
    a.update({k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in sys.argv[2:])})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm, cfg = load_flowmap(sys.argv[1], device)
    task, _ = make_task(cfg)
    nd = cfg["task"].get("n_digits", 2)
    out = {"data": stats(task.decode(task.sample(a["n"], torch.Generator().manual_seed(a["seed"]))), nd)}
    z = torch.randn(a["n"], task.seq_len, task.vocab_size, generator=torch.Generator().manual_seed(a["seed"])).to(device)
    for K in a["Ks"]:
        y = torch.cat([sample(fm, zb, K).argmax(-1) for zb in z.split(1000)])
        s = task.decode(y.cpu())
        out[f"K={K}"] = stats(s, nd)
        out[f"K={K}"]["examples"] = s[:6]
    print(yaml.safe_dump(out, sort_keys=False, width=200))


if __name__ == "__main__":
    main()
