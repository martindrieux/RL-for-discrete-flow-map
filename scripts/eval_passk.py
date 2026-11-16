"""Does RL find new solutions or sharpen the pretrained distribution? pass@k on held-out Countdown or Sudoku questions.

For n held-out questions, S answers are sampled per question (flow map: z ~ N(0, I), option A with K jumps, T = 1;
autoregressive: ancestral sampling at T = 1). With c correct answers out of S, the unbiased estimate is
pass@k = 1 - C(S - c, k) / C(S, k). Also reported: mean number of distinct correct answer strings per question
(Countdown: also distinct canonical expressions, see canonical; Sudoku: pass@1 by number of blanks), and
for the autoregressive model the greedy answer's accuracy.

Mode-seeking decodings, to compare RL with what decoding alone gives: zs=s scales the flow map's noise,
z ~ N(0, s^2 I) (s = 0: one deterministic answer per question); T=t is the sampling temperature of both models.

beam=w also reports the accuracy of the best of w beams of the autoregressive model (w = 1 is greedy decoding).

python scripts/eval_passk.py fm runs/ckpt_cd_fm_100k.pt [K=4] [n=500] [S=64] [zs=1] [T=1]
python scripts/eval_passk.py ar runs/ckpt_cd_ar_40k.pt beam=8
python scripts/eval_passk.py ar runs/ckpt_cd_ar_100k.pt
"""
import json
import math
import sys

import torch

from rlfm.ar import ARNet, ar_sample
from rlfm.common import make_flowmap, make_task
from rlfm.data.countdown import SLOTS
from rlfm.rewards.countdown import parse
from rlfm.policy import policy_logits, sample_tokens


def canonical(expr: str, nums: list[int]) -> str:
    """Expression tree with numbers as values, + and * chains flattened and their operands sorted: A+B, B+A and
    (A+B)+C, A+(B+C) get the same string; - and / keep their order."""
    st = []
    for c in expr:
        if c in SLOTS:
            st.append(("n", str(nums[SLOTS.index(c)])))
        else:
            b, a = st.pop(), st.pop()
            if c in "+*":
                kids = [t for x in (a, b) for t in (x[1] if x[0] == c else [x])]
                st.append((c, kids))
            else:
                st.append((c, [a, b]))

    def show(t):
        if t[0] == "n":
            return t[1]
        kids = [show(x) for x in t[1]]
        return t[0] + "(" + ",".join(sorted(kids) if t[0] in "+*" else kids) + ")"
    return show(st[0])


@torch.no_grad()
def ar_beam(net: ARNet, q: torch.Tensor, La: int, width: int) -> torch.Tensor:
    """Beam search on sum_i log p(y_i | c, y_<i): the highest-scoring answer among `width` beams, [N, La]."""
    N, Lq = q.shape
    seqs, scores = q[:, None, :], torch.zeros(N, 1, device=q.device)
    for _ in range(La):
        nb = seqs.shape[1]
        logp = torch.log_softmax(net(seqs.reshape(N * nb, -1))[:, -1], -1).view(N, nb, -1)
        V = logp.shape[-1]
        scores, idx = (scores[:, :, None] + logp).view(N, -1).topk(min(width, nb * V), -1)
        beam, tok = idx // V, idx % V
        seqs = torch.cat([seqs.gather(1, beam[:, :, None].expand(-1, -1, seqs.shape[2])), tok[:, :, None]], 2)
    return seqs[:, 0, Lq:]


def pass_at_k(c: int, S: int, k: int) -> float:
    return 1.0 if S - c < k else 1.0 - math.comb(S - c, k) / math.comb(S, k)


@torch.no_grad()
def main():
    kind, path = sys.argv[1], sys.argv[2]
    kw = dict(a.split("=") for a in sys.argv[3:])
    K, n, S, chunk = int(kw.get("K", 4)), int(kw.get("n", 500)), int(kw.get("S", 64)), int(kw.get("chunk", 50))
    zs, T = float(kw.get("zs", 1.0)), float(kw.get("T", 1.0))
    beam = int(kw.get("beam", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    pre_cfg = ck.get("pre_cfg", ck["cfg"])
    task, rew = make_task(pre_cfg)
    La = task.seq_len - task.prompt_len
    if kind == "fm":
        net = make_flowmap(pre_cfg, task.vocab_size, task.seq_len, task).to(device)
    else:
        m = pre_cfg["model"]
        net = ARNet(task.vocab_size, d=m["dim"], depth=m["depth"], heads=m["heads"]).to(device)
    net.load_state_dict(ck["ema"])
    net.eval()
    q_all = task.sample_questions(n, heldout=True, generator=torch.Generator().manual_seed(4321)).to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    countdown = pre_cfg["task"]["name"] == "countdown"
    counts, distinct, canon, greedy, beamed = [], [], [], [], []
    for i in range(0, n, chunk):
        q = q_all[i:i + chunk]
        qr = q.repeat_interleave(S, 0)
        if kind == "fm":
            z = zs * torch.randn(qr.shape[0], La, task.vocab_size, device=device, generator=gen)
            logits = policy_logits(net, z, K, "A", 0.9, qr)
            y = logits.argmax(-1) if T == 0 else sample_tokens(logits, T, gen)
        else:
            y = ar_sample(net, qr, La, T, gen=gen) if T > 0 else ar_sample(net, qr, La, greedy=True)
            yg = ar_sample(net, q, La, greedy=True)
            greedy += rew.correct(task.decode(torch.cat([q, yg], 1).cpu())).tolist()
            if beam:
                yb = ar_beam(net, q, La, beam)
                beamed += rew.correct(task.decode(torch.cat([q, yb], 1).cpu())).tolist()
        s = task.decode(torch.cat([qr, y], 1).cpu())
        c = rew.correct(s).view(-1, S)
        counts += c.sum(1).tolist()
        for j in range(q.shape[0]):
            ok = [s[j * S + t] for t in range(S) if c[j, t]]
            distinct.append(len(set(ok)))
            if countdown:
                canon.append(len({canonical(parse(x)[2], parse(x)[0]) for x in ok}))
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= S]
    out = {"kind": kind, "ckpt": path, "K": K if kind == "fm" else None, "n": n, "S": S, "zs": zs, "T": T,
           **{f"pass@{k}": sum(pass_at_k(c, S, k) for c in counts) / n for k in ks},
           "distinct_correct": sum(distinct) / n}
    if countdown:
        out["distinct_canonical"] = sum(canon) / n
    else:  # Sudoku: pass@1 by number of blanks
        blanks = (q_all == 0).sum(1).tolist()
        for lo, hi in ((30, 39), (40, 49), (50, 81)):
            sel = [c for c, b in zip(counts, blanks) if lo <= b <= hi]
            if sel:
                out[f"pass@1_blanks{lo}-{hi}"] = sum(sel) / (S * len(sel))
    if greedy:
        out["greedy"] = sum(greedy) / n
    if beamed:
        out[f"beam{beam}"] = sum(beamed) / n
    print(json.dumps(out))


if __name__ == "__main__":
    main()
