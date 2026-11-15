"""Autoregressive baseline on the same tasks: pretraining (next-token cross-entropy), supervised fine-tuning, RL.

python scripts/ar.py configs/countdown_pretrain.yaml mode=pretrain out_dir=runs/ar_countdown
python scripts/ar.py configs/arith_rl.yaml mode=rl init=runs/ar_countdown/last.pt out_dir=runs/ar_rl reward=exact
python scripts/ar.py configs/arith_rl.yaml mode=sft init=runs/ar_countdown/last.pt out_dir=runs/ar_sft
RL: same rollouts, rewards, leave-one-out advantages and evaluation as scripts/rl.py, with pi the autoregressive policy.
"""
import copy
import os
import time

import torch
import torch.nn.functional as F

from rlfm.ar import ARNet, answer_logits, ar_logp, ar_sample
from rlfm.common import EMA, load_config, make_task, require_holdout
from rlfm.rl import advantages, pg_loss
from rlfm.tracking import Run


def build(cfg, task):
    m = cfg["model"]
    return ARNet(task.vocab_size, d=m["dim"], depth=m["depth"], heads=m["heads"])


@torch.no_grad()
def evaluate(net, ref, task, reward_mod, evalset, temp, seed, prefix_k=""):
    """Fixed training and held-out questions: correct answers when sampling at T and greedy; KL to the reference on the
    sampled answers (sum over answer tokens of KL(pi_i || ref_i))."""
    m, La = {}, task.answer_len
    for name, q in evalset.items():
        g = torch.Generator(device=q.device).manual_seed(seed)
        for dec, y in (("sample", ar_sample(net, q, La, temp, gen=g)), ("argmax", ar_sample(net, q, La, greedy=True))):
            s = task.decode(torch.cat([q, y], 1).cpu())
            m[f"{dec}_{name}_correct"] = reward_mod.correct(s).float().mean().item()
            m[f"{dec}_{name}_wellformed"] = reward_mod.well_formed(s).float().mean().item()
            if hasattr(task, "true_answer"):
                m[f"{dec}_{name}_digit_acc"] = (y == task.true_answer(q)).float().mean().item()
            if dec == "sample" and name == "train" and ref is not None:
                lp = F.log_softmax(answer_logits(net, q, y) / temp, -1)
                lr = F.log_softmax(answer_logits(ref, q, y) / temp, -1)
                m["kl_ref"] = (lp.exp() * (lp - lr)).sum(-1).sum(-1).mean().item()
                m["entropy"] = -(lp.exp() * lp).sum(-1).mean().item()
    m["sample_correct"], m["argmax_correct"] = m["sample_train_correct"], m["argmax_train_correct"]
    return m


def main():
    cfg = load_config()
    mode = cfg.get("mode", "pretrain")
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "pretrain":
        require_holdout(cfg, cfg.get("allow_no_holdout", False), "pretraining")
        pre_cfg, net, ck = cfg, None, None
    else:
        ck = torch.load(cfg["init"], map_location=device, weights_only=False)
        pre_cfg = ck["pre_cfg"] if "pre_cfg" in ck else ck["cfg"]
    task, reward_mod = make_task(pre_cfg)
    net = build(pre_cfg, task).to(device)
    if ck is not None:
        net.load_state_dict(ck["ema"])
    if mode == "pretrain" and cfg.get("init"):
        ck0 = torch.load(cfg["init"], map_location=device, weights_only=False)
        net.load_state_dict(ck0["model"])
    tr = cfg["train"]
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"], weight_decay=tr.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / tr.get("warmup", 1)))
    ema = EMA(net, tr["ema"]) if mode == "pretrain" else None
    ref = None
    if mode != "pretrain":
        ref = copy.deepcopy(net).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    trk = cfg.get("track", {})
    run = Run(out, use_nb=trk.get("nb", False), subproject=trk.get("subproject"), title=trk.get("title"),
              formula=trk.get("formula"), algo=trk.get("algo"), config=cfg, question=trk.get("question"))
    print(f"params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M mode={mode}", flush=True)
    ev = cfg["eval"]
    g_eval = torch.Generator().manual_seed(ev["seed"])
    evalset = {name: task.sample_questions(ev["n"], heldout=held, generator=g_eval).to(device)
               for name, held in (("train", False), ("heldout", True))}
    rl = cfg.get("rl", {})
    temp, G, B = rl.get("temp", 1.0), rl.get("group_size", 8), rl.get("groups", 32)
    reward_kind = cfg.get("reward", "exact")
    Lq, La = task.prompt_len, task.answer_len
    every = tr.get("eval_every", ev.get("every", 250))
    eval_net = lambda: ema.model if ema is not None else net
    run.log(0, **evaluate(eval_net(), ref, task, reward_mod, evalset, temp, ev["seed"]))
    t0 = time.time()
    for step in range(1, tr["steps"] + 1):
        if mode in ("pretrain", "sft"):
            ids = task.sample(tr.get("batch_size", B * G)).to(device)
            loss = -ar_logp(net, ids[:, :Lq], ids[:, Lq:]).mean()
            logs = {"loss": loss}
        else:
            q = task.sample_questions(B, heldout=False).to(device).repeat_interleave(G, 0)
            y = ar_sample(net, q, La, temp)
            s = task.decode(torch.cat([q, y], 1).cpu())
            if reward_kind == "per_digit":
                r = (y == task.true_answer(q)).float()
            else:
                r = reward_mod.reward(s).to(device)
            A = advantages(r, G, rl.get("advantage", "loo"))
            logp = ar_logp(net, q, y, temp)
            loss = pg_loss(logp, logp.detach(), A, None)
            logs = {"pg_loss": loss, "reward_train": r.mean()}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), tr.get("grad_clip", 1.0))
        opt.step()
        sched.step()
        if ema is not None:
            ema.update(net)
        if step % tr["log_every"] == 0:
            run.log(step, grad_norm=gn.item(), sec=time.time() - t0, **{k: v.item() for k, v in logs.items()})
        if step % every == 0 or step == tr["steps"]:
            m = evaluate(eval_net(), ref, task, reward_mod, evalset, temp, ev["seed"])
            run.log(step, **m)
            torch.save({"cfg": cfg, "pre_cfg": pre_cfg, "step": step, "ema": eval_net().state_dict(),
                        "model": net.state_dict()}, os.path.join(out, "last.pt"))
    run.finish(**m)


if __name__ == "__main__":
    main()
