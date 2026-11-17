"""Pretrain a flow map with the FMLM direct-training loss (diagonal CE + self-distilled semigroup).

python scripts/pretrain.py configs/arith_pretrain.yaml out_dir=runs/pretrain seed=0
"""
import os
import time

import torch

from rlfm.common import (EMA, training_task, cond_mode, eval_conditional, eval_sampler, load_config, make_flowmap, make_task, one_hot,
                         require_holdout, split)
from rlfm.flowmap import pretrain_loss
from rlfm.tracking import Run


def main():
    cfg = load_config()
    require_holdout(cfg, cfg.get("allow_no_holdout", False), "pretraining")
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task, reward_mod = make_task(cfg)
    train_task = training_task(task, cfg)
    V, L = task.vocab_size, task.seq_len
    fm = make_flowmap(cfg, V, L, task).to(device)
    init_ck = None
    if cfg.get("init"):  # resume from a pretraining checkpoint (weights and their EMA), with this config's optimizer
        init_ck = torch.load(cfg["init"], map_location=device, weights_only=False)
        fm.load_state_dict(init_ck["model"])
    start = init_ck["step"] if (init_ck is not None and cfg.get("resume")) else 0  # resume: steps go on from the checkpoint
    conditional = cond_mode(cfg) != "none"
    tr = cfg["train"]
    opt = torch.optim.AdamW(fm.net.parameters(), lr=tr["lr"], weight_decay=tr.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / tr["warmup"]))
    ema = EMA(fm, tr["ema"])
    if init_ck is not None:
        ema.model.load_state_dict(init_ck["ema"])
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    trk = cfg.get("track", {})
    run = Run(out, use_nb=trk.get("nb", False), subproject=trk.get("subproject"), title=trk.get("title"),
              formula=trk.get("formula"), algo=trk.get("algo"), config=cfg, question=trk.get("question"))
    print(f"params: {sum(p.numel() for p in fm.net.parameters()) / 1e6:.2f}M  V={V} L={L}  device={device}", flush=True)
    t0, best = time.time(), -1.0
    for step in range(start + 1, tr["steps"] + 1):
        ids = train_task.sample(tr["batch_size"]).to(device)
        if conditional:
            q, ans = split(ids, task)
            losses = pretrain_loss(fm, one_hot(ans, V), q=q)
        else:
            losses = pretrain_loss(fm, one_hot(ids, V))
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(fm.net.parameters(), tr.get("grad_clip", 1.0))
        opt.step()
        sched.step()
        ema.update(fm)
        if step % tr["log_every"] == 0:
            run.log(step, loss=losses["loss"].item(), loss_diag=losses["loss_diag"].item(),
                    loss_off=losses["loss_off"].item(), grad_norm=gn.item(), sec=time.time() - t0)
        if step % tr["eval_every"] == 0 or step == tr["steps"]:
            ev = (eval_conditional if conditional else eval_sampler)
            m = ev(ema.model, task, reward_mod, cfg["eval"]["Ks"], cfg["eval"]["n"], cfg["eval"]["seed"], device)
            run.log(step, **m)
            ck = {"cfg": cfg, "step": step, "ema": ema.model.state_dict(), "model": fm.state_dict()}
            torch.save(ck, os.path.join(out, "last.pt"))
            if cfg.get("keep_ckpts") and step % tr.get("ckpt_every", tr["eval_every"]) == 0:  # to start RL from a chosen accuracy
                torch.save(ck, os.path.join(out, f"ckpt_{step}.pt"))
            score = m["K1_train_correct"] if conditional else m["K1_wellformed"]
            if score > best:
                best = score
                torch.save(ck, os.path.join(out, "best.pt"))
    run.finish(**m)


if __name__ == "__main__":
    main()
