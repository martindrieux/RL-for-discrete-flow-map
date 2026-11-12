"""RL fine-tuning of a pretrained flow map on the categorical likelihood log pi(y | z).

python scripts/rl.py configs/arith_rl.yaml init=runs/pretrain/last.pt out_dir=runs/rl_x seed=0 rl.temp=2
"""
import copy
import os
import time

import torch

from rlfm.common import load_config, load_flowmap, make_task, task_metrics
from rlfm.policy import policy_logits, sample_tokens, token_entropy, token_kl
from rlfm.rl import RLConfig, collect, rl_update
from rlfm.tracking import Run


@torch.no_grad()
def evaluate(fm, ref, task, reward_mod, cfg: RLConfig, z: torch.Tensor, seed: int) -> dict:
    """Same noises every time: reward under sampling at T and under argmax, entropy, KL to the reference."""
    logits = policy_logits(fm, z, cfg.K, cfg.option, cfg.tau_star)
    logits_ref = policy_logits(ref, z, cfg.K, cfg.option, cfg.tau_star)
    g = torch.Generator(device=z.device).manual_seed(seed)
    y = sample_tokens(logits, cfg.temp, g)
    m = task_metrics(task, reward_mod, y, "sample_")
    m.update(task_metrics(task, reward_mod, logits.argmax(-1), "argmax_"))
    m["entropy"] = token_entropy(logits, cfg.temp).mean().item()
    m["kl_ref"] = token_kl(logits, logits_ref, cfg.temp).sum(-1).mean().item()
    return m


def main():
    cfg_all = load_config()
    torch.manual_seed(cfg_all.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm, pre_cfg = load_flowmap(cfg_all["init"], device)
    task, reward_mod = make_task(pre_cfg)
    V, L = task.vocab_size, task.seq_len
    ref = copy.deepcopy(fm).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rc = RLConfig(**cfg_all["rl"])
    tr = cfg_all["train"]
    opt = torch.optim.AdamW(fm.net.parameters(), lr=tr["lr"], weight_decay=0.0)
    out = cfg_all["out_dir"]
    os.makedirs(out, exist_ok=True)
    trk = cfg_all.get("track", {})
    run = Run(out, use_nb=trk.get("nb", False), subproject=trk.get("subproject"), title=trk.get("title"),
              formula=trk.get("formula"), algo=trk.get("algo"), config=cfg_all, question=trk.get("question"))

    def reward_fn(y):
        return reward_mod.reward(task.decode(y.cpu()))

    ev = cfg_all["eval"]
    z_eval = torch.randn(ev["n"], L, V, generator=torch.Generator().manual_seed(ev["seed"])).to(device)
    run.log(0, **evaluate(fm, ref, task, reward_mod, rc, z_eval, ev["seed"]))
    t0 = time.time()
    for step in range(1, tr["steps"] + 1):
        ro = collect(fm, ref, reward_fn, rc, L, V, device)
        diag = rl_update(fm, opt, ro, rc, tr.get("grad_clip", 1.0))
        if step % tr["log_every"] == 0:
            run.log(step, reward_train=ro.reward.mean().item(), sec=time.time() - t0, **diag)
        if step % ev["every"] == 0 or step == tr["steps"]:
            m = evaluate(fm, ref, task, reward_mod, rc, z_eval, ev["seed"])
            run.log(step, **m)
    torch.save({"cfg": cfg_all, "pre_cfg": pre_cfg, "ema": fm.state_dict()}, os.path.join(out, "last.pt"))
    run.finish(**m)


if __name__ == "__main__":
    main()
