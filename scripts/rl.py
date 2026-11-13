"""RL fine-tuning of a pretrained flow map on the categorical likelihood log pi(y | z).

python scripts/rl.py configs/arith_rl.yaml init=runs/pretrain/last.pt out_dir=runs/rl_x seed=0 rl.temp=2
"""
import copy
import os
import time

import torch

from rlfm.common import cond_mode, load_config, load_flowmap, make_task, require_holdout, task_metrics
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


@torch.no_grad()
def evaluate_conditional(fm, ref, task, reward_mod, cfg: RLConfig, evalset: dict, seed: int) -> dict:
    """Fixed training and held-out questions with fixed noises: correct answers when sampling at T and with argmax."""
    m = {}
    for name, (q, z) in evalset.items():
        logits = policy_logits(fm, z, cfg.K, cfg.option, cfg.tau_star, q)
        logits_ref = policy_logits(ref, z, cfg.K, cfg.option, cfg.tau_star, q)
        g = torch.Generator(device=z.device).manual_seed(seed)
        for dec, y in (("sample", sample_tokens(logits, cfg.temp, g)), ("argmax", logits.argmax(-1))):
            s = task.decode(torch.cat([q, y], 1).cpu())
            m[f"{dec}_{name}_correct"] = reward_mod.correct(s).float().mean().item()
            m[f"{dec}_{name}_digit_acc"] = (y == task.true_answer(q)).float().mean().item()
        if name == "train":
            m["entropy"] = token_entropy(logits, cfg.temp).mean().item()
            m["kl_ref"] = token_kl(logits, logits_ref, cfg.temp).sum(-1).mean().item()
    m["sample_correct"] = m["sample_train_correct"]  # headline metric of the project
    m["argmax_correct"] = m["argmax_train_correct"]
    return m


def main():
    cfg_all = load_config()
    torch.manual_seed(cfg_all.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm, pre_cfg = load_flowmap(cfg_all["init"], device)
    require_holdout(pre_cfg, cfg_all.get("allow_no_holdout", False), "RL (the pretrained checkpoint has no held-out set)")
    task, reward_mod = make_task(pre_cfg)
    conditional = cond_mode(pre_cfg) != "none"
    V = task.vocab_size
    L = task.seq_len - task.prompt_len if conditional else task.seq_len
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

    reward_kind = cfg_all.get("reward", "exact")  # exact: 1 for a correct equation; digits: fraction of correct answer digits
    assert reward_kind in ("exact", "digits")
    if reward_kind == "digits" and not conditional:
        raise ValueError("reward=digits needs a conditional model (the true answer comes from the question)")

    def reward_fn(y, q=None):
        if reward_kind == "digits":
            return reward_mod.digit_fraction(y, task.true_answer(q))
        full = y if q is None else torch.cat([q, y], 1)
        return reward_mod.reward(task.decode(full.cpu()))

    question_fn = (lambda B: task.sample_questions(B, heldout=False)) if conditional else None
    ev = cfg_all["eval"]
    g_eval = torch.Generator().manual_seed(ev["seed"])
    if conditional:  # n training questions and n held-out questions, each with its own fixed noise
        evalset = {name: (task.sample_questions(ev["n"], heldout=held, generator=g_eval).to(device),
                          torch.randn(ev["n"], L, V, generator=g_eval).to(device))
                   for name, held in (("train", False), ("heldout", True))}

        def do_eval():
            return evaluate_conditional(fm, ref, task, reward_mod, rc, evalset, ev["seed"])
    else:
        z_eval = torch.randn(ev["n"], L, V, generator=g_eval).to(device)

        def do_eval():
            return evaluate(fm, ref, task, reward_mod, rc, z_eval, ev["seed"])
    run.log(0, **do_eval())
    t0 = time.time()
    for step in range(1, tr["steps"] + 1):
        ro = collect(fm, ref, reward_fn, rc, L, V, device, question_fn=question_fn)
        diag = rl_update(fm, opt, ro, rc, tr.get("grad_clip", 1.0))
        if step % tr["log_every"] == 0:
            run.log(step, reward_train=ro.reward.mean().item(), sec=time.time() - t0, **diag)
        if step % ev["every"] == 0 or step == tr["steps"]:
            m = do_eval()
            run.log(step, **m)
    torch.save({"cfg": cfg_all, "pre_cfg": pre_cfg, "ema": fm.state_dict()}, os.path.join(out, "last.pt"))
    run.finish(**m)


if __name__ == "__main__":
    main()
