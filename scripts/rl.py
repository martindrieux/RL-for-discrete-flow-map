"""RL fine-tuning of a pretrained flow map on the categorical likelihood log pi(y | z).

python scripts/rl.py configs/arith_rl.yaml init=runs/pretrain/last.pt out_dir=runs/rl_x seed=0 rl.temp=2

At every evaluation the policy, optimizer and random number generators are saved to out_dir/state.pt; with resume=true
a run whose out_dir holds state.pt goes on from that step (a preempted SLURM job), otherwise it starts from init.
"""
import copy
import os
import time

import torch

from rlfm.common import (cond_mode, load_config, load_flowmap, load_train_state, make_task, require_holdout, resumed_title,
                         save_train_state, task_metrics, training_task)
from rlfm.policy import chain_logits, policy_logits, sample_tokens, stochastic_chain, token_entropy, token_kl
from rlfm.rl import RLConfig, collect, rl_update, sft_loss
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
    """Fixed training and held-out questions with fixed noises: correct answers when sampling at T and with argmax.

    "zero" is the argmax answer from the noise z = 0, the mode-seeking decoding.
    Stochastic chain: "sample" samples at every jump, "argmax" takes the argmax at every jump, "det" is the argmax of
    the deterministic flow map's last jump; entropy and KL are over the tokens of all jumps of the sampled chain.
    """
    m = {}
    for name, (q, z) in evalset.items():
        g = torch.Generator(device=z.device).manual_seed(seed)
        if cfg.stochastic:
            xs, ys, logits = stochastic_chain(fm, z, cfg.K, cfg.temp, q, g)
            logits_ref = chain_logits(ref, xs, cfg.K, q)
            decs = (("sample", ys[:, -1]), ("argmax", stochastic_chain(fm, z, cfg.K, cfg.temp, q, greedy=True)[1][:, -1]),
                    ("det", policy_logits(fm, z, cfg.K, cfg.option, cfg.tau_star, q).argmax(-1)))
        else:
            logits = policy_logits(fm, z, cfg.K, cfg.option, cfg.tau_star, q)
            logits_ref = policy_logits(ref, z, cfg.K, cfg.option, cfg.tau_star, q)
            decs = (("sample", sample_tokens(logits, cfg.temp, g)), ("argmax", logits.argmax(-1)),
                    ("zero", policy_logits(fm, torch.zeros_like(z), cfg.K, cfg.option, cfg.tau_star, q).argmax(-1)))
        for dec, y in decs:
            s = task.decode(torch.cat([q, y], 1).cpu())
            m[f"{dec}_{name}_correct"] = reward_mod.correct(s).float().mean().item()
            if hasattr(task, "true_answer"):
                m[f"{dec}_{name}_digit_acc"] = (y == task.true_answer(q)).float().mean().item()
            else:
                m[f"{dec}_{name}_wellformed"] = reward_mod.well_formed(s).float().mean().item()
        if name == "train":
            m["entropy"] = token_entropy(logits, cfg.temp).mean().item()
            m["kl_ref"] = token_kl(logits, logits_ref, cfg.temp).flatten(1).sum(-1).mean().item()
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
    if rc.stochastic and not conditional:
        raise ValueError("rl.stochastic is evaluated only for conditional models so far")
    tr = cfg_all["train"]
    opt = torch.optim.AdamW(fm.net.parameters(), lr=tr["lr"], weight_decay=0.0)
    out = cfg_all["out_dir"]
    os.makedirs(out, exist_ok=True)
    trk = cfg_all.get("track", {})
    state_path, start = os.path.join(out, "state.pt"), 0
    if cfg_all.get("resume") and os.path.exists(state_path):  # ref stays the init checkpoint; fm and opt go on
        start = load_train_state(state_path, fm, opt, device)["step"]
        print(f"resumed from {state_path} at step {start}", flush=True)
        if trk.get("title"):
            trk["title"] = resumed_title(trk["title"], start, tr["steps"])
    run = Run(out, use_nb=trk.get("nb", False), subproject=trk.get("subproject"), title=trk.get("title"),
              formula=trk.get("formula"), algo=trk.get("algo"), config=cfg_all, question=trk.get("question"))

    # exact: 1 for a correct equation; digits: fraction of correct answer digits;
    # per_digit: one reward per answer digit (1 if correct), so each digit gets its own advantage
    reward_kind = cfg_all.get("reward", "exact")
    assert reward_kind in ("exact", "digits", "per_digit")
    if reward_kind != "exact" and not conditional:
        raise ValueError(f"reward={reward_kind} needs a conditional model (the true answer comes from the question)")

    def reward_fn(y, q=None):
        if reward_kind == "digits":
            return reward_mod.digit_fraction(y, task.true_answer(q))
        if reward_kind == "per_digit":
            return reward_mod.digit_correct(y, task.true_answer(q))
        full = y if q is None else torch.cat([q, y], 1)
        return reward_mod.reward(task.decode(full.cpu()))

    train_task = training_task(task, cfg_all)
    question_fn = (lambda B: train_task.sample_questions(B, heldout=False)) if conditional else None
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
    m = None
    if start == 0:
        run.log(0, **do_eval())
    t0 = time.time()
    sft = cfg_all.get("sft", False)  # supervised fine-tuning on the true answers instead of RL (same policy, same budget)
    if sft and not conditional:
        raise ValueError("sft needs a conditional model (the true answer comes from the question)")
    for step in range(start + 1, tr["steps"] + 1):
        if sft:
            N = rc.groups * rc.group_size
            if hasattr(task, "true_answer"):
                q = train_task.sample_questions(N, heldout=False).to(device)
                y_star = task.true_answer(q)
            else:  # several valid answers: the pretraining pairs (question, one valid answer)
                ids = train_task.sample(N).to(device)
                q, y_star = ids[:, :task.prompt_len], ids[:, task.prompt_len:]
            z = torch.randn(N, L, V, device=device)
            loss = sft_loss(fm, z, y_star, rc, q)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(fm.parameters(), tr.get("grad_clip", 1.0))
            opt.step()
            if step % tr["log_every"] == 0:
                run.log(step, sft_loss=loss.item(), grad_norm=gn.item(), sec=time.time() - t0)
        else:
            ro = collect(fm, ref, reward_fn, rc, L, V, device, question_fn=question_fn)
            diag = rl_update(fm, opt, ro, rc, tr.get("grad_clip", 1.0))
            if step % tr["log_every"] == 0:
                run.log(step, reward_train=ro.reward.mean().item(), sec=time.time() - t0, **diag)
        if step % ev["every"] == 0 or step == tr["steps"]:
            m = do_eval()
            run.log(step, **m)
            save_train_state(state_path, step, fm, opt)
    if m is None:  # resumed at the last step
        m = do_eval()
    torch.save({"cfg": cfg_all, "pre_cfg": pre_cfg, "ema": fm.state_dict()}, os.path.join(out, "last.pt"))
    run.finish(**m)


if __name__ == "__main__":
    main()
