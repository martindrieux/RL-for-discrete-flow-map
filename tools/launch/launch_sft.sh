cd ~/projects/RL-for-discrete-flow-map
M=$1; K=$2; LR=$3; S=${4:-5000}
Q='From the 4-digit multiplication model at 200k steps, how far can the same budget go with the true answers given (supervised fine-tuning of the policy, or more pretraining)? An upper bound for RL, which plateaus at 26-28% held-out correct.'
n=sft_${M}_K${K}_lr${LR}_s$S; mkdir -p runs/$n
if [ "$M" = policy ]; then
  T="Supervised fine-tuning of the policy, $K jump(s), lr $LR"
  F='$\mathcal{L}_{sft} = -\frac{1}{N L_a}\sum_{n,i} \log p_{\theta,i}(y^*_{n,i} \mid z_n, c_n)$, $K = \cfg{rl.K}$, $\eta = \cfg{train.lr}$'
  CMD=".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=runs/ckpt_mul4_200k.pt out_dir=runs/$n sft=true rl.K=$K train.lr=$LR train.steps=$S eval.every=250 track.algo=flowmap-sft"
else
  T="More pretraining, lr $LR, $S steps"
  F='direct flow-map loss resumed from 200k steps, $\eta = \cfg{train.lr}$, \cfg{train.steps} more steps'
  CMD=".venv/bin/python scripts/pretrain.py configs/arith_cond_pretrain.yaml init=runs/ckpt_mul4_200k.pt out_dir=runs/$n task.n_digits=4 task.op=mul model.dim=256 model.depth=6 model.heads=8 train.lr=$LR train.warmup=1 train.steps=$S train.eval_every=$((S / 20)) eval.Ks=[1,4] track.algo=fmlm-direct"
fi
tmux new-session -d -s $n "$CMD seed=0 track.nb=true track.subproject=sft-vs-rl-mul 'track.title=$T' 'track.formula=$F' 'track.question=$Q' > runs/$n/log.txt 2>&1"
echo launched $n
