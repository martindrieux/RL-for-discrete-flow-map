cd ~/projects/RL-for-discrete-flow-map
V=$1; K=$2
Q='From the 4-digit multiplication model at 200k steps, does RL that also trains the intermediate jumps (sampling at every jump, or reward-weighted regression on intermediate predictions) beat RL on the last jump only, and does it make more jumps useful?'
A='$A_{g,i} = r_{g,i} - \frac{1}{G-1}\sum_{j \ne g} r_{j,i}$'
case $V in
  det)  EXTRA=""; T="Last jump only, $K jump(s)"; F="$A, policy = last jump, \$K = \\cfg{rl.K}\$, gradient through all jumps";;
  stoch) EXTRA="rl.stochastic=true"; T="Sampling at every jump, $K jumps"; F="$A, \$\\hat y_k \\sim \\mathrm{Cat}(\\hat\\delta_k)\$ at every jump, \$\\log\\pi = \\sum_{k,i} \\log \\hat\\delta_{k,i}(\\hat y_{k,i})\$, \$K = \\cfg{rl.K}\$";;
  rwr)  EXTRA="rl.rwr=1.0"; T="Last jump plus regression on intermediate predictions, $K jumps"; F="$A, \$\\mathcal{L}_{pg} + \\cfg{rl.rwr} \\mathcal{L}_{rwr}\$ on \$\\ell_{t_k,1}(x_k)\$, \$k < K-1\$, \$K = \\cfg{rl.K}\$";;
esac
n=rl_chain_${V}_K$K; mkdir -p runs/$n
tmux new-session -d -s $n ".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=runs/ckpt_mul4_200k.pt out_dir=runs/$n reward=per_digit rl.K=$K $EXTRA train.lr=3e-5 train.steps=5000 eval.every=250 track.nb=true track.subproject=rl-mul-chain track.algo=flowmap-pg 'track.title=$T' 'track.formula=$F' 'track.question=$Q' seed=0 > runs/$n/log.txt 2>&1"
echo launched $n
