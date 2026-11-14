cd ~/projects/RL-for-discrete-flow-map
V=$1; K=$2; AL=$3; G=${4:-8}
Q='From the 4-digit multiplication model at 200k steps, does pushing the policy to explore (an entropy bonus, or more samples per question) let RL find more correct digits?'
A='$A_{g,i} = r_{g,i} - \frac{1}{G-1}\sum_{j \ne g} r_{j,i}$'
NG=$((256 / G))   # keep 256 answers per step
case $V in
  det)   EXTRA="";                   P="last jump";;
  stoch) EXTRA="rl.stochastic=true"; P="sampling at every jump";;
esac
T="Entropy bonus $AL, $G answers per question, $P, $K jump(s)"
F="$A, \$\\mathcal{L}_{pg} - \\cfg{rl.entropy} \\mathcal{H}\$, \$G = \\cfg{rl.group_size}\$, \$K = \\cfg{rl.K}\$"
n=rl_explore_${V}_K${K}_a${AL}_G$G; mkdir -p runs/$n
tmux new-session -d -s $n ".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=runs/ckpt_mul4_200k.pt out_dir=runs/$n reward=per_digit rl.K=$K rl.entropy=$AL rl.group_size=$G rl.groups=$NG $EXTRA train.lr=3e-5 train.steps=5000 eval.every=250 track.nb=true track.subproject=rl-mul-explore track.algo=flowmap-pg 'track.title=$T' 'track.formula=$F' 'track.question=$Q' seed=0 > runs/$n/log.txt 2>&1"
echo launched $n
