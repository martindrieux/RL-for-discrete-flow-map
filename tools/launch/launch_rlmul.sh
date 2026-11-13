cd ~/projects/RL-for-discrete-flow-map
K=$1
F='$A_g = r_g - \frac{1}{G-1}\sum_{j \ne g} r_j$, $\mathcal{L} = -\frac{1}{GL}\sum_{g,i} A_g \log p_{\theta,i}(y_{g,i} \mid z_g, c_b)$, $K = \cfg{rl.K}$ jumps, gradient through all jumps'
Q='On 4-digit multiplication (pretrained model at about 6% exact answers), does RL on log pi(y | z, c) raise the fraction of correct answers, including on held-out questions, and how does it depend on the number of jumps K?'
n=rl_mul4_K$K; mkdir -p runs/$n
tmux new-session -d -s $n ".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=runs/ckpt_mul4_100k.pt out_dir=runs/$n rl.K=$K train.lr=3e-5 train.steps=5000 eval.every=250 track.nb=true track.subproject=rl-mul-cond track.algo=flowmap-pg 'track.title=4-digit multiplication, leave-one-out baseline, $K jump(s)' 'track.formula=$F' 'track.question=$Q' seed=0 > runs/$n/log.txt 2>&1"
echo launched $n
