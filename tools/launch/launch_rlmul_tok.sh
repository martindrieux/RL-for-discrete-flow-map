cd ~/projects/RL-for-discrete-flow-map
K=$1
F='$r_{g,i} = \mathbb{1}[y_{g,i} = y^*_i(c_b)]$, $A_{g,i} = r_{g,i} - \frac{1}{G-1}\sum_{j \ne g} r_{j,i}$, $K = \cfg{rl.K}$ jumps, gradient through all jumps'
n=rl_mul4_tok_K$K; mkdir -p runs/$n
tmux new-session -d -s $n ".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=runs/ckpt_mul4_100k.pt out_dir=runs/$n reward=per_digit rl.K=$K train.lr=3e-5 train.steps=5000 eval.every=250 track.nb=true track.subproject=rl-mul-cond track.algo=flowmap-pg 'track.title=One advantage per digit, $K jump(s)' 'track.formula=$F' seed=0 > runs/$n/log.txt 2>&1"
echo launched $n
