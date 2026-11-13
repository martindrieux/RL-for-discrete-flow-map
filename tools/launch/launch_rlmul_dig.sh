cd ~/projects/RL-for-discrete-flow-map
K=$1
F='$r_g = \frac{1}{L_a}\sum_i \mathbb{1}[y_{g,i} = y^*_i(c_b)]$, $A_g = r_g - \frac{1}{G-1}\sum_{j \ne g} r_j$, $K = \cfg{rl.K}$ jumps, gradient through all jumps'
n=rl_mul4_dig_K$K; mkdir -p runs/$n
tmux new-session -d -s $n ".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=runs/ckpt_mul4_100k.pt out_dir=runs/$n reward=digits rl.K=$K train.lr=3e-5 train.steps=5000 eval.every=250 track.nb=true track.subproject=rl-mul-cond track.algo=flowmap-pg 'track.title=Per-digit reward, $K jump(s)' 'track.formula=$F' seed=0 > runs/$n/log.txt 2>&1"
echo launched $n
