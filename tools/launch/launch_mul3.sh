# 3-digit multiplication from a partly trained model (10k pretraining steps): RL vs supervised fine-tuning vs more pretraining.
# args: pretrain | rl K reward lr | sft K lr | pre lr
cd ~/projects/RL-for-discrete-flow-map
M=$1
CK=runs/ckpt_mul3_10k.pt
Q='From a partly trained 3-digit multiplication model (10k pretraining steps), how does RL compare with supervised fine-tuning on the true answers and with more pretraining, for the same number of steps?'
WAIT="until [ -f $CK ]; do sleep 30; done;"
case $M in
  pretrain)
    n=mul3_10k
    CMD=".venv/bin/python scripts/pretrain.py configs/arith_cond_pretrain.yaml out_dir=runs/$n task.n_digits=3 task.op=mul model.dim=256 model.depth=6 model.heads=8 train.steps=10000 train.eval_every=1000 seed=0 track.nb=true track.subproject=pretrain-mul track.algo=fmlm-direct 'track.title=3-digit multiplication, stopped at 10k steps' 'track.formula=same as the 3-digit run, \$\\cfg{train.steps}\$ steps' && cp runs/$n/last.pt $CK"
    WAIT="";;
  rl)
    K=$2; R=$3; LR=$4; n=mul3_rl_${R}_K${K}_lr$LR
    CMD=".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=$CK out_dir=runs/$n reward=$R rl.K=$K train.lr=$LR train.steps=5000 eval.every=250 seed=0 track.nb=true track.subproject=rl-mul3 track.algo=flowmap-pg 'track.title=RL, reward $R, $K jump(s), lr $LR' 'track.formula=reward \`$R\`, \$K = \\cfg{rl.K}\$, \$\\eta = \\cfg{train.lr}\$' 'track.question=$Q'";;
  sft)
    K=$2; LR=$3; n=mul3_sft_K${K}_lr$LR
    CMD=".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=$CK out_dir=runs/$n sft=true rl.K=$K train.lr=$LR train.steps=5000 eval.every=250 seed=0 track.nb=true track.subproject=rl-mul3 track.algo=flowmap-sft 'track.title=Supervised fine-tuning on the true answers, $K jump(s), lr $LR' 'track.formula=\$\\mathcal{L}_{sft}\$, \$K = \\cfg{rl.K}\$, \$\\eta = \\cfg{train.lr}\$' 'track.question=$Q'";;
  pre)
    LR=$2; n=mul3_pre_lr$LR
    CMD=".venv/bin/python scripts/pretrain.py configs/arith_cond_pretrain.yaml init=$CK out_dir=runs/$n task.n_digits=3 task.op=mul model.dim=256 model.depth=6 model.heads=8 train.lr=$LR train.warmup=1 train.steps=5000 train.eval_every=250 eval.Ks=[1,4] seed=0 track.nb=true track.subproject=rl-mul3 track.algo=fmlm-direct 'track.title=More pretraining, lr $LR' 'track.formula=direct flow-map loss resumed from 10k steps, \$\\eta = \\cfg{train.lr}\$' 'track.question=$Q'";;
esac
mkdir -p runs/$n
tmux new-session -d -s $n "$WAIT ( $CMD ) > runs/$n/log.txt 2>&1"
echo launched $n
