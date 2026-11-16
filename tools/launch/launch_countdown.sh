# Countdown: RL vs supervised fine-tuning vs continued pretraining, for the flow map and the autoregressive baseline.
# args: fm|ar  rl|sft|pre  [K] [seed] [steps]
# optional env: CKPT=<checkpoint> PRE=<pretraining steps of CKPT, for names and titles>
#               BETA=<KL weight to the pretrained model, rl only>  LR=<learning rate, default 3e-5>
cd ~/projects/RL-for-discrete-flow-map
NET=$1; M=$2; K=${3:-1}
Q='On Countdown (several valid answers), does RL improve on supervised fine-tuning with one valid answer per question, for the flow map and for an autoregressive transformer of the same size?'
S=${4:-0}; ST=${5:-5000}
PRE=${PRE:-20k}; LR=${LR:-3e-5}; BETA=${BETA:-0}
CK=${CKPT:-runs/ckpt_cd_${NET}_20k.pt}
TR="track.nb=true track.subproject=rl-countdown 'track.question=$Q'"
COMMON="train.lr=$LR train.steps=$ST eval.every=250 seed=$S $TR"
TAIL="from $PRE pretraining steps, seed $S, $ST steps"
B=""; BT=""; BN=""
if [ "$BETA" != 0 ]; then B="rl.beta=$BETA"; BT=", KL weight $BETA"; BN="_b$BETA"; fi
LT=""; LN=""
if [ "$LR" != 3e-5 ]; then LT=", lr $LR"; LN="_lr$LR"; fi
if [ $NET = fm ]; then
  n=cd_fm_${M}_K${K}${BN}${LN}_s${S}_n${ST}_pre$PRE
  case $M in
    rl)  CMD=".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=$CK out_dir=runs/$n reward=exact rl.K=$K $B track.algo=flowmap-pg 'track.title=Flow map, RL, $K jump(s)$BT$LT, $TAIL' 'track.formula=\$r = \\mathbb{1}[e(n) = v]\$, \$K = \\cfg{rl.K}\$, \$\\beta = \\cfg{rl.beta}\$' $COMMON";;
    sft) CMD=".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=$CK out_dir=runs/$n sft=true rl.K=$K track.algo=flowmap-sft 'track.title=Flow map, supervised fine-tuning, $K jump(s)$LT, $TAIL' 'track.formula=\$\\mathcal{L}_{sft}\$ on the generating expression, \$K = \\cfg{rl.K}\$' $COMMON";;
    pre) n=cd_fm_pre${LN}_s${S}_n${ST}_pre$PRE
         CMD=".venv/bin/python scripts/pretrain.py configs/countdown_pretrain.yaml init=$CK out_dir=runs/$n train.lr=$LR train.warmup=1 train.steps=$ST train.eval_every=250 eval.Ks=[1,4] seed=$S track.algo=fmlm-direct 'track.title=Flow map, continued pretraining, lr $LR, $TAIL' 'track.formula=direct flow-map loss resumed, \$\\eta = \\cfg{train.lr}\$' $TR";;
  esac
else
  n=cd_ar_${M}${BN}${LN}_s${S}_n${ST}_pre$PRE
  case $M in
    rl)  CMD=".venv/bin/python scripts/ar.py configs/arith_rl.yaml init=$CK out_dir=runs/$n mode=rl reward=exact $B track.algo=ar-pg 'track.title=Autoregressive, RL$BT$LT, $TAIL' 'track.formula=\$r = \\mathbb{1}[e(n) = v]\$, same advantages as the flow map, \$\\beta = \\cfg{rl.beta}\$' $COMMON";;
    sft) CMD=".venv/bin/python scripts/ar.py configs/arith_rl.yaml init=$CK out_dir=runs/$n mode=sft train.batch_size=256 track.algo=ar-ce 'track.title=Autoregressive, supervised fine-tuning$LT, $TAIL' 'track.formula=\$\\mathcal{L}_{ar}\$ on the generating expression' $COMMON";;
    pre) CMD=".venv/bin/python scripts/ar.py configs/countdown_pretrain.yaml init=$CK out_dir=runs/$n mode=pretrain train.lr=$LR train.warmup=1 train.steps=$ST train.eval_every=250 seed=$S track.algo=ar-ce 'track.title=Autoregressive, continued pretraining, lr $LR, $TAIL' 'track.formula=\$\\mathcal{L}_{ar}\$ resumed, \$\\eta = \\cfg{train.lr}\$' $TR";;
  esac
fi
mkdir -p runs/$n
tmux new-session -d -s $n "( $CMD ) > runs/$n/log.txt 2>&1"
echo launched $n
