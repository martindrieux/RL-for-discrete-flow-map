# Sudoku: pretraining, then RL vs continued pretraining, for the flow map and the autoregressive baseline.
# args: fm|ar  pretrain|rl|sft|pre  [K] [seed] [steps]
# optional env: CKPT=<checkpoint> PRE=<pretraining steps of CKPT, for names and titles>
#               BETA=<KL weight to the pretrained model, rl only>  LR=<learning rate, default 3e-5>
#               DIM, DEPTH=<network size for pretrain and pre (default 256, 6); must match CKPT for pre>
#               MICRO=<slices of the RL batch whose gradients are summed, default 1; only memory changes>
#               ZS=<noise scale of the RL rollouts, z ~ N(0, ZS^2 I), default 1; rl only>
#               TMB=<train_min_blanks: training puzzles have at least TMB blanks, evaluation unchanged; rl and pre>
#               TXB=<train_max_blanks: training puzzles have at most TXB blanks (easy only), evaluation unchanged; pretrain, pre, rl>
#               GS=<group size G>  GR=<groups B per step>  (rl only; defaults 8 and 32)
#               RESUME=<pretraining checkpoint ckpt_N.pt, pretrain only: go on from step N in the same folder>  CKE=<steps between kept checkpoints, default 10000>
cd ~/projects/RL-for-discrete-flow-map
NET=$1; M=$2; K=${3:-4}
S=${4:-0}; ST=${5:-5000}
PRE=${PRE:-20k}; LR=${LR:-3e-5}; BETA=${BETA:-0}; DIM=${DIM:-256}; DEPTH=${DEPTH:-6}; MICRO=${MICRO:-1}
CK=${CKPT:-runs/ckpt_sd_${NET}_$PRE.pt}
QP='Given a 9x9 Sudoku puzzle with a unique solution, do the flow map and an autoregressive transformer of the same size learn to write the solution, on puzzles from held-out grids?'
Q='On Sudoku (one valid answer), does RL improve the flow map as much as an autoregressive transformer of the same size, from the same held-out accuracy, beyond continued pretraining?'
TR="track.nb=true track.subproject=rl-sudoku 'track.question=$Q'"
COMMON="train.lr=$LR train.steps=$ST eval.every=250 seed=$S $TR"
TAIL="from $PRE, seed $S, $ST steps"
B=""; BT=""; BN=""
[ "$MICRO" != 1 ] && B="rl.micro=$MICRO"   # gradient summed over MICRO slices of the batch: same step, less memory
if [ "$BETA" != 0 ]; then B="$B rl.beta=$BETA"; BT=", KL weight $BETA"; BN="_b$BETA"; fi
ZS=${ZS:-1}
if [ "$ZS" != 1 ]; then B="$B rl.zs=$ZS"; BT="$BT, noise $ZS"; BN="${BN}_zs$ZS"; fi
if [ -n "$GS" ]; then B="$B rl.group_size=$GS rl.groups=${GR:-32}"; BT="$BT, G=$GS"; BN="${BN}_G$GS"; fi
if [ -n "$TMB" ]; then B="$B train_min_blanks=$TMB"; BT="$BT, >=$TMB blanks"; BN="${BN}_hard$TMB"; fi
if [ -n "$TXB" ]; then B="$B train_max_blanks=$TXB"; BT="$BT, <=$TXB blanks"; BN="${BN}_easy$TXB"; fi
LT=""; LN=""
if [ "$LR" != 3e-5 ]; then LT=", lr $LR"; LN="_lr$LR"; fi
SZ="model.dim=$DIM model.depth=$DEPTH"
RS=""; RT=""; LOG=log.txt
if [ -n "$RESUME" ]; then
  RSTEP=$(basename "$RESUME" .pt | sed 's/ckpt_//'); LOG=log_resume_$RSTEP.txt
  RS="init=$RESUME resume=true train.ckpt_every=${CKE:-10000}"; RT=", resumed at step $RSTEP"
fi
if [ $NET = fm ]; then
  n=sd_fm_${M}_K${K}${BN}${LN}_s${S}_n${ST}_pre$PRE
  case $M in
    pretrain) n=sd_fm_pretrain_d${DIM}x${DEPTH}${TXB:+_easy$TXB}_s$S
         CMD=".venv/bin/python scripts/pretrain.py configs/sudoku_pretrain.yaml out_dir=runs/$n $SZ ${TXB:+train_max_blanks=$TXB} train.ckpt_every=${CKE:-10000} seed=$S track.nb=true track.subproject=pretrain-sudoku 'track.question=$QP' track.algo=fmlm-direct $RS 'track.title=Flow map pretraining on Sudoku, width $DIM, depth $DEPTH${TXB:+, puzzles with at most $TXB blanks}$RT' 'track.formula=direct flow-map loss, puzzle \$\\to\$ solution, \$d = \\cfg{model.dim}\$, depth \$\\cfg{model.depth}\$'";;
    rl)  CMD=".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=$CK out_dir=runs/$n reward=exact rl.K=$K $B track.algo=flowmap-pg 'track.title=Flow map, RL, $K jump(s)$BT$LT, $TAIL' 'track.formula=\$r = \\mathbb{1}[\\text{grid solved}]\$, \$K = \\cfg{rl.K}\$, \$\\beta = \\cfg{rl.beta}\$' $COMMON";;
    sft) CMD=".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=$CK out_dir=runs/$n sft=true rl.K=$K track.algo=flowmap-sft 'track.title=Flow map, supervised fine-tuning, $K jump(s)$LT, $TAIL' 'track.formula=\$\\mathcal{L}_{sft}\$ on the solution, \$K = \\cfg{rl.K}\$' $COMMON";;
    pre) n=sd_fm_pre${LN}${TMB:+_hard$TMB}${TXB:+_easy$TXB}_s${S}_n${ST}_pre$PRE
         CMD=".venv/bin/python scripts/pretrain.py configs/sudoku_pretrain.yaml init=$CK out_dir=runs/$n $SZ ${TMB:+train_min_blanks=$TMB} ${TXB:+train_max_blanks=$TXB} keep_ckpts=false train.lr=$LR train.warmup=1 train.steps=$ST train.eval_every=250 eval.Ks=[1,4] seed=$S track.algo=fmlm-direct 'track.title=Flow map, continued pretraining, lr $LR${TMB:+, >=$TMB blanks}${TXB:+, <=$TXB blanks}, $TAIL' 'track.formula=direct flow-map loss resumed, \$\\eta = \\cfg{train.lr}\$' $TR";;
  esac
else
  n=sd_ar_${M}${BN}${LN}_s${S}_n${ST}_pre$PRE
  case $M in
    pretrain) n=sd_ar_pretrain_d${DIM}x${DEPTH}_s$S
         CMD=".venv/bin/python scripts/ar.py configs/sudoku_pretrain.yaml mode=pretrain out_dir=runs/$n $SZ seed=$S track.nb=true track.subproject=pretrain-sudoku 'track.question=$QP' track.algo=ar-ce $RS 'track.title=Autoregressive pretraining on Sudoku, width $DIM, depth $DEPTH$RT' 'track.formula=\$\\mathcal{L}_{ar}\$, puzzle \$\\to\$ solution left to right, \$d = \\cfg{model.dim}\$, depth \$\\cfg{model.depth}\$'";;
    rl)  CMD=".venv/bin/python scripts/ar.py configs/arith_rl.yaml init=$CK out_dir=runs/$n mode=rl reward=exact $B track.algo=ar-pg 'track.title=Autoregressive, RL$BT$LT, $TAIL' 'track.formula=\$r = \\mathbb{1}[\\text{grid solved}]\$, same advantages as the flow map, \$\\beta = \\cfg{rl.beta}\$' $COMMON";;
    sft) CMD=".venv/bin/python scripts/ar.py configs/arith_rl.yaml init=$CK out_dir=runs/$n mode=sft train.batch_size=256 track.algo=ar-ce 'track.title=Autoregressive, supervised fine-tuning$LT, $TAIL' 'track.formula=\$\\mathcal{L}_{ar}\$ on the solution' $COMMON";;
    pre) CMD=".venv/bin/python scripts/ar.py configs/sudoku_pretrain.yaml init=$CK out_dir=runs/$n mode=pretrain $SZ keep_ckpts=false train.lr=$LR train.warmup=1 train.steps=$ST train.eval_every=250 seed=$S track.algo=ar-ce 'track.title=Autoregressive, continued pretraining, lr $LR, $TAIL' 'track.formula=\$\\mathcal{L}_{ar}\$ resumed, \$\\eta = \\cfg{train.lr}\$' $TR";;
  esac
fi
mkdir -p runs/$n
tmux new-session -d -s $n "( $CMD ) > runs/$n/$LOG 2>&1"
echo launched $n
