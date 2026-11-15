# Countdown, from checkpoints at 20k pretraining steps: RL vs supervised fine-tuning on the generating expressions,
# for the flow map and the autoregressive baseline. args: fm rl|sft K | ar rl|sft
cd ~/projects/RL-for-discrete-flow-map
NET=$1; M=$2; K=${3:-1}
Q='On Countdown (several valid answers), does RL improve on supervised fine-tuning with one valid answer per question, for the flow map and for an autoregressive transformer of the same size?'
S=${4:-0}; ST=${5:-5000}
COMMON="train.lr=3e-5 train.steps=$ST eval.every=250 seed=$S track.nb=true track.subproject=rl-countdown 'track.question=$Q'"
if [ $NET = fm ]; then
  CK=runs/ckpt_cd_fm_20k.pt; n=cd_fm_${M}_K${K}_s${S}_n$ST
  if [ $M = rl ]; then X="reward=exact rl.K=$K track.algo=flowmap-pg 'track.title=Flow map, RL, $K jump(s), seed $S, $ST steps' 'track.formula=\$r = \\mathbb{1}[e(n) = v]\$, \$K = \\cfg{rl.K}\$'";
  else X="sft=true rl.K=$K track.algo=flowmap-sft 'track.title=Flow map, supervised fine-tuning, $K jump(s), seed $S, $ST steps' 'track.formula=\$\\mathcal{L}_{sft}\$ on the generating expression, \$K = \\cfg{rl.K}\$'"; fi
  CMD=".venv/bin/python scripts/rl.py configs/arith_rl.yaml init=$CK out_dir=runs/$n $X $COMMON"
else
  CK=runs/ckpt_cd_ar_20k.pt; n=cd_ar_${M}_s${S}_n$ST
  if [ $M = rl ]; then X="mode=rl reward=exact track.algo=ar-pg 'track.title=Autoregressive, RL, seed $S, $ST steps' 'track.formula=\$r = \\mathbb{1}[e(n) = v]\$, same advantages as the flow map'";
  else X="mode=sft train.batch_size=256 track.algo=ar-ce 'track.title=Autoregressive, supervised fine-tuning, seed $S, $ST steps' 'track.formula=\$\\mathcal{L}_{ar}\$ on the generating expression'"; fi
  CMD=".venv/bin/python scripts/ar.py configs/arith_rl.yaml init=$CK out_dir=runs/$n $X $COMMON"
fi
mkdir -p runs/$n
tmux new-session -d -s $n "( $CMD ) > runs/$n/log.txt 2>&1"
echo launched $n
