#!/bin/sh
# List lab machines whose GPU looks idle, best first. Usage: tools/free_gpus.sh [max_mem_used_MiB] [max_util_%]
# Rule: never run on more than 10 machines at once.
MAXMEM=${1:-2000}; MAXUTIL=${2:-10}
DIR=$(cd "$(dirname "$0")" && pwd)
probe() {
  r=$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$1" 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; cut -d" " -f1 /proc/loadavg; nproc; tmux ls 2>/dev/null | wc -l' 2>/dev/null | tr '\n' ',')
  [ -n "$r" ] && echo "$1,$r"
}
export -f probe 2>/dev/null
grep -v '^#' "$DIR/machines.txt" | xargs -P 16 -n 1 sh -c "$(declare -f probe 2>/dev/null || type probe | tail -n +2); probe \"\$0\"" 2>/dev/null \
  | awk -F, -v M="$MAXMEM" -v U="$MAXUTIL" '$3+0<=M && $5+0<=U {printf "%-12s %-28s mem %5d/%5d MiB  util %3d%%  load %s/%s  our_tmux %s\n",$1,$2,$3,$4,$5,$6,$7,$8}' \
  | sort -t'/' -k2 -rn
