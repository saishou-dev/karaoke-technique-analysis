#!/bin/bash
# 再読取(2026-09-03): kill時刻(03:10:31)にENDした4曲+二重起動の悪魔の子。旧結果は out_wao_redo_20260903/ へ退避。
# 既存ランナーは out dir 存在で skip するため、先に mkdir で先取りしてから読取(競合回避)。
export LANG=C.UTF-8 LC_ALL=C.UTF-8
KV=/home/toshiaki/karaoke-vision; cd $KV; source .venv/bin/activate
BANDS="/mnt/d/研究アーカイブ/成果物/動画解析/bands_wao"
mkdir -p out_wao_redo_20260903
CPUS="$1"; shift
for name in "$@"; do
  out="out_wao/$name"
  if [ -d "$out" ]; then mv "$out" "out_wao_redo_20260903/${name}__old_$(date +%H%M%S)"; fi
  mkdir "$out" || { echo "claim failed $name"; continue; }
  echo "=== REREAD $name $(date +%H:%M:%S) cpus=$CPUS ==="
  taskset -c "$CPUS" python -u read_counters.py "$BANDS/$name.mp4" "$out" out/cluster_reps_wao_ext.npy roi_wao.json > "$out/read_log.txt" 2>&1
  grep -E "readable|final" "$out/read_log.txt" | head -12
  echo "=== END $name $(date +%H:%M:%S) ==="
done
echo REREAD_DONE
