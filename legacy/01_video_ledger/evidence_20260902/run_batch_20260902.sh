#!/bin/bash
# 9/2全29本: クロップ完了(.partなし=mv済み)を待ちながら読取を進めるランナー(並列安全: mkdirロック)
# v2: taskset でCPUを固定(引数1=CPUリスト 例 0-2)。OpenCVスレッドの過剰生成(5プロセス×16スレッドのスピン待ち)で
#     1本30分超に劣化した問題の対策。読取結果は決定的でスレッド数に依存しない。
export LANG=C.UTF-8 LC_ALL=C.UTF-8
CPUS="${1:-0-15}"
cd /home/toshiaki/karaoke-vision
source .venv/bin/activate
BANDS="/mnt/d/研究アーカイブ/成果物/動画解析/bands_wao"
CSV="/home/toshiaki/karaoke-vision/band_scale_final_20260902.csv"
total=$(tail -n +2 "$CSV" | wc -l)
while true; do
  done_n=$(find out_wao -maxdepth 2 -name counter_events.csv -path "*_20260902_*" | wc -l)
  if [ "$done_n" -ge "$total" ]; then break; fi
  [ -f /home/toshiaki/karaoke-vision/STOP_20260902 ] && { echo "STOP flag found"; exit 2; }
  claimed=0
  while IFS= read -r name; do
    band="$BANDS/$name.mp4"
    out="out_wao/$name"
    if [ -f "$out/counter_events.csv" ] || [ -d "$out" ]; then continue; fi
    if [ -f "$band" ]; then
      if mkdir "$out" 2>/dev/null; then
        sleep 1
        if [ "$(pgrep -fc "read_counters.py $band ")" -ge 1 ]; then echo "DUP-GUARD $name (already being read)"; continue; fi
        claimed=1
        echo "=== READ $name $(date +%H:%M:%S) cpus=$CPUS ==="
        taskset -c "$CPUS" python -u read_counters.py "$band" "$out" out/cluster_reps_wao_ext.npy roi_wao.json > "$out/read_log.txt" 2>&1
        grep -E "final|rescued" "$out/read_log.txt" | head -8
        echo "=== END $name $(date +%H:%M:%S) ==="
      fi
    fi
  done < <(tail -n +2 "$CSV" | cut -d, -f1)
  if [ "$claimed" -eq 0 ]; then sleep 20; fi
done
echo ALL_READS_DONE
