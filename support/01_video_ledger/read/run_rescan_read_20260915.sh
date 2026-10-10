#!/bin/bash
# 9/15 再走査 第2段＝較正済み ROI で読み直す（WSL）。使い方: KV_PAR=6 [KV_PREP=clahe] bash run_rescan_read_20260915.sh <video_name ...>（省略＝roi/*.json 全部）
# 出力: results_wao_20260915_rescan/<video>/{counter_events.csv,counter_events.json,read_log.txt}。既存の results_wao_20260915 は触らない。
export LANG=C.UTF-8
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; KV=~/karaoke-vision; S=20260915
RES="$ROOT/results_wao_${S}_rescan${KV_SUFFIX:-}"; ROI="$ROOT/dataset/rescan_$S/roi"; LOG="$RES/_runner.log"; mkdir -p "$RES"
cp "$ROOT/scripts/read_counters_wao_20260915.py" $KV/
PAR="${KV_PAR:-6}"
# 曲名に空白があるので1行1曲で書く（xargs basename は空白で割れる＝17:58 に5曲取りこぼした原因）
if [ $# -gt 0 ]; then printf '%s\n' "$@"; else for j in "$ROI"/*.json; do b=$(basename "$j"); echo "${b%.json}"; done; fi > /tmp/rescan_list_$S.txt
echo "[$(date +%H:%M:%S)] start par=$PAR prep=${KV_PREP:-none} n=$(wc -l < /tmp/rescan_list_$S.txt)" >> "$LOG"
one() {
  name="$1"; out="$RES/$name"; mkdir -p "$out"
  [ -f "$out/counter_events.csv" ] && { echo "[$(date +%H:%M:%S)] skip $name" >> "$LOG"; return; }
  echo "[$(date +%H:%M:%S)] START $name" >> "$LOG"
  $KV/.venv/bin/python $KV/read_counters_wao_20260915.py "$ROOT/bands_wao/$name.mp4" "$out" "$ROOT/templates/wao_20260816/cluster_reps_wao_ext.npy" "$ROI/$name.json" > "$out/read_log.txt" 2>&1
  echo "[$(date +%H:%M:%S)] END $name rc=$?" >> "$LOG"
}
export -f one; export RES ROI KV ROOT LOG KV_PREP KV_MAXFRAMES
tr '\n' '\0' < /tmp/rescan_list_$S.txt | xargs -0 -n1 -P "$PAR" bash -c 'one "$0"'
echo "[$(date +%H:%M:%S)] ALL DONE" >> "$LOG"
