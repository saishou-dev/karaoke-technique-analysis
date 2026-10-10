#!/bin/bash
export LANG=C.UTF-8; cd ~/karaoke-vision
R="/mnt/g/研究アーカイブ/成果物/動画解析"
mkdir -p "$R/results_wao_20260911_v2"
for name in "天国_20260911_131710" "鯨の唄_20260911_125623"; do
  mkdir -p "$R/results_wao_20260911_v2/$name"
  ( taskset -c 4-7 .venv/bin/python read_counters_wao.py "$R/bands_wao/$name.mp4" "$R/results_wao_20260911_v2/$name" "$R/templates/wao_20260816/cluster_reps_wao_ext.npy" "$R/templates/wao_20260816/roi_wao.json" > "$R/results_wao_20260911_v2/$name/read_log.txt" 2>&1 ) &
  sleep 1
done; wait; echo TEST_DONE
