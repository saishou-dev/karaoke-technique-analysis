#!/bin/bash
pkill -f "run_read_2026091[1].sh"; pkill -f "read_counters_wa[o].py"; sleep 2; rm -rf /tmp/claims_20260911
R="/mnt/g/研究アーカイブ/成果物/動画解析/results_wao_20260911"
cd "$R" && for d in *; do case "$d" in ダーリン_20260911_103233|_runner.log) ;; *) rm -rf "$d";; esac; done
echo "[$(date +%H:%M:%S)] --- clean relaunch v2 (no .part) ---" >> "$R/_runner.log"
cp "/mnt/g/研究アーカイブ/成果物/動画解析/scripts/run_read_20260911.sh" ~/karaoke-vision/ && chmod +x ~/karaoke-vision/run_read_20260911.sh
bash "/mnt/g/研究アーカイブ/成果物/動画解析/scripts/launch_read_20260911.sh"
sleep 10; bash "/mnt/g/研究アーカイブ/成果物/動画解析/scripts/status_read_20260911.sh"
