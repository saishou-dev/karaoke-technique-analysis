#!/bin/bash
# 読取ランナー×4 を WSL で完全分離起動（setsid nohup）。使い方: KV_SESSION=20260915 [KV_TARGET=N] bash launch_read_20260915.sh
# 事前に read_counters_wao.py と run_read_20260915.sh を ~/karaoke-vision にコピーする（このスクリプトが行う）。
S="${KV_SESSION:-20260915}"; SC="/mnt/g/研究アーカイブ/成果物/動画解析/scripts"
cp "$SC/read_counters_wao.py" ~/karaoke-vision/ && cp "$SC/run_read_20260915.sh" ~/karaoke-vision/ && chmod +x ~/karaoke-vision/run_read_20260915.sh
cd ~/karaoke-vision
for i in 0 1 2 3; do KV_SESSION="$S" KV_TARGET="${KV_TARGET:-0}" setsid nohup bash ./run_read_20260915.sh "$i" > /dev/null 2>&1 < /dev/null & done
sleep 3; pgrep -fa "run_read_20260915.sh [0-3]"
