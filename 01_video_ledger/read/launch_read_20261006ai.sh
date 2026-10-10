#!/bin/bash
# 2026-10-06 版＝読取器を read_counters_ai_20261006ai.py（ROI 輝度正規化）に差し替えただけ（launch_read_20260915ai.sh の複製）
# 読取ランナー×4 を WSL で完全分離起動（setsid nohup）DAM Ai 版。使い方: KV_SESSION=20260915 [KV_TARGET=N] [KV_AI_BASE=<wslパス>] bash launch_read_20260915ai.sh
# 事前に read_counters_ai_20261006ai.py と run_read_20261006ai.sh を ~/karaoke-vision にコピーする（このスクリプトが行う）。
S="${KV_SESSION:-20260915}"; SC="/mnt/g/研究アーカイブ/成果物/動画解析/scripts"
cp "$SC/read_counters_ai_20261006ai.py" ~/karaoke-vision/ && cp "$SC/run_read_20261006ai.sh" ~/karaoke-vision/ && chmod +x ~/karaoke-vision/run_read_20261006ai.sh
cd ~/karaoke-vision
N="${KV_RUNNERS:-4}"
for i in $(seq 0 $((N-1))); do KV_SESSION="$S" KV_TARGET="${KV_TARGET:-0}" KV_AI_BASE="${KV_AI_BASE:-}" setsid nohup bash ./run_read_20261006ai.sh "$i" > /dev/null 2>&1 < /dev/null & done
sleep 3; pgrep -fa "run_read_20261006ai.sh [0-3]"
