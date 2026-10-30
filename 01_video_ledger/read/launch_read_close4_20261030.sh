#!/bin/bash
# close4 用の読取ランナー×3 を WSL で分離起動（2026-10-30）。使い方: bash launch_read_close4_20261030.sh <jobs.tsv>
# 止め方＝ジョブ表を空にする（各ランナーは読み中の1曲を終えたら抜ける・プロセスの強制終了はしない）。
SC="/mnt/g/研究アーカイブ/成果物/動画解析/scripts"; J="$1"
cp "$SC/run_read_close4_20261030.sh" ~/karaoke-vision/ && cd ~/karaoke-vision || exit 1
for i in 0 1 2; do KV_JOBS="$J" setsid nohup bash ./run_read_close4_20261030.sh "$i" > /dev/null 2>&1 < /dev/null & sleep 2; done
sleep 3; pgrep -fa "run_read_close4_20261030.sh [0-2]"
