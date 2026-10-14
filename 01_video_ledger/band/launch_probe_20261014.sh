#!/bin/bash
# 帯プローブを WSL で完全分離起動（setsid nohup）2026-10-14 版。launch_read_20260915.sh と同じ「スクリプトファイル経由」の型
# （9/15・9/24 は監督からのインライン起動＝bash -lc "… &" が無反応だった。読取ランナーのスクリプト経由起動は毎回成功している）。
# 使い方: KV_SESSION=20261014 KV_PROBE_SCRIPT=wao_probe_v2_20260915.py KV_PROBE_LOG=/mnt/g/.../band_probe_v2_20261014.log \
#         bash launch_probe_20261014.sh /mnt/g/.../pipeline_20261014/probe_videos.txt
#   第1引数＝対象動画の WSL パス一覧（1行1本・UTF-8）。プローブは一覧の動画だけを処理する（他機種の動画を拾わない）。
export LANG=C.UTF-8
S="${KV_SESSION:?KV_SESSION}"; SC="/mnt/g/研究アーカイブ/成果物/動画解析/scripts"
LIST="$1"; [ -f "$LIST" ] || { echo "ERR list not found: $LIST"; exit 2; }
mapfile -t V < <(sed 's/\r$//' "$LIST" | grep -v '^$')
[ "${#V[@]}" -gt 0 ] || { echo "ERR list empty"; exit 2; }
cd ~/karaoke-vision || exit 1
KV_SESSION="$S" setsid nohup .venv/bin/python "$SC/${KV_PROBE_SCRIPT:?KV_PROBE_SCRIPT}" "$S" "${V[@]}" >> "${KV_PROBE_LOG:?KV_PROBE_LOG}" 2>&1 < /dev/null &
sleep 3
echo "launched videos=${#V[@]} procs=$(pgrep -f "${KV_PROBE_SCRIPT%.py}" | wc -l)"
