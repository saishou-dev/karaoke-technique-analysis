#!/bin/bash
# 2026-10-22 作成（Opus 部下Agent）。やり直し実行の事前準備＝30曲ぶんの claim を先取りする。
# 目的＝前任の監督が残した読取ランナー（run_read_20260915.sh・02:49 時点で36本）が、
#       crop で生まれる新しい帯動画を calib の前に既定 ROI で読んでしまうのを防ぐ。
# 監督は read 工程の頭で `rm -rf /tmp/claims_20261021` をしてから自分のランナーを起動するので、
# この先取りは read 工程に入った瞬間に自動で解除される（その時点では roi.json が揃っている）。
export LANG=C.UTF-8
S=20261021
LIST="/mnt/g/研究アーカイブ/成果物/動画解析/dataset/pipeline_20261021/_names30_20261022.txt"
CLAIM="/tmp/claims_$S"
mkdir -p "$CLAIM"
n=0
while IFS= read -r x; do
  x="${x%$'\r'}"
  [ -n "$x" ] || continue
  mkdir -p "$CLAIM/$x" && n=$((n+1))
done < "$LIST"
echo "read=$n claims=$(ls "$CLAIM" | wc -l)"
