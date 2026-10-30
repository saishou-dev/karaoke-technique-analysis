#!/usr/bin/env bash
# lane_z.sh = 条件Z（全画面のフレーム）の1レーンを走らせる（2026-10-30）
#   bash lane_z.sh extract          … 元動画を逐次デコードして sub4 のフレームを保存＋帯との対応確認
#   bash lane_z.sh run <手法>        … easyocr_readtext / paddle_full / tess_psm11 のどれか1つ
# 1プロセス専用4コア（OMP_NUM_THREADS=4）。進捗は ~/ocr_gran_20261029/heartbeat_z.txt に追記する。
# Windows 側から Start-Process（Hidden）で `wsl -e bash -lc "bash lane_z.sh ... # gran_z_lane"` の形で起こす。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/ocr_compare_20261018/.venv312/bin/python"
BASE="$HOME/ocr_gran_20261029"
HB="$BASE/heartbeat_z.txt"
mkdir -p "$BASE/logs" "$BASE/full_z"
. "$HOME/ocr_compare_20261018/env.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# 2026-10-30 の割り付け（試走の実測＝Tesseract 約1秒・EasyOCR 約20秒・PaddleOCR 一式 約75秒／フレーム）
#   tess_psm11       … sub4 の全フレーム（対応が確かめられた曲だけ）
#   easyocr_readtext … --every 32（標本番号が32の倍数）
#   paddle_full      … --every 64（標本番号が64の倍数）
# 2026-10-30 10:50 追加の制約＝同時2本まで（4コア×2）・nice 15・元動画と帯 npy は1回だけ読む
# 全部の予測がそろった時点で heartbeat_z.txt に Z_DONE を書く
export Z_EXPECT="pred_Z_tess_psm11.csv pred_Z_easyocr_readtext.csv pred_Z_paddle_full.csv"
mode="$1"
if [ "$mode" = "extract" ]; then
  echo "START extract $(date +%F' '%H:%M:%S)" >> "$HB"
  shift; nice -n 15 "$PY" "$HERE/run_gran_z.py" extract --out "$BASE/full_z" "$@" >> "$BASE/logs/Z_extract.log" 2>&1
  echo "RC=$? extract $(date +%F' '%H:%M:%S)" >> "$HB"
else
  cand="$2"; shift 2; tag="$cand$(echo "$@" | tr -c 'a-z0-9' '_')"
  echo "START Z_$tag $(date +%F' '%H:%M:%S)" >> "$HB"
  nice -n 15 "$PY" "$HERE/run_gran_z.py" run --cand "$cand" --frames "$BASE/full_z" --out "$HERE/pred" "$@" \
    > "$BASE/logs/Z_${tag}.log" 2>&1
  echo "RC=$? Z_$tag $(date +%F' '%H:%M:%S)" >> "$HB"
fi
