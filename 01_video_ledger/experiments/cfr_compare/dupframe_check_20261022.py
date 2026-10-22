# -*- coding: utf-8 -*-
r"""固定化で入った重複フレームと、失われた中身を実測する（2026-10-22）。

ffmpeg の stats が出す dup=／drop= は信用せず、できた切り出し映像を1フレームずつ復号して
「直前のフレームと中身が同じか」を数える。重複フレーム＝新しい情報が入っていないフレーム。
あわせて「中身が違うフレーム（＝実質の情報量）」が元録画のフレーム数に対してどれだけ残ったかを出す。

判定＝縮小したグレースケール（310x15）の平均絶対差が THRESH 未満なら「直前と同じ」。
再エンコードの誤差があるので 0 ではなく小さなしきい値を使い、しきい値を 3 段で出して感度も見る。

出力＝report\dupframe.csv／dupframe.log
使い方＝python dupframe_check_20261022.py [--only B_cfr60] [--videos 1]
"""
import argparse
import concurrent.futures as cf
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfr_common_20261022 import OUT, VARIANTS, VIDEOS, band_path, ensure_out, log

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--videos", type=int, default=0, help="先頭から何曲ぶん測るか（0＝全部）")
ap.add_argument("--workers", type=int, default=6)
A = ap.parse_args()

ensure_out()
REP = os.path.join(OUT, "report")
LOG = os.path.join(REP, "dupframe.log")
THRESHS = (0.05, 0.5, 2.0)
rows = []
want_v = [k for k in VARIANTS if not A.only or k in A.only.split(",")]
want_s = VIDEOS[:A.videos] if A.videos else VIDEOS

def one(job):
    variant, video = job
    path = band_path(variant, video)
    if not os.path.exists(path):
        return None, "[skip] 切り出し映像が無い %s %s" % (variant, video)
    cap = cv2.VideoCapture(path)
    prev = None
    n = 0
    same = {t: 0 for t in THRESHS}
    diffs = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (310, 15)).astype(np.float32)
        if prev is not None:
            d = float(np.mean(np.abs(g - prev)))
            diffs.append(d)
            for t in THRESHS:
                if d < t:
                    same[t] += 1
        prev = g
    cap.release()
    ds = sorted(diffs)
    r = dict(video=video, variant=variant, n_frames=n,
             d_median=round(ds[len(ds) // 2], 4), d_p05=round(ds[len(ds) // 20], 4),
             d_min=round(ds[0], 4), d_max=round(ds[-1], 4))
    for t in THRESHS:
        r["same_lt_%g" % t] = same[t]
        r["unique_lt_%g" % t] = n - same[t]
    msg = ("[dup] %-26s %-18s 全%dフレーム｜直前と同じ: <0.05 %d／<0.5 %d／<2.0 %d ｜中身が違うフレーム: %d／%d／%d\n"
           "      連続フレーム差 min=%.4f p05=%.4f 中央=%.4f max=%.4f"
           % (video[:26], variant, n, same[0.05], same[0.5], same[2.0],
              n - same[0.05], n - same[0.5], n - same[2.0],
              r["d_min"], r["d_p05"], r["d_median"], r["d_max"]))
    return r, msg


jobs = [(v, s) for s in want_s for v in want_v]
with cf.ThreadPoolExecutor(max_workers=A.workers) as ex:
    for r, msg in ex.map(one, jobs):
        log(msg, LOG)
        if r:
            rows.append(r)

path = os.path.join(REP, "dupframe.csv")
old = []
if os.path.exists(path):
    old = [x for x in csv.DictReader(open(path, encoding="utf-8"))
           if (x["variant"], x["video"]) not in {(r["variant"], r["video"]) for r in rows}]
cols = list(rows[0].keys())
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in old + rows:
        w.writerow({k: r.get(k, "") for k in cols})
log("DONE dupframe.csv", LOG)
