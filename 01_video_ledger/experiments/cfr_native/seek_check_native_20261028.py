# -*- coding: utf-8 -*-
r"""指標⑤＝フレーム番号での頭出し（シーク）が当たるかを、今回作った帯動画で実測する（2026-10-28）。

やり方は 10/22 の seek_check_20261022.py と同じ。
頭から順に復号して基準のフレームを集め（ずれない読み方）、同じフレーム番号へ頭出しして読んだ画像が
基準のどのフレームと最も一致するかを探し、その差を「シークのずれ（フレーム）」とする。

対象＝cfr_native_20261028\bands\ に書き出した固定60fps の帯動画（混ぜて作った版・作り直した版）。
10/22 で測った A（可変）・B2（固定60）・D2（固定59.3354）の値はそのまま引き継ぐ。

出力＝report\seek_check_native.csv
使い方＝python seek_check_native_20261028.py [--n 20]
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_common_20261028 import OUT, ensure_out, log


def small(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (310, 15)).astype(np.float32)


def one(path, n_probe, win):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = [int(n * (i + 1) / (n_probe + 1)) for i in range(n_probe)]
    need = set()
    for t in targets:
        need.update(range(max(0, t - win), min(n, t + win + 1)))
    ref = {}
    i = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        if i in need:
            ref[i] = small(img)
    offs = []
    for t in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, img = cap.read()
        if not ok:
            continue
        s = small(img)
        best, bd = None, None
        for j in range(max(0, t - win), min(n, t + win + 1)):
            if j not in ref:
                continue
            d = float(np.abs(ref[j] - s).mean())
            if bd is None or d < bd:
                bd, best = d, j
        offs.append(best - t)
    cap.release()
    a = [abs(x) for x in offs]
    return dict(video=os.path.basename(path), n_frames=n, n_probe=len(offs),
                exact_hits=sum(1 for x in offs if x == 0),
                abs_median=int(sorted(a)[len(a) // 2]) if a else "",
                abs_max=max(a) if a else "", offsets=";".join(str(x) for x in offs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--win", type=int, default=90)
    A = ap.parse_args()
    ensure_out()
    LOG = os.path.join(OUT, "report", "seek_check_native.log")
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "bands", "*.mp4"))):
        r = one(p, A.n, A.win)
        rows.append(r)
        log("[seek] %-40s ぴったり %d/%d 最大 %s フレーム" % (r["video"][:40], r["exact_hits"],
                                                       r["n_probe"], r["abs_max"]), LOG)
    if rows:
        with open(os.path.join(OUT, "report", "seek_check_native.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    log("DONE seek_check_native", LOG)
