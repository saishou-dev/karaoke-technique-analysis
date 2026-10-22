# -*- coding: utf-8 -*-
r"""固定化の副作用（良い方）＝フレーム番号での頭出し（シーク）が当たるようになるかを測る（2026-10-22）。

現行の帯動画は passthrough＝可変フレームレートなので、cv2 の CAP_PROP_POS_FRAMES でフレーム番号を指定した
頭出しはずれる（RUNBOOK §5D・2026-10-15 実測＝I LOVE... しゃくり15 で 7 フレーム差／目視追記10件のうち4件が最大56フレーム＝0.96秒ずれ）。
固定フレームレートにすればこのずれが消えるのかを、同じ帯動画の A と B・C で実測する。

やり方＝頭から順に復号して基準のフレームを集め（ずれない読み方）、同じフレーム番号へ頭出しして読んだ画像が
基準のどのフレームと最も一致するかを探し、その差を「シークのずれ（フレーム）」とする。

出力＝report\seek_check.csv／seek_check.log
使い方＝python seek_check_20261022.py [--n 20]
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
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--win", type=int, default=90)
ap.add_argument("--workers", type=int, default=6)
A = ap.parse_args()

ensure_out()
REP = os.path.join(OUT, "report")
LOG = os.path.join(REP, "seek_check.log")
rows = []


def small(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (310, 15)).astype(np.float32)


def one(job):
        variant, video = job
        path = band_path(variant, video)
        if not os.path.exists(path):
            return None, "[skip] 帯動画が無い %s %s" % (variant, video)
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        targets = [int(n * (i + 1) / (A.n + 1)) for i in range(A.n)]
        need = set()
        for t in targets:
            need.update(range(max(0, t - A.win), min(n, t + A.win + 1)))
        ref = {}
        i = -1
        while True:
            ok, img = cap.read()
            if not ok:
                break
            i += 1
            if i in need:
                ref[i] = small(img)
        cap.release()
        cap2 = cv2.VideoCapture(path)
        offs = []
        for t in targets:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, img = cap2.read()
            if not ok:
                continue
            q = small(img)
            best, bestd = None, None
            for k in range(max(0, t - A.win), min(n, t + A.win + 1)):
                if k not in ref:
                    continue
                d = float(np.mean(np.abs(ref[k] - q)))
                if bestd is None or d < bestd:
                    bestd, best = d, k
            if best is not None:
                offs.append(best - t)
        cap2.release()
        a = sorted(abs(x) for x in offs)
        r = dict(video=video, variant=variant, n_frames=n, n_probe=len(offs),
                 exact_hits=sum(1 for x in offs if x == 0),
                 abs_median=(a[len(a) // 2] if a else ""), abs_max=(a[-1] if a else ""),
                 signed_min=(min(offs) if offs else ""), signed_max=(max(offs) if offs else ""),
                 offsets=";".join(str(x) for x in offs))
        return r, ("[seek] %-26s %-18s %d点中 ぴったり%d点｜ずれ 中央%s フレーム 最大%s フレーム（%s〜%s）"
                   % (video[:26], variant, len(offs), r["exact_hits"], r["abs_median"], r["abs_max"],
                      r["signed_min"], r["signed_max"]))


jobs = [(v, s) for s in VIDEOS for v in VARIANTS]
with cf.ThreadPoolExecutor(max_workers=A.workers) as ex:
    for r, msg in ex.map(one, jobs):
        log(msg, LOG)
        if r:
            rows.append(r)

with open(os.path.join(REP, "seek_check.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
log("DONE seek_check.csv", LOG)
