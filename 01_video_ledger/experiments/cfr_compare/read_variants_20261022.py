# -*- coding: utf-8 -*-
r"""手順3前半＝各方式の帯動画に現行の読み取り（カウンターの増分検出）をかける（2026-10-22）。

読取器は現行と同じ v0.8e（説明用の単一ファイル版 read_counters_wao_single_20261015.py・出力は皮版と md5 一致）。
ROI は曲ごとの較正値（results_wao_20261014\<video>\roi.json）をそのまま使う＝読取条件は全方式で同一にして、
フレームレートの扱いの違いだけを見る。

出力＝dataset\cfr_compare_2026-10-22\results\<variant>__<video>\counter_events.csv ＋ read_log.txt
使い方＝python read_variants_20261022.py [--only A_passthrough] [--workers 3]
"""
import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfr_common_20261022 import (OUT, READER, REPS, RESULTS_EXIST, VARIANTS, VIDEOS,
                                 band_path, ensure_out, log, result_dir)

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--workers", type=int, default=3)
A = ap.parse_args()

ensure_out()
LOG = os.path.join(OUT, "report", "read_variants.log")
PY = sys.executable
WANT = [k for k in VARIANTS if not A.only or k in A.only.split(",")]


def one(job):
    variant, video = job
    out = result_dir(variant, video)
    os.makedirs(out, exist_ok=True)
    roi_src = os.path.join(RESULTS_EXIST, video, "roi.json")
    roi = os.path.join(out, "roi.json")
    if os.path.exists(roi_src):
        shutil.copyfile(roi_src, roi)
    else:
        return variant, video, "ERR roi.json が無い %s" % roi_src
    band = band_path(variant, video)
    if not os.path.exists(band):
        return variant, video, "ERR 帯動画が無い %s" % band
    t0 = time.time()
    p = subprocess.run([PY, "-X", "utf8", READER, band, out, REPS, roi],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    with open(os.path.join(out, "read_log.txt"), "w", encoding="utf-8") as f:
        f.write(p.stdout or "")
        if p.returncode != 0:
            f.write("\n[stderr]\n" + (p.stderr or ""))
    if p.returncode != 0:
        return variant, video, "ERR rc=%d %s" % (p.returncode, (p.stderr or "")[-200:].replace("\n", " "))
    return variant, video, "ok %.0fs" % (time.time() - t0)


jobs = [(v, s) for v in WANT for s in VIDEOS
        if not os.path.exists(os.path.join(result_dir(v, s), "counter_events.csv"))]
log("[read] jobs=%d workers=%d reader=%s" % (len(jobs), A.workers, os.path.basename(READER)), LOG)
with cf.ThreadPoolExecutor(max_workers=A.workers) as ex:
    for v, s, msg in ex.map(one, jobs):
        log("[read] %-14s %-26s %s" % (v, s[:26], msg), LOG)
log("DONE read_variants", LOG)
