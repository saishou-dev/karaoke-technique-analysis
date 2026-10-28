# -*- coding: utf-8 -*-
r"""手順1＝1フレームごとの「読めた値」と「直前フレームとの画素差」を書き出す（2026-10-28）。

数字の切り出し・正規化・照合は読取器 v0.8e（read_counters_wao_single_20261015.py）から一字一句そのまま写した。
違うのは「読んだ結果をその場で増分検出まで通すのではなく、生のまま保存する」ところだけ。
こうしておけば、安定の数え方や複製の扱いを変えた規則を OCR のやり直しなしで何通りも当てられる。

入力＝native_common_20261028.SOURCES の3種類
  file  … 既にある固定fps の帯動画をそのまま1フレームずつ読む
  blend … 可変のままの帯動画＋その pts 表から、60fps の格子の各時刻 t=k/60 で
          その時刻を挟む前後2フレームを時間の重み（線形）で混ぜた画を作って読む。
          同じ画を2回出さない＝複製フレームが存在しない固定60fps。

出力＝cfr_native_20261028\dumps\<source>__<video>.npz
  diff_prev  … 直前フレームとの平均絶対差（V=max(B,G,R) の画素値・0〜255）
  fr_<名前> / val_<名前> … そのカウンターで値が読めたフレーム番号と値
  meta       … fps・フレーム数・入力の種類（json 文字列）

使い方＝python dump_reads_20261028.py [--only g60,blend60] [--videos 0,1] [--workers 3]
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_common_20261028 import (COUNTERS, OUT, REPS, SOURCES, VIDEOS, dump_path, ensure_out, log,
                                    roi_path)

# ===== 以下 read_counters_wao_single_20261015.py からの写し（挙動を変えないため手を入れない）=====
THRESHOLDS = (215, 200, 185, 228, 170, 150, 130)
NORM = (28, 40)
NCC_MIN = 0.75
MIN_GAP2 = int(os.environ.get("KV_MIN_GAP2", "5"))

LABELS_DEFAULT = {0: 0, 1: 2, 2: 3, 3: 1, 4: 2, 5: 6, 6: 1, 7: 2, 8: 1, 9: 7,
                  10: 4, 11: 5, 12: 6, 13: 9, 14: None, 15: 5, 16: 0, 17: 0, 18: None,
                  19: 2, 20: 1, 21: 6, 22: 1, 23: 0, 24: 2, 25: 2, 26: 2, 27: 2,
                  28: None, 29: 6, 30: 2, 31: 8, 32: 2, 33: 4, 34: 0, 35: 4,
                  36: None, 37: 7, 38: None, 39: 8}
_LJ = os.path.splitext(REPS)[0] + "_labels.json"
LABELS = ({int(k): v for k, v in json.load(open(_LJ)).items()} or LABELS_DEFAULT) if os.path.exists(_LJ) else LABELS_DEFAULT

_reps = np.load(REPS)
_rf = []
for _r in _reps:
    _v = _r.ravel() - _r.mean()
    _rf.append(_v / max(1e-6, np.linalg.norm(_v)))
REPS_FLAT = np.stack(_rf)


def glyph_boxes(roi_gray):
    best1 = None
    best2 = None
    for T in THRESHOLDS:
        m = (roi_gray > T).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        boxes = []
        junk = 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 28 <= h <= 58 and 3 <= w <= 44 and area >= 50:
                boxes.append((x, y, w, h))
            elif area > 30:
                junk += area
        if not (1 <= len(boxes) <= 2 and junk <= 1200):
            continue
        boxes.sort(key=lambda b: b[0])
        if len(boxes) == 2:
            gap = boxes[1][0] - (boxes[0][0] + boxes[0][2])
            hdiff = abs(boxes[0][3] - boxes[1][3])
            ydiff = abs(boxes[0][1] - boxes[1][1])
            if gap > 14 or gap < MIN_GAP2 or hdiff > 8 or ydiff > 6:
                continue
            if best2 is None:
                best2 = boxes
        elif best1 is None:
            best1 = boxes
    return best2 if best2 is not None else best1


def norm_glyph(roi_gray, box):
    x, y, w, h = box
    pad = 2
    y0, y1 = max(0, y - pad), min(roi_gray.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(roi_gray.shape[1], x + w + pad)
    g = roi_gray[y0:y1, x0:x1].astype(np.float32)
    g = (g - g.min()) / max(1.0, (g.max() - g.min()))
    return cv2.resize(g, NORM)


def classify(gl):
    v = gl.ravel() - gl.mean()
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    scores = REPS_FLAT @ (v / n)
    bi = int(np.argmax(scores))
    if scores[bi] < NCC_MIN:
        return None
    return LABELS.get(bi)
# ===== 写しここまで =====


def read_one_frame(g, rois, fidx, store):
    """V=max(B,G,R) の1フレームから6カウンターを読み、読めたものだけ store に積む。"""
    for name, roi in rois.items():
        x1, y1, x2, y2 = roi
        rg = g[y1:y2, x1:x2]
        boxes = glyph_boxes(rg)
        if boxes is None:
            continue
        digits = []
        ok = True
        for b in boxes:
            d = classify(norm_glyph(rg, b))
            if d is None:
                ok = False
                break
            digits.append(d)
        if not ok:
            continue
        val = digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]
        store[name][0].append(fidx)
        store[name][1].append(val)


def gen_file(path):
    """固定fps の帯動画をそのまま1フレームずつ返す。"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    yield ("meta", fps, nframes)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        yield ("frame", img)
    cap.release()


def gen_blend(path, pts_csv, rate, gap_only=False):
    """可変のままの帯動画＋pts 表から、60fps の格子の各時刻で前後フレームを時間の重みで混ぜて返す。

    出力フレーム k の時刻 t=k/rate について、P[i] <= t <= P[i+1] となる前後2フレームを
    w=(t-P[i])/(P[i+1]-P[i]) で混ぜる（w=0 なら前のフレームそのまま）。
    抜けフレームの位置（間隔が2倍・3倍の所）では w が 0.33〜0.67 になり「前後の中間」が入る。
    間隔がそろっている所では w が 0 付近／1 付近になり、ほぼ元のフレームがそのまま出る。
    同じ画を2度出さないので、複製フレームによる安定ランの伸びが起きない。
    """
    P = [float(r["pts_sec"]) for r in csv.DictReader(open(pts_csv, encoding="utf-8"))]
    _d = sorted(P[i + 1] - P[i] for i in range(len(P) - 1))
    med = _d[len(_d) // 2] if _d else 1.0 / rate
    cap = cv2.VideoCapture(path)
    nsrc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = min(len(P), nsrc)
    P = P[:N]
    nout = int(P[-1] * rate) + 1
    yield ("meta", rate, nout)
    ok, cur = cap.read()
    if not ok:
        cap.release()
        return
    i = 0           # cur のフレーム番号
    nxt = None      # i+1 のフレーム
    for k in range(nout):
        t = k / rate
        while i + 1 < N and P[i + 1] <= t:
            if nxt is None:
                ok, nxt = cap.read()
                if not ok:
                    break
            cur = nxt
            nxt = None
            i += 1
        if t <= P[i] or i + 1 >= N:
            yield ("frame", cur)
            continue
        if nxt is None:
            ok, nxt = cap.read()
            if not ok:
                yield ("frame", cur)
                continue
        if gap_only and (P[i + 1] - P[i]) <= med * 1.4:
            # 間隔がそろっている所＝近い方の元フレームをそのまま使う（複製が残る）
            yield ("frame", cur if (t - P[i]) <= (P[i + 1] - t) else nxt)
            continue
        w = (t - P[i]) / max(1e-9, P[i + 1] - P[i])
        if w <= 1e-6:
            yield ("frame", cur)
        else:
            yield ("frame", cv2.addWeighted(cur, 1.0 - w, nxt, w, 0.0))
    cap.release()


def one(job):
    source, video = job
    out = dump_path(source, video)
    if os.path.exists(out):
        return source, video, "skip（既にある）", 0.0
    S = SOURCES[source]
    band = S["band"] % video
    if not os.path.exists(band):
        return source, video, "ERR 帯動画が無い %s" % band, 0.0
    rois = {k: tuple(v) for k, v in json.load(open(roi_path(video), encoding="utf-8")).items()}
    store = {k: ([], []) for k in COUNTERS}
    t0 = time.time()
    if S["kind"] in ("blend", "gapblend"):
        gen = gen_blend(band, S["pts"] % video, S["rate"], gap_only=(S["kind"] == "gapblend"))
    else:
        gen = gen_file(band)
    kind, fps, nframes = next(gen)
    assert kind == "meta"
    diffs = np.zeros(nframes + 16, dtype=np.float32)
    prev = None
    fidx = -1
    for kind, img in gen:
        fidx += 1
        g = img.max(axis=2)
        if fidx < len(diffs):
            if prev is not None:
                diffs[fidx] = float(np.abs(g.astype(np.int16) - prev).mean())
            prev = g.astype(np.int16)
        read_one_frame(g, rois, fidx, store)
    n_real = fidx + 1
    arrs = dict(diff_prev=diffs[:n_real])
    for k in COUNTERS:
        arrs["fr_" + k] = np.asarray(store[k][0], dtype=np.int32)
        arrs["val_" + k] = np.asarray(store[k][1], dtype=np.int16)
    arrs["meta"] = np.asarray([json.dumps(dict(source=source, video=video, fps=fps,
                                               nframes_declared=nframes, nframes_read=n_real,
                                               rate=SOURCES[source]["rate"],
                                               elapsed_sec=round(time.time() - t0, 1)),
                                          ensure_ascii=False)])
    np.savez_compressed(out + ".part.npz", **arrs)
    os.replace(out + ".part.npz", out)
    return source, video, "ok frames=%d readable=%d" % (n_real, sum(len(store[k][0]) for k in COUNTERS)), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--videos", default="")
    ap.add_argument("--workers", type=int, default=3)
    A = ap.parse_args()
    ensure_out()
    LOG = os.path.join(OUT, "report", "dump_reads.log")
    srcs = [s for s in SOURCES if not A.only or s in A.only.split(",")]
    vids = VIDEOS if not A.videos else [VIDEOS[int(i)] for i in A.videos.split(",")]
    jobs = [(s, v) for s in srcs for v in vids]
    log("[dump] jobs=%d workers=%d" % (len(jobs), A.workers), LOG)
    with cf.ThreadPoolExecutor(max_workers=A.workers) as ex:
        for s, v, msg, dt in ex.map(one, jobs):
            log("[dump] %-8s %-26s %s （%.0f 秒）" % (s, v[:26], msg, dt), LOG)
    log("DONE dump_reads", LOG)
