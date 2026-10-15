#!/usr/bin/env python3
"""動画⇔録音WAV対応付け v2: 相互相関の上位ピーク×撮影時刻の整合で解く
- 各(動画,テイク)ペアの上位Kピークから「テイク開始時刻の含意」を計算
- 同一テイクに属する動画群は同じ開始時刻を含意するはず → クラスタ投票
"""
import numpy as np
import sys
import os
import glob
import re
import csv

SW = sys.argv[1]
SR = 8000
HOP = 80
K = 8
TOL = 3.0  # 開始時刻クラスタ許容(秒)

def envelope(path):
    x = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(x) // HOP
    e = np.sqrt(np.mean(x[:n*HOP].reshape(n, HOP) ** 2, axis=1))
    e = np.log1p(e * 100)
    e -= e.mean()
    return e

def topk_xcorr(v, t, k):
    n = len(v) + len(t)
    N = 1 << (n - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(t, N) * np.conj(np.fft.rfft(v, N)), N)
    valid = len(t) - len(v)
    if valid < 1:
        return []
    c = c[:valid]
    t2 = np.concatenate([[0], np.cumsum(t * t)])
    win = t2[len(v):len(v)+valid] - t2[:valid]
    r = c / (np.sqrt(win) * np.linalg.norm(v) + 1e-9)
    peaks = []
    rr = r.copy()
    for _ in range(k):
        i = int(np.argmax(rr))
        if rr[i] <= 0:
            break
        peaks.append((i / 100.0, float(rr[i])))
        lo, hi = max(0, i - 500), min(len(rr), i + 500)  # 5秒近傍を潰す
        rr[lo:hi] = -1
    return peaks

def clock(name):
    m = re.search(r"_(\d{8})_(\d{6})$", name)
    hh, mm, ss = int(m.group(2)[:2]), int(m.group(2)[2:4]), int(m.group(2)[4:6])
    return hh * 3600 + mm * 60 + ss

vids = {}
for p in sorted(glob.glob(os.path.join(SW, "vid_*.s16"))):
    name = os.path.basename(p)[4:-4]
    e = envelope(p)
    vids[name] = (e, clock(name), len(e) / 100.0)
takes = {}
for p in sorted(glob.glob(os.path.join(SW, "wav_*.s16"))):
    name = os.path.basename(p)[4:-4]
    e = envelope(p)
    takes[name] = (e, len(e) / 100.0)

# 全ペアの上位ピーク
cands = []  # (video, take, offset, score, implied_start_clock)
for vn, (ve, vclk, vdur) in vids.items():
    for tn, (te, tdur) in takes.items():
        for off, sc in topk_xcorr(ve, te, K):
            if off + vdur > tdur + 60:
                continue
            cands.append((vn, tn, off, sc, vclk - off))

# テイクごとに開始時刻クラスタを作り、スコア合計で評価
clusters = []  # (total_score, take, start, [(video, offset, score)])
for tn in takes:
    tc = [c for c in cands if c[1] == tn]
    for _, _, off0, sc0, st0 in tc:
        members = {}
        for vn, _, off, sc, st in tc:
            if abs(st - st0) <= TOL:
                if vn not in members or sc > members[vn][1]:
                    members[vn] = (off, sc)
        # 同一テイク内で動画同士が重ならないか検査(重なったらスコア低い方を落とす)
        items = sorted(((off, sc, vn) for vn, (off, sc) in members.items()))
        pruned = []
        for off, sc, vn in items:
            if pruned and off < pruned[-1][0] + vids[pruned[-1][2]][2] - 5:
                if sc > pruned[-1][1]:
                    pruned[-1] = (off, sc, vn)
                continue
            pruned.append((off, sc, vn))
        total = sum(sc for _, sc, _ in pruned)
        clusters.append((total, tn, st0, [(vn, off, sc) for off, sc, vn in pruned]))

clusters.sort(key=lambda c: -c[0])
assigned = {}
used_takes = set()
for total, tn, st, members in clusters:
    if tn in used_takes:
        continue
    new = [m for m in members if m[0] not in assigned]
    if not new:
        continue
    used_takes.add(tn)
    for vn, off, sc in new:
        assigned[vn] = (tn, off, sc, st)

def hms(s):
    s = int(s)
    return f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"

rows = []
for vn in sorted(vids, key=lambda v: vids[v][1]):
    if vn in assigned:
        tn, off, sc, st = assigned[vn]
        rows.append((vn, tn, off, sc))
        print(f"{vn}  clock={hms(vids[vn][1])}  -> {tn} @ {off:.2f}s (score {sc:.3f}, take開始={hms(st)})")
    else:
        rows.append((vn, "UNMATCHED", "", ""))
        print(f"{vn}  clock={hms(vids[vn][1])}  -> UNMATCHED")

with open(os.path.join(SW, "sync_map.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["video", "take", "offset_sec", "score"])
    for vn, tn, off, sc in rows:
        w.writerow([vn, tn, off if off == "" else f"{off:.3f}", sc if sc == "" else f"{sc:.3f}"])
print("done")
