#!/usr/bin/env python3
"""動画⇔録音 対応付け v3: グローバル割当
- 各(動画,テイク)の上位Kピークを候補に、スコア降順の貪欲割当
- 制約: 動画は1箇所のみ / 同一テイク内で区間が重複しない
- 検証: 割当区間の非無音率(クリップ上に乗っているか)
"""
import numpy as np
import sys
import os
import glob
import csv

SW = sys.argv[1]
HOP = 80
K = 8

def envelope(path):
    x = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(x) // HOP
    e = np.sqrt(np.mean(x[:n*HOP].reshape(n, HOP) ** 2, axis=1))
    return e

def prep(e):
    l = np.log1p(e * 100)
    return l - l.mean()

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
        lo, hi = max(0, i - 500), min(len(rr), i + 500)
        rr[lo:hi] = -1
    return peaks

vids = {}
for p in sorted(glob.glob(os.path.join(SW, "vid_*.s16"))):
    name = os.path.basename(p)[4:-4]
    e = envelope(p)
    vids[name] = (prep(e), len(e) / 100.0)
takes = {}
for p in sorted(glob.glob(os.path.join(SW, "wav_*.s16"))):
    name = os.path.basename(p)[4:-4]
    e = envelope(p)
    takes[name] = (prep(e), e, len(e) / 100.0)  # (正規化, 生エンベロープ, 長さ)

cands = []
for vn, (ve, vdur) in vids.items():
    for tn, (te, raw, tdur) in takes.items():
        for off, sc in topk_xcorr(ve, te, K):
            if off + vdur > tdur + 30:
                continue
            cands.append((sc, vn, tn, off))
cands.sort(reverse=True)

assigned = {}
used = {tn: [] for tn in takes}  # [(start,end)]
for sc, vn, tn, off in cands:
    if vn in assigned:
        continue
    vdur = vids[vn][1]
    conflict = any(not (off + vdur <= s + 5 or off >= e - 5) for s, e in used[tn])
    if conflict:
        continue
    assigned[vn] = (tn, off, sc)
    used[tn].append((off, off + vdur))

rows = []
for vn in sorted(vids, key=lambda v: v):
    if vn in assigned:
        tn, off, sc = assigned[vn]
        raw = takes[tn][1]
        seg = raw[int(off*100):int((off+vids[vn][1])*100)]
        active = float(np.mean(seg > 0.005)) if len(seg) else 0.0
        rows.append((vn, tn, off, sc, active))
        print(f"{vn} -> {tn} @ {off:.2f}s score={sc:.3f} 非無音率={active:.0%}")
    else:
        rows.append((vn, "UNMATCHED", "", "", ""))
        print(f"{vn} -> UNMATCHED")

with open(os.path.join(SW, "sync_map.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["video", "take", "offset_sec", "score", "active_ratio"])
    for vn, tn, off, sc, ar in rows:
        w.writerow([vn, tn,
                    off if off == "" else f"{off:.3f}",
                    sc if sc == "" else f"{sc:.3f}",
                    ar if ar == "" else f"{ar:.3f}"])
print("done")
