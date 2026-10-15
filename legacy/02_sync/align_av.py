#!/usr/bin/env python3
"""動画⇔録音WAVの自動対応付け(クラップなし)
音量エンベロープ(100Hz)のFFT相互相関で (テイク, オフセット秒, スコア) を推定
"""
import numpy as np
import sys
import os
import glob
import csv

SW = sys.argv[1]  # sync_work dir
SR = 8000
HOP = 80  # 10ms -> 100Hz envelope

def envelope(path):
    x = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(x) // HOP
    e = np.sqrt(np.mean(x[:n*HOP].reshape(n, HOP) ** 2, axis=1))
    e = np.log1p(e * 100)
    e -= e.mean()
    return e

def xcorr_best(v, t):
    """vをtの中で探す。返り値: (offset_frames, score)"""
    n = len(v) + len(t)
    N = 1 << (n - 1).bit_length()
    V = np.fft.rfft(v, N)
    T = np.fft.rfft(t, N)
    c = np.fft.irfft(T * np.conj(V), N)
    valid = len(t) - len(v)
    if valid < 1:
        return None
    c = c[:valid]
    # 正規化: 各位置のtの局所ノルム
    t2 = np.concatenate([[0], np.cumsum(t * t)])
    win = t2[len(v):len(v)+valid] - t2[:valid]
    denom = np.sqrt(win) * np.linalg.norm(v) + 1e-9
    r = c / denom
    i = int(np.argmax(r))
    return i, float(r[i])

takes = {}
for p in sorted(glob.glob(os.path.join(SW, "wav_*.s16"))):
    takes[os.path.basename(p)[4:-4]] = envelope(p)

rows = []
for p in sorted(glob.glob(os.path.join(SW, "vid_*.s16"))):
    name = os.path.basename(p)[4:-4]
    v = envelope(p)
    best = None
    for tname, t in takes.items():
        r = xcorr_best(v, t)
        if r is None:
            continue
        off, score = r
        if best is None or score > best[2]:
            best = (tname, off / 100.0, score)
    if best:
        rows.append((name, best[0], best[1], best[2]))
        print(f"{name}: take={best[0]} offset={best[1]:.2f}s score={best[2]:.3f}")
    else:
        print(f"{name}: NO_MATCH")

with open(os.path.join(SW, "sync_map.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["video", "take", "offset_sec", "score"])
    for r in rows:
        w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}"])
print("done")
