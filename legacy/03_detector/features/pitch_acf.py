#!/usr/bin/env python3
"""自己相関(ACF)ピッチトラッカー v1 (D9: 自作=ビブラート検出の核と同系)
sync_mapで対応済みの各曲区間から f0 曲線を抽出
出力: dataset/pitch/<video>_f0.csv (audio_time_sec, f0_hz, conf)
"""
import numpy as np
import csv
import os
import sys
import glob

BASE = sys.argv[1]
SW = os.path.join(BASE, "sync_work")
SR = 8000
WIN = 320   # 40ms
HOP = 80    # 10ms
F0_MIN, F0_MAX = 70.0, 800.0
LAG_MIN = int(SR / F0_MAX)
LAG_MAX = int(SR / F0_MIN)
CONF_T = 0.5

def load(path):
    return np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0

def f0_track(x):
    n = (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + (np.arange(n) * HOP)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    # FFTで一括ACF
    N = 1 << int(np.ceil(np.log2(WIN * 2)))
    F = np.fft.rfft(frames, N, axis=1)
    acf = np.fft.irfft(F * np.conj(F), N, axis=1)[:, :LAG_MAX + 2]
    e0 = acf[:, 0] + 1e-9
    r = acf / e0[:, None]
    seg = r[:, LAG_MIN:LAG_MAX + 1]
    lag_rel = np.argmax(seg, axis=1)
    lag = lag_rel + LAG_MIN
    conf = seg[np.arange(n), lag_rel]
    # 放物線補間
    l0 = np.clip(lag, LAG_MIN + 1, LAG_MAX - 1)
    ym = r[np.arange(n), l0 - 1]
    y0 = r[np.arange(n), l0]
    yp = r[np.arange(n), l0 + 1]
    denom = (ym - 2 * y0 + yp)
    delta = np.where(np.abs(denom) > 1e-9, 0.5 * (ym - yp) / denom, 0.0)
    lag_f = l0 + np.clip(delta, -1, 1)
    f0 = SR / lag_f
    voiced = (conf > CONF_T) & (e0 > 1e-5)
    return f0, conf, voiced

sync = {}
with open(os.path.join(SW, "sync_map.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        sync[row["video"]] = row

outdir = os.path.join(BASE, "dataset", "pitch")
os.makedirs(outdir, exist_ok=True)
takes = {}
for p in glob.glob(os.path.join(SW, "wav_*.s16")):
    takes[os.path.basename(p)[4:-4]] = load(p)

for video, row in sync.items():
    if row["take"] in ("", "UNMATCHED"):
        print(f"{video}: skip (no audio)")
        continue
    off = float(row["offset_sec"])
    vp = os.path.join(SW, f"vid_{video}.s16")
    vdur = os.path.getsize(vp) / 2 / SR
    t = takes[row["take"]]
    seg = t[int(off * SR):int((off + vdur) * SR)]
    f0, conf, voiced = f0_track(seg)
    path = os.path.join(outdir, f"{video}_f0.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["audio_time_sec", "f0_hz", "conf", "voiced"])
        for i in range(len(f0)):
            at = off + (i * HOP + WIN / 2) / SR
            w.writerow([f"{at:.3f}", f"{f0[i]:.2f}", f"{conf[i]:.3f}", int(voiced[i])])
    vr = float(np.mean(voiced))
    med = float(np.median(f0[voiced])) if voiced.any() else 0
    print(f"{video}: frames={len(f0)} voiced={vr:.0%} median_f0={med:.0f}Hz -> {os.path.basename(path)}")
print("done")
