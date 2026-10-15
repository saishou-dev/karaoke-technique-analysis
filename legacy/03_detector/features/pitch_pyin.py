#!/usr/bin/env python3
"""pYIN(librosa)による高品質f0抽出 -> dataset/pitch_pyin/<video>_f0.csv (10msホップ・録音時間軸)"""
import numpy as np
import librosa
import csv
import os
import sys
import glob

BASE = sys.argv[1]
SONGS = sys.argv[2]
SR = 22050
HOP = 220  # ~10ms

sync = {}
with open(os.path.join(BASE, "sync_work", "sync_map.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        sync[r["video"]] = r

outdir = os.path.join(BASE, "dataset", "pitch_pyin")
os.makedirs(outdir, exist_ok=True)
for p in sorted(glob.glob(os.path.join(SONGS, "*.wav"))):
    video = os.path.splitext(os.path.basename(p))[0]
    off = float(sync[video]["offset_sec"])
    y, _ = librosa.load(p, sr=SR, mono=True)
    f0, vflag, vprob = librosa.pyin(y, fmin=70, fmax=800, sr=SR,
                                    frame_length=2048, hop_length=HOP)
    times = librosa.times_like(f0, sr=SR, hop_length=HOP)
    path = os.path.join(outdir, f"{video}_f0.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["audio_time_sec", "f0_hz", "conf", "voiced"])
        for t, fz, vf, vp in zip(times, f0, vflag, vprob):
            w.writerow([f"{off + t:.3f}",
                        "0.00" if np.isnan(fz) else f"{fz:.2f}",
                        f"{vp:.3f}", int(bool(vf))])
    vr = float(np.mean(vflag))
    med = float(np.nanmedian(f0))
    print(f"{video}: voiced={vr:.0%} median_f0={med:.0f}Hz", flush=True)
print("done")
