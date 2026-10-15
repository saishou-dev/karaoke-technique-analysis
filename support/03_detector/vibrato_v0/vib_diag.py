#!/usr/bin/env python3
"""DAMビブラートイベント近傍のf0を診断: 各ゲートがどこで落ちるか"""
import numpy as np
import csv
import os
import sys

BASE = sys.argv[1]
VIDEO = "明日への手紙_20260628_113919"
FR = 100

T, F, V = [], [], []
with open(os.path.join(BASE, "dataset", "pitch", VIDEO + "_f0.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        T.append(float(r["audio_time_sec"])); F.append(float(r["f0_hz"])); V.append(r["voiced"] == "1")
T = np.array(T); F = np.array(F); V = np.array(V)

evs = []
with open(os.path.join(BASE, "dataset", "G1_events_20260628.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["video"] == VIDEO and r["counter"] == "vibrato" and r["audio_time_sec"]:
            evs.append(float(r["audio_time_sec"]))

cents = np.where(V & (F > 0), 1200 * np.log2(np.maximum(F, 1) / 440.0), np.nan)
for t in evs[2:10]:
    i = int((t - T[0]) * FR)
    for label, sl in [("event-0.6..event", slice(max(0, i-60), i)), ("event..+0.6", slice(i, i+60))]:
        w = cents[sl]
        vr = np.mean(~np.isnan(w))
        wv = w[~np.isnan(w)]
        if len(wv) < 20:
            print(f"t={t:.2f} {label}: voiced={vr:.0%} (少なすぎ)")
            continue
        wd = wv - np.convolve(wv, np.ones(25)/25, "same")
        wd = wd[12:-12]
        depth = np.sqrt(2) * np.std(wd)
        ac = np.correlate(wd - wd.mean(), wd - wd.mean(), "full")[len(wd)-1:]
        ac = ac / (ac[0] + 1e-9)
        if len(ac) > 30:
            seg = ac[14:31]
            j = int(np.argmax(seg)); pk = seg[j]; lag = 14 + j
        else:
            pk, lag = 0, 0
        print(f"t={t:.2f} {label}: voiced={vr:.0%} depth={depth:.0f}c acf={pk:.2f}@{lag*10}ms f0中央={np.median(wv):.0f}c")
