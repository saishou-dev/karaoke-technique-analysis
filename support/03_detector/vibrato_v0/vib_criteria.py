#!/usr/bin/env python3
"""DAM加点あり/なし区間の特徴分布比較 (加点条件の手がかり)"""
import csv
import os
import sys
import numpy as np

BASE = sys.argv[1]
segs = []
with open(os.path.join(BASE, "dataset", "vibrato_segments_v0.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        segs.append({"video": r["video"], "s": float(r["start"]), "e": float(r["end"]),
                     "rate": float(r["rate_hz"]), "depth": float(r["depth_cent"])})
evs = {}
with open(os.path.join(BASE, "dataset", "G1_events_20260628.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["counter"] == "vibrato" and r["audio_time_sec"]:
            evs.setdefault(r["video"], []).append(float(r["audio_time_sec"]))

hit, miss = [], []
for g in segs:
    ok = any(g["s"] - 0.2 <= t <= g["e"] + 0.8 for t in evs.get(g["video"], []))
    (hit if ok else miss).append(g)

def stats(group, key):
    v = np.array([g[key] if key != "dur" else g["e"] - g["s"] for g in group])
    return f"中央{np.median(v):.2f} / 25-75%[{np.percentile(v,25):.2f}-{np.percentile(v,75):.2f}]"

print(f"加点あり区間 n={len(hit)} / 加点なし区間 n={len(miss)}")
for key, label in [("dur", "長さ(秒)"), ("rate", "速さ(Hz)"), ("depth", "深さ(cent)")]:
    print(f"{label}: 加点あり {stats(hit, key)} | 加点なし {stats(miss, key)}")
