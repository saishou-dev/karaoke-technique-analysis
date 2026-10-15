#!/usr/bin/env python3
"""pYIN検証: 曲ごとのpYIN音域(百分位) vs DAM CSVのsingingRange(MIDIノート)"""
import numpy as np
import csv
import os
import sys
import glob

BASE = sys.argv[1]
CSVPATH = sys.argv[2]

def to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)

dam = {}
with open(CSVPATH, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["scoringDateTime"].startswith("20260628"):
            key = r["scoringDateTime"][8:12]
            dam[key] = (r["contentsName"], r.get("singingRangeLowest"), r.get("singingRangeHighest"))

ok = 0
n = 0
for p in sorted(glob.glob(os.path.join(BASE, "dataset", "pitch_pyin", "*_f0.csv"))):
    video = os.path.basename(p)[:-7]
    hhmm = video.split("_")[-1][:4]
    F = []
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["voiced"] == "1" and float(r["f0_hz"]) > 0:
                F.append(float(r["f0_hz"]))
    F = np.array(F)
    lo, hi = to_midi(np.percentile(F, 2)), to_midi(np.percentile(F, 98))
    # 撮影時刻とscoringDateTimeの近傍マッチ
    best = None
    for key, (name, sl, sh) in dam.items():
        diff = abs(int(key[:2]) * 60 + int(key[2:]) - (int(hhmm[:2]) * 60 + int(hhmm[2:])))
        if best is None or diff < best[0]:
            best = (diff, name, sl, sh)
    _, name, sl, sh = best
    if sl and sh:
        sl, sh = int(sl), int(sh)
        e_lo, e_hi = lo - sl, hi - sh
        good = abs(e_lo) <= 2 and abs(e_hi) <= 2
        ok += good
        n += 1
        print(f"{video[:14]}({name[:10]}): pYIN[{lo:.1f}-{hi:.1f}] DAM[{sl}-{sh}] 誤差[{e_lo:+.1f},{e_hi:+.1f}] {'OK' if good else 'NG'}")
print(f"±2半音以内: {ok}/{n}")
