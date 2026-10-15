#!/usr/bin/env python3
"""ビブラート検出 v0 (Phase 3 G3-1 プロトタイプ)
要旨設計: f0の自己相関 / 周期140-300ms / 窓約500ms / 深さと速さを算出
評価: DAMのビブラート加点イベント(G1)との突合(v0=区間ヒット率)
"""
import numpy as np
import csv
import os
import sys
import glob

BASE = sys.argv[1]
PITCH_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "dataset", "pitch")
FR = 100          # f0フレームレート(Hz)
WIN = 51          # 約500ms窓
LAG_MIN, LAG_MAX = 14, 30   # 140-300ms (3.3-7.1Hz)
ACF_T = 0.22
DEPTH_MIN, DEPTH_MAX = 15.0, 300.0  # cents
VOICED_MIN = 0.6
SEG_MIN = 0.25    # 秒

def moving_avg(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")

def clean_f0(f0, voiced):
    """メディアン5＋オクターブ跳び補正"""
    c = np.where(voiced & (f0 > 0), 1200 * np.log2(np.maximum(f0, 1) / 440.0), np.nan)
    # オクターブ補正: 直前の有声値から700c超跳んだら±1200で近い方へ
    prev = np.nan
    for i in range(len(c)):
        if np.isnan(c[i]):
            continue
        if not np.isnan(prev):
            while c[i] - prev > 700:
                c[i] -= 1200
            while c[i] - prev < -700:
                c[i] += 1200
        prev = 0.9 * (prev if not np.isnan(prev) else c[i]) + 0.1 * c[i]
    # メディアン5
    m = np.copy(c)
    for i in range(2, len(c) - 2):
        w = c[i-2:i+3]
        w = w[~np.isnan(w)]
        if len(w):
            m[i] = np.median(w)
    return m

def detect(times, f0, voiced):
    cents = clean_f0(f0, voiced)
    # 母線(ゆっくりした音程変化)を除去して揺れ成分だけに
    filled = np.copy(cents)
    idx = np.where(~np.isnan(filled))[0]
    if len(idx) < WIN:
        return []
    filled = np.interp(np.arange(len(filled)), idx, filled[idx])
    mod = filled - moving_avg(filled, 25)  # 250ms移動平均を引く
    n = len(mod)
    half = WIN // 2
    ok = np.zeros(n, bool)
    rate = np.zeros(n)
    depth = np.zeros(n)
    vmask = voiced.astype(float)
    for i in range(half, n - half):
        if np.mean(vmask[i-half:i+half+1]) < VOICED_MIN:
            continue
        w = mod[i-half:i+half+1]
        w = w - w.mean()
        d = np.sqrt(2) * np.std(w)
        if not (DEPTH_MIN <= d <= DEPTH_MAX):
            continue
        ac = np.correlate(w, w, "full")[len(w)-1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[LAG_MIN:LAG_MAX+1]
        j = int(np.argmax(seg))
        if seg[j] < ACF_T:
            continue
        ok[i] = True
        rate[i] = FR / (LAG_MIN + j)
        depth[i] = d
    # 区間化
    segs = []
    i = 0
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and (ok[j+1] or (j + 2 < n and ok[j+2])):
                j += 1
            if (j - i + 1) / FR >= SEG_MIN:
                m = ok[i:j+1]
                segs.append({
                    "start": times[i], "end": times[j],
                    "rate": float(np.mean(rate[i:j+1][m])),
                    "depth": float(np.mean(depth[i:j+1][m])),
                })
            i = j + 1
        else:
            i += 1
    return segs

# G1のビブラートイベント
vib_events = {}
with open(os.path.join(BASE, "dataset", "G1_events_20260628.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["counter"] == "vibrato" and r["audio_time_sec"]:
            vib_events.setdefault(r["video"], []).append(
                (float(r["audio_time_sec"]), int(r["delta"])))

out_rows = []
tot_ev = tot_hit = tot_seg = tot_seg_hit = 0
for p in sorted(glob.glob(os.path.join(PITCH_DIR, "*_f0.csv"))):
    video = os.path.basename(p)[:-7]
    T, F, V = [], [], []
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            T.append(float(r["audio_time_sec"]))
            F.append(float(r["f0_hz"]))
            V.append(r["voiced"] == "1" and float(r["f0_hz"]) > 0)
    T = np.array(T); F = np.array(F); V = np.array(V)
    segs = detect(T, F, V)
    evs = vib_events.get(video, [])
    hits = 0
    seg_used = set()
    for t, d in evs:
        for k, s in enumerate(segs):
            if s["start"] - 0.2 <= t <= s["end"] + 0.8:
                hits += 1
                seg_used.add(k)
                break
    tot_ev += len(evs); tot_hit += hits
    tot_seg += len(segs); tot_seg_hit += len(seg_used)
    rates = [s['rate'] for s in segs]
    depths = [s['depth'] for s in segs]
    print(f"{video}: segs={len(segs)} DAMev={len(evs)} hit={hits} "
          f"recall={hits/max(1,len(evs)):.0%} seghit={len(seg_used)}/{len(segs)} "
          f"rate~{np.mean(rates):.1f}Hz depth~{np.mean(depths):.0f}c" if segs else f"{video}: segs=0 DAMev={len(evs)}")
    for s in segs:
        out_rows.append({"video": video, "start": f"{s['start']:.2f}", "end": f"{s['end']:.2f}",
                         "rate_hz": f"{s['rate']:.2f}", "depth_cent": f"{s['depth']:.1f}"})

path = os.path.join(BASE, "dataset", "vibrato_segments_v0.csv")
with open(path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["video", "start", "end", "rate_hz", "depth_cent"])
    w.writeheader(); w.writerows(out_rows)
print(f"TOTAL: DAM events={tot_ev} hit={tot_hit} recall={tot_hit/max(1,tot_ev):.0%} / "
      f"segments={tot_seg} with-event={tot_seg_hit} ({tot_seg_hit/max(1,tot_seg):.0%})")
print("saved:", path)
