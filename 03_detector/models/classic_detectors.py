# -*- coding: utf-8 -*-
"""古典検出器（非ML）：ピッチ輪郭から しゃくり（出だしの上行グライド）と ビブラート（周期的な揺れ）を検出。
   しきい値は dev（9/11）で件数差最小に格子探索し、test（9/24）で件数整合と事象F1を測る。
   使い方: python classic_detectors.py"""
import sys, json, itertools, csv
import numpy as np
from scipy.ndimage import median_filter
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"; FR = 100
index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
def load(d):
    z = np.load(f"{ROOT}/feats/{d['video']}.npz"); st = z["st"].astype(float); v = z["voiced"] > 0
    st[~v] = np.nan; return st, v
def segments(st, v, min_len=8, jump=0.8):
    sm = st.copy(); ok = ~np.isnan(sm)
    idx = np.where(ok)[0]
    if len(idx) < 10: return [], sm
    sm[idx] = median_filter(sm[idx], size=5, mode="nearest")
    segs = []; i = 0; n = len(sm)
    while i < n:
        if not ok[i]: i += 1; continue
        j = i
        while j + 1 < n and ok[j + 1]:
            a_ = np.nanmedian(sm[max(i, j - 4):j + 1]); b_ = np.nanmedian(sm[j + 1:j + 6])
            if abs(b_ - a_) > jump and (j - i) >= min_len: break
            j += 1
        if (j - i + 1) >= min_len: segs.append((i, j))
        i = j + 1
    return segs, sm
def det_shakuri(sm, segs, rise_min, win, gap_min=0):
    """音符開始 win フレーム内の最低点から到達音高への上行が rise_min 半音以上"""
    out = []
    prev_end = -10**9
    for i, j in segs:
        n = j - i + 1; gap = i - prev_end; prev_end = j
        if n < win + 5 or gap < gap_min: continue
        head = sm[i:i + win]; tgt = np.nanmedian(sm[i + win:i + min(n, win + 25)])
        if tgt - np.nanmin(head) >= rise_min and np.nanargmin(head) < win * 0.7: out.append(i / FR)
    return out
def det_vibrato(sm, segs, depth_min, dur_min, r_min):
    """区間内の残差（31フレーム中央値を引く）の自己相関で 4〜8Hz の周期性が r_min 以上、深さ depth_min セント以上、長さ dur_min 秒以上"""
    out = []
    L = int(dur_min * FR)
    for i, j in segs:
        s = sm[i:j + 1]
        if len(s) < L: continue
        s = np.interp(np.arange(len(s)), np.where(~np.isnan(s))[0], s[~np.isnan(s)])
        d = s - median_filter(s, size=31, mode="nearest"); d -= d.mean(); den = np.dot(d, d)
        if den <= 1e-9: continue
        ac = np.correlate(d, d, "full")[len(d) - 1:] / den; lags = np.arange(12, 26)  # 4〜8.3Hz
        r = ac[lags].max(); depth = np.std(d) * 2 * np.sqrt(2) * 100
        if r >= r_min and depth >= depth_min: out.append((j + 1) / FR)  # 加点は揺れの終わりで数える
    return out
def match(pred, truth, tol):
    used = set(); hit = 0
    for p in sorted(pred):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - p) > tol: continue
            if best is None or abs(t - p) < abs(truth[best] - p): best = k
        if best is not None: used.add(best); hit += 1
    return hit
dev = [d for d in index if d["session"] == "20260911"]; test = [d for d in index if d["session"] == "20260924"]
cache = {d["video"]: segments(*load(d)) for d in dev + test}
def truth(d, c): lo, hi = d["span"]; return [t for cc, t in d["events"] if cc == c and lo <= t <= hi]
def run(det, ds, params):
    out = {}
    for d in ds:
        segs, sm = cache[d["video"]]; lo, hi = d["span"]; out[d["video"]] = [t for t in det(sm, segs, *params) if lo <= t <= hi]
    return out
def count_err(pred, ds, c): return sum(abs(len(pred[d["video"]]) - len(truth(d, c))) for d in ds), sum(len(truth(d, c)) for d in ds)
def report(name, c, pred, ds):
    e, t = count_err(pred, ds, c); lines = [f"### {name}: {c} 件数整合={1 - e / t:.3f}（Σ正解{t}・Σ|差|{e}）"]
    for tol in (1.0, 2.0):
        H = sum(match(pred[d["video"]], truth(d, c), tol) for d in ds); NP = sum(len(pred[d["video"]]) for d in ds)
        p = H / NP if NP else 0; r = H / t; lines.append(f"- ±{tol}s 再現率{r:.3f} 適合率{p:.3f} F1 {2*p*r/(p+r) if p+r else 0:.3f}（予測{NP}）")
    return lines
rep = ["# 古典検出器（devで件数校正→test）", ""]
grid_s = list(itertools.product([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], [10, 15, 20, 30], [0, 5, 10, 20]))
best = min(grid_s, key=lambda p: count_err(run(det_shakuri, dev, p), dev, "shakuri")[0]); rep.append(f"しゃくり 最良パラメタ（上行半音, 窓フレーム, 直前無声フレーム）={best}")
rep += report("dev", "shakuri", run(det_shakuri, dev, best), dev) + report("test", "shakuri", run(det_shakuri, test, best), test) + [""]
grid_v = list(itertools.product([20, 35, 50, 70], [0.3, 0.4, 0.5, 0.7], [0.3, 0.45, 0.6]))
bestv = min(grid_v, key=lambda p: count_err(run(det_vibrato, dev, p), dev, "vibrato")[0]); rep.append(f"ビブラート 最良パラメタ（深さcent, 長さs, 自己相関）={bestv}")
rep += report("dev", "vibrato", run(det_vibrato, dev, bestv), dev) + report("test", "vibrato", run(det_vibrato, test, bestv), test)
txt = "\n".join(rep); print(txt); open(f"{ROOT}/out/classic_report.md", "w", encoding="utf-8").write(txt)
ps = run(det_shakuri, test, best); pv = run(det_vibrato, test, bestv)
with open(f"{ROOT}/out/classic_counts_test.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["video", "type", "pred"])
    for d in test: w.writerow([d["video"], "shakuri", len(ps[d["video"]])]); w.writerow([d["video"], "vibrato", len(pv[d["video"]])])
