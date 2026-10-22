# -*- coding: utf-8 -*-
"""偶然あたりの水準（対照）を出す（手順3の裏取り）。

候補が密なので「±0.5 秒以内に候補があった割合」はそのままでは偶然でも高く出る。
そこで**カウンター増分の時刻を曲の中で一定量ずらした偽のイベント列**を作り、同じ手続きで
一致率を測る。本物の一致率と対照の差が、音程バーの発光が持っている実際の情報量になる。

ずらし量＝±5, ±7, ±9, ±11, ±13 秒（曲の長さで巡回）。対照はその平均。
出力: <OUT_DIR>/control.csv ・ lag_hist.csv（遅れの分布：本物と対照）
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P

SHIFTS = [-13, -11, -9, -7, -5, 5, 7, 9, 11, 13]

def rate(ct, Ts, tol):
    """各イベントについて ±tol 秒以内に候補があった割合"""
    if len(ct) == 0 or len(Ts) == 0: return 0.0
    idx = np.searchsorted(ct, Ts)
    ok = 0
    for T, i in zip(Ts, idx):
        d = min([abs(T - ct[j]) for j in (i - 1, i) if 0 <= j < len(ct)] or [9e9])
        if d <= tol: ok += 1
    return ok / len(Ts)

def nearest_lag(ct, Ts, win_lo=-3.0, win_hi=0.5):
    out = []
    for T in Ts:
        sel = ct[(ct >= T + win_lo) & (ct <= T + win_hi)]
        if len(sel): out.append(T - sel[np.argmin(np.abs(T - sel))])
    return np.array(out)

def main(videos, qs):
    g1 = pd.read_csv(P.G1_CSV)
    rows = []; hist_rows = []
    for q in qs:
        per = {}
        for v in videos:
            P.detect(v, q, quiet=True)
        for v in videos:
            cand = pd.read_csv(os.path.join(P.OUT_DIR, f"cand_{v}.csv"))
            ev = g1[g1.video == v]
            Ts = np.sort(ev.video_time_sec.astype(float).values)
            feat = pd.read_csv(os.path.join(P.OUT_DIR, f"feat_{v}.csv"))
            dur = float(feat.t.max())
            for meth in P.METHODS:
                ct = np.sort(cand[cand.method == meth].t.values)
                key = (q, meth)
                d = per.setdefault(key, dict(n_ev=0, n_cand=0, r05=0., r10=0., c05=[], c10=[],
                                             lag_real=[], lag_ctrl=[]))
                d["n_ev"] += len(Ts); d["n_cand"] += len(ct)
                d["r05"] += rate(ct, Ts, 0.5) * len(Ts)
                d["r10"] += rate(ct, Ts, 1.0) * len(Ts)
                d["lag_real"].append(nearest_lag(ct, Ts))
                for sh in SHIFTS:
                    Ts2 = np.sort((Ts + sh) % dur)
                    d["c05"].append(rate(ct, Ts2, 0.5) * len(Ts2))
                    d["c10"].append(rate(ct, Ts2, 1.0) * len(Ts2))
                    d["lag_ctrl"].append(nearest_lag(ct, Ts2))
        for (q_, meth), d in per.items():
            n = d["n_ev"]; ns = len(SHIFTS)
            r05 = d["r05"] / n; r10 = d["r10"] / n
            c05 = sum(d["c05"]) / (n * ns); c10 = sum(d["c10"]) / (n * ns)
            rows.append(dict(q=q_, method=meth, n_events=n, n_cand=d["n_cand"],
                             cand_per_sec=round(d["n_cand"] / 812.3, 3),
                             r05=round(r05, 3), ctrl05=round(c05, 3), gain05=round(r05 - c05, 3),
                             r10=round(r10, 3), ctrl10=round(c10, 3), gain10=round(r10 - c10, 3)))
            if abs(q_ - 0.95) < 1e-9:
                lr = np.concatenate(d["lag_real"]); lc = np.concatenate(d["lag_ctrl"])
                bins = np.arange(-0.5, 3.01, 0.25)
                hr, _ = np.histogram(lr, bins); hc, _ = np.histogram(lc, bins)
                for b0, a, bq in zip(bins[:-1], hr, hc):
                    hist_rows.append(dict(method=meth, bin_lo=round(b0, 2), bin_hi=round(b0 + .25, 2),
                                          real=int(a), real_frac=round(a / max(1, len(lr)), 4),
                                          ctrl=int(bq), ctrl_frac=round(bq / max(1, len(lc)), 4)))
    df = pd.DataFrame(rows).sort_values(["method", "q"])
    df.to_csv(os.path.join(P.OUT_DIR, "control.csv"), index=False, encoding="utf-8-sig")
    hd = pd.DataFrame(hist_rows)
    hd.to_csv(os.path.join(P.OUT_DIR, "lag_hist.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string())
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", nargs="*", type=float, default=[0.95, 0.98, 0.99, 0.995])
    ap.add_argument("--videos", nargs="*", default=None)
    a = ap.parse_args()
    main(a.videos or P.VIDEOS, a.qs)
