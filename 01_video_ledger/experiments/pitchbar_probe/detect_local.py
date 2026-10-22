# -*- coding: utf-8 -*-
"""局所化した発光検出（方式 L）と、偶然あたりの水準との比較。

1 巡目（領域全体の平均・面積）では背景映像の明暗に埋もれて偶然を超えなかった。
2 巡目は領域を横 96 本の短冊に割った断面（prof_<video>.npz）を使う。

  d(c, t) = p(c, t) − p(c, t−2)                     短冊ごとの 2 フレーム差
  s(t)    = max_c [ d(c, t) − median_c d(c, t) ]    画面全体の明暗の変化を引いた、いちばん立った短冊

  L1_gold   ＝ 金色画素の短冊差の最大（演奏済みバーの金色・加点の火花）
  L2_white  ＝ 白い画素の短冊差の最大（閃光）
  L3_bright ＝ 明るい画素の短冊差の最大
  L4_mix    ＝ L1 と L2 をそれぞれの中央絶対偏差で正規化して足したもの

対照＝カウンター増分の時刻を曲の中で ±5〜±13 秒ずらした偽のイベント列。
出力: <OUT_DIR>/control_local.csv ・ cand_local_<video>.csv ・ lag_hist_local.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P
from control import SHIFTS, rate, nearest_lag

KINDS = {"L1_gold": 0, "L2_white": 1, "L3_bright": 2}

def scores(video):
    z = np.load(os.path.join(P.OUT_DIR, f"prof_{video}.npz"))
    prof, t = z["prof"], z["t"]          # (N, 3, ncol)
    out = {}
    for name, k in KINDS.items():
        p = prof[:, k, :]
        d = np.full_like(p, np.nan)
        d[2:] = p[2:] - p[:-2]
        s = np.nanmax(d - np.nanmedian(d, axis=1, keepdims=True), axis=1)
        out[name] = s
    def nz(x):
        m = np.nanmedian(np.abs(x - np.nanmedian(x))) or 1.0
        return (x - np.nanmedian(x)) / m
    out["L4_mix"] = nz(out["L1_gold"]) + nz(out["L2_white"])
    return t, out

def candidates(video, q):
    t, sc = scores(video)
    rows = []
    for name, s in sc.items():
        pk, thr = P.peaks(t, s, q=q)
        for tt, v in pk:
            rows.append(dict(video=video, method=name, t=round(tt, 3), peak=round(float(v), 6)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P.OUT_DIR, f"cand_local_{video}.csv"), index=False, encoding="utf-8-sig")
    return df

def main(videos, qs):
    g1 = pd.read_csv(P.G1_CSV)
    rows = []; hist = []
    for q in qs:
        per = {}
        for v in videos:
            cand = candidates(v, q)
            ev = g1[g1.video == v]
            Ts = np.sort(ev.video_time_sec.astype(float).values)
            z = np.load(os.path.join(P.OUT_DIR, f"prof_{v}.npz")); dur = float(z["t"].max())
            for meth in sorted(set(cand.method)):
                ct = np.sort(cand[cand.method == meth].t.values)
                d = per.setdefault(meth, dict(n_ev=0, n_cand=0, r05=0., r10=0., c05=[], c10=[],
                                              lr=[], lc=[], dur=0.))
                d["n_ev"] += len(Ts); d["n_cand"] += len(ct); d["dur"] += dur
                d["r05"] += rate(ct, Ts, 0.5) * len(Ts)
                d["r10"] += rate(ct, Ts, 1.0) * len(Ts)
                d["lr"].append(nearest_lag(ct, Ts))
                for sh in SHIFTS:
                    Ts2 = np.sort((Ts + sh) % dur)
                    d["c05"].append(rate(ct, Ts2, 0.5) * len(Ts2))
                    d["c10"].append(rate(ct, Ts2, 1.0) * len(Ts2))
                    d["lc"].append(nearest_lag(ct, Ts2))
        for meth, d in per.items():
            n = d["n_ev"]; ns = len(SHIFTS)
            r05 = d["r05"] / n; r10 = d["r10"] / n
            c05 = sum(d["c05"]) / (n * ns); c10 = sum(d["c10"]) / (n * ns)
            rows.append(dict(q=q, method=meth, n_events=n, n_cand=d["n_cand"],
                             cand_per_sec=round(d["n_cand"] / d["dur"], 3),
                             r05=round(r05, 3), ctrl05=round(c05, 3), gain05=round(r05 - c05, 3),
                             r10=round(r10, 3), ctrl10=round(c10, 3), gain10=round(r10 - c10, 3)))
            if abs(q - 0.99) < 1e-9:
                lr = np.concatenate(d["lr"]); lc = np.concatenate(d["lc"])
                bins = np.arange(-0.5, 3.01, 0.25)
                hr, _ = np.histogram(lr, bins); hc, _ = np.histogram(lc, bins)
                for b0, a, bq in zip(bins[:-1], hr, hc):
                    hist.append(dict(method=meth, bin_lo=round(b0, 2), bin_hi=round(b0 + .25, 2),
                                     real=int(a), real_frac=round(a / max(1, len(lr)), 4),
                                     ctrl=int(bq), ctrl_frac=round(bq / max(1, len(lc)), 4)))
    df = pd.DataFrame(rows).sort_values(["method", "q"])
    df.to_csv(os.path.join(P.OUT_DIR, "control_local.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(hist).to_csv(os.path.join(P.OUT_DIR, "lag_hist_local.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string())
    return df

JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}

def write_match(videos, q, method, win_lo=-3.0, win_hi=0.5):
    """選んだ方式の候補とカウンター増分を 1 対 1 で結び、技法ごとの遅れの分布を出す"""
    g1 = pd.read_csv(P.G1_CSV)
    rows = []
    for v in videos:
        cand = candidates(v, q)
        ct = np.sort(cand[cand.method == method].t.values)
        ev = g1[g1.video == v].sort_values("video_time_sec")
        used = set()
        for _, r in ev.iterrows():
            T = float(r.video_time_sec)
            sel = np.where((ct >= T + win_lo) & (ct <= T + win_hi))[0]
            free = [j for j in sel if j not in used]
            pool = free if free else list(sel)
            best = min(pool, key=lambda j: abs(T - ct[j])) if pool else None
            if best is not None: used.add(best)
            rows.append(dict(video=v, method=method, counter=r.counter, 技法=JP.get(r.counter, r.counter),
                             count_value=int(r.count_value), multi=int(r.multi_increment), T=round(T, 3),
                             flash_t=round(float(ct[best]), 3) if best is not None else np.nan,
                             lag=round(T - float(ct[best]), 3) if best is not None else np.nan,
                             n_in_win=len(sel)))
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(P.OUT_DIR, "match_local.csv"), index=False, encoding="utf-8-sig")
    agg = []
    for c, g in m.groupby("counter"):
        lag = g.lag.dropna()
        agg.append(dict(counter=c, 技法=JP.get(c, c), n=len(g), 対応=len(lag),
                        中央値=round(float(lag.median()), 3) if len(lag) else np.nan,
                        第1四分位=round(float(lag.quantile(.25)), 3) if len(lag) else np.nan,
                        第3四分位=round(float(lag.quantile(.75)), 3) if len(lag) else np.nan,
                        遅れ1秒超=int((lag > 1.0).sum()),
                        遅れ1秒超の割合=round(float((lag > 1.0).mean()), 3) if len(lag) else np.nan))
    ag = pd.DataFrame(agg).sort_values("中央値", ascending=False)
    ag.to_csv(os.path.join(P.OUT_DIR, "lag_by_counter_local.csv"), index=False, encoding="utf-8-sig")
    print(ag.to_string())
    return m, ag

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", nargs="*", type=float, default=[0.95, 0.98, 0.99, 0.995, 0.998])
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--write_match", nargs=2, metavar=("Q", "METHOD"), default=None)
    a = ap.parse_args()
    vids = a.videos or P.VIDEOS
    if a.write_match:
        write_match(vids, float(a.write_match[0]), a.write_match[1])
    else:
        main(vids, a.qs)
