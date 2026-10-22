# -*- coding: utf-8 -*-
"""閾値（上位 1−q の分位点）を振って、候補数と一致率の釣り合いを見る。
出力: <OUT_DIR>/sweep.csv（3 曲合計）
q を下げると候補が増え、取りこぼしは減るがカウンターにない候補が増える。
指標:
  再現率 r05/r10 ＝ カウンター増分のうち ±0.5s／±1s に候補があった割合（補正なしの生の時刻差）
  補正後 adj05/adj10 ＝ 曲をまたいだ遅れの中央値を引いてからの一致率
  適合率 prec     ＝ 候補のうちイベントに割り当たった数 ÷ 候補数（カウンターにない候補の裏返し）
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P

def run_q(videos, q, win_lo=-3.0, win_hi=0.5):
    for v in videos:
        P.detect(v, q, quiet=True)
    m, s = P.match(videos, win_lo, win_hi)
    out = []
    for meth, g in m.groupby("method"):
        n = len(g)
        lag = g.lag
        med = float(lag.median())
        adj = lag - med
        extra = int(s[s.method == meth].n_cand_unused.sum())
        ncand = int(s[s.method == meth].n_cand.sum())
        out.append(dict(q=q, method=meth, n_events=n, n_cand=ncand,
                        matched=int(lag.notna().sum()),
                        miss=int(lag.isna().sum()), extra=extra,
                        prec=round((ncand - extra) / max(1, ncand), 3),
                        r05=round(float((lag.abs() <= .5).sum()) / n, 3),
                        r10=round(float((lag.abs() <= 1.).sum()) / n, 3),
                        lag_med=round(med, 3),
                        adj05=round(float((adj.abs() <= .5).sum()) / n, 3),
                        adj10=round(float((adj.abs() <= 1.).sum()) / n, 3)))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", nargs="*", type=float, default=[0.95, 0.98, 0.99, 0.995, 0.998, 0.999])
    ap.add_argument("--videos", nargs="*", default=None)
    a = ap.parse_args()
    vids = a.videos or P.VIDEOS
    rows = []
    for q in a.qs:
        rows += run_q(vids, q)
        print(f"q={q} 済", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P.OUT_DIR, "sweep.csv"), index=False, encoding="utf-8-sig")
    print(df.sort_values(["method", "q"]).to_string())
