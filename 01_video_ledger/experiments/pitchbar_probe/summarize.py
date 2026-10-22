# -*- coding: utf-8 -*-
"""音程バーの発光候補とカウンター増分時刻の突合まとめ（手順3）。
出力: <OUT_DIR>/summary.md ・ lag_by_counter.csv ・ method_compare.csv
指標:
  一致率（±0.5s・±1s）＝ 生の時刻差（補正なし）
  補正後一致率        ＝ 曲ごとの遅れの中央値を引いてから ±0.5s・±1s
  候補の余り          ＝ どのイベントにも割り当たらなかった候補の数（＝カウンターにない発光）
  取りこぼし          ＝ 窓内に候補が 1 つも無かったイベントの数（＝カウンターにあるのに発光が無い）
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P

JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}

def main(videos):
    m = pd.read_csv(os.path.join(P.OUT_DIR, "match.csv"))
    s = pd.read_csv(os.path.join(P.OUT_DIR, "match_summary.csv"))
    rows = []
    for (v, meth), g in m.groupby(["video", "method"]):
        n = len(g)
        lag = g.lag
        has = lag.notna()
        med = float(lag[has].median()) if has.any() else np.nan
        adj = lag - med
        srow = s[(s.video == v) & (s.method == meth)].iloc[0]
        rows.append(dict(
            video=v, method=meth, n_events=n, n_cand=int(srow.n_cand),
            matched=int(has.sum()),
            miss=int((~has).sum()),
            extra=int(srow.n_cand_unused),
            r05=round(float((lag.abs() <= 0.5).sum()) / n, 3),
            r10=round(float((lag.abs() <= 1.0).sum()) / n, 3),
            lag_med=round(med, 3) if has.any() else np.nan,
            adj05=round(float((adj.abs() <= 0.5).sum()) / n, 3),
            adj10=round(float((adj.abs() <= 1.0).sum()) / n, 3),
        ))
    mc = pd.DataFrame(rows).sort_values(["video", "method"])
    mc.to_csv(os.path.join(P.OUT_DIR, "method_compare.csv"), index=False, encoding="utf-8-sig")

    # 最良の方式＝補正後 ±0.5s 一致率の 3 曲合計が最大
    best = mc.groupby("method").adj05.mean().sort_values(ascending=False)
    bm = best.index[0]
    b = m[m.method == bm]
    lag_rows = []
    for (c,), g in b.groupby(["counter"]):
        lag = g.lag.dropna()
        lag_rows.append(dict(counter=c, 技法=JP.get(c, c), n=len(g), matched=len(lag),
                             中央値=round(float(lag.median()), 3) if len(lag) else np.nan,
                             第1四分位=round(float(lag.quantile(.25)), 3) if len(lag) else np.nan,
                             第3四分位=round(float(lag.quantile(.75)), 3) if len(lag) else np.nan,
                             最大=round(float(lag.max()), 3) if len(lag) else np.nan,
                             遅れ1秒超=int((lag > 1.0).sum())))
    lg = pd.DataFrame(lag_rows).sort_values("中央値", ascending=False)
    lg.to_csv(os.path.join(P.OUT_DIR, "lag_by_counter.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(P.OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# 音程バーの発光と加点時刻 突合まとめ 2026-10-22\n\n")
        f.write("## 方式ごとの比較\n\n")
        f.write(mc.to_markdown(index=False) + "\n\n")
        f.write(f"最良の方式＝{bm}\n\n")
        f.write("## 技法ごとの遅れ（カウンター増分時刻 − 発光時刻［秒］・最良の方式）\n\n")
        f.write(lg.to_markdown(index=False) + "\n")
    print(mc.to_string())
    print()
    print(lg.to_string())
    print(f"\n最良の方式={bm}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", nargs="*", default=None)
    a = ap.parse_args()
    main(a.videos or P.VIDEOS)
