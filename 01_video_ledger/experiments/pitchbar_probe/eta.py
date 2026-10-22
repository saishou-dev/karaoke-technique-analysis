# -*- coding: utf-8 -*-
"""イベント同期平均（ETA）＝ カウンター増分の時刻をそろえて特徴量を平均する。

候補の拾い方（閾値・併合幅）に依存しない、いちばん感度の高い確かめ方。
カウンターが増える前に音程バーが光るなら、増分時刻の手前に山が立つ。
対照＝増分時刻を ±5〜±13 秒ずらした偽のイベント列で同じ平均を取る。

出力: <OUT_DIR>/eta.csv（列＝技法・特徴量・相対時刻・本物の平均・対照の平均・差÷対照の標準偏差）
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P
from detect_local import scores, KINDS
from control import SHIFTS as FAR_SHIFTS
NEAR_SHIFTS = [-3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0]   # 歌っている最中どうしで比べる近い対照
SHIFTS = FAR_SHIFTS

JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}
PRE, POST = 3.0, 1.0
FPS = P.FPS_NOMINAL

def gather(videos, by_counter=False):
    g1 = pd.read_csv(P.G1_CSV)
    n_pre, n_post = int(PRE * FPS), int(POST * FPS)
    acc = {}      # key -> list of windows
    for v in videos:
        t, sc = scores(v)
        # 1 巡目の領域全体の特徴量も混ぜる
        f = P.add_all_diffs(pd.read_csv(os.path.join(P.OUT_DIR, f"feat_{v}.csv")))
        sc = dict(sc)
        sc["W_dV_cursor"] = f.dV.values
        sc["W_dgold_all"] = f.dgold_all.values
        dur = float(t.max())
        ev = g1[g1.video == v]
        for name, s in sc.items():
            s = np.asarray(s, float)
            nn = min(len(s), len(t))
            s = s[:nn]; tt = t[:nn]
            for grp, Ts in ([("全体", ev.video_time_sec.astype(float).values)] +
                            ([(JP.get(c, c), g.video_time_sec.astype(float).values)
                              for c, g in ev.groupby("counter")] if by_counter else [])):
                for kind, offs in (("real", [0.0]), ("ctrl", globals()["SHIFTS"])):
                    for off in offs:
                        for T in Ts:
                            T2 = (T + off) % dur if off else T
                            j = int(np.searchsorted(tt, T2))
                            a, b = j - n_pre, j + n_post
                            if a < 0 or b > nn: continue
                            acc.setdefault((name, grp, kind), []).append(s[a:b])
    return acc, n_pre, n_post

def main(videos):
    acc, n_pre, n_post = gather(videos, by_counter=True)
    rel = (np.arange(n_pre + n_post) - n_pre) / FPS
    rows = []
    for (name, grp, kind), lst in acc.items():
        if kind != "real": continue
        R = np.vstack(lst); C = np.vstack(acc[(name, grp, "ctrl")])
        mr = np.nanmean(R, axis=0); mc = np.nanmean(C, axis=0)
        sc_ = np.nanstd(C, axis=0) / np.sqrt(len(R))      # 本物の件数での標準誤差
        z = (mr - mc) / np.where(sc_ > 0, sc_, np.nan)
        for i in range(0, len(rel), 3):                    # 約 20ms → 50ms 間隔に間引く
            rows.append(dict(feature=name, group=grp, n_real=len(R), n_ctrl=len(C),
                             rel_sec=round(float(rel[i]), 3),
                             real=round(float(mr[i]), 5), ctrl=round(float(mc[i]), 5),
                             z=round(float(z[i]), 2)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P.OUT_DIR, "eta.csv"), index=False, encoding="utf-8-sig")
    # 要約＝特徴量×グループごとに |z| が最大の相対時刻
    s = (df[df.group == "全体"].loc[df[df.group == "全体"].groupby("feature").z.idxmax()]
         [["feature", "n_real", "rel_sec", "real", "ctrl", "z"]])
    print("=== 全体（|z| 最大の相対時刻。rel_sec が負＝カウンター増分より前） ===")
    print(s.to_string(index=False))
    print()
    g = df[df.group != "全体"]
    s2 = g.loc[g.groupby(["feature", "group"]).z.idxmax()][["feature", "group", "n_real", "rel_sec", "z"]]
    print("=== 技法ごと ===")
    print(s2.sort_values(["feature", "z"], ascending=[True, False]).to_string(index=False))
    return df

def curve(df, feature, group="全体"):
    return df[(df.feature == feature) & (df.group == group)].sort_values("rel_sec")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--near", action="store_true", help="対照を近いずらし（±1.5〜±3秒）にする")
    ap.add_argument("--curve", default=None, help="この特徴量の曲線を印字する")
    a = ap.parse_args()
    if a.near:
        import control as C
        SHIFTS = NEAR_SHIFTS
        C.SHIFTS = NEAR_SHIFTS
        globals()["SHIFTS"] = NEAR_SHIFTS
    df = main(a.videos or P.VIDEOS)
    if a.curve:
        print()
        print(f"=== {a.curve} 全体の曲線 ===")
        print(curve(df, a.curve).to_string(index=False))
