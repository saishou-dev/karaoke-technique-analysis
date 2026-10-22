# -*- coding: utf-8 -*-
"""4段目：技法加点表示のテンプレート照合（方式A）を、1〜3段目と同じ土俵で評価する。

標準名（材料40）＝正規化相互相関によるテンプレートマッチング。実装＝`04_marks/detect_marks.py`（無変更で呼ぶだけ）。

**公平に比べるための注意（これが本スクリプトの肝）**

方式A は曲の全長を見ていない。各加点 T のまわりの窓 [T−lmax, T+0.5] だけを復号する
（lmax＝ビブラート 3.0 秒・他 2.2 秒）。つまり**答えを使って探す場所を絞っている**。

一方、1〜3段目の発光系は答えを使わず曲の全長を走査した。

そのまま並べると方式A が不当に有利になる（時刻をずらした偽イベントは窓の外に落ちて必ず外れる）。

そこで 2 つそろえる。
  ① 対照を**窓の中のずらし**にする（±1.00／±1.25／±1.50 秒）。窓の外に出る偽イベントは使わない。
  ② 発光系の候補も**方式A と同じ窓の中だけ**に絞って、同じ対照で測り直す。

これで「同じ窓・同じ対照・同じ許容幅」の表になる。

出力:
  <OUT_DIR>/methodA_compare.csv   方式A と発光系を同じ窓・同じ対照で並べた表
  <OUT_DIR>/methodA_lag.csv       技法別の遅れ（方式A）
  <OUT_DIR>/methodA_match.csv     1 件ごとの対応
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P
import detect_grid as D
import peaks3 as K

MARKS_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
PARTIAL = os.path.join(MARKS_DIR, "partial_20261014")
G1_ALL = r"G:\研究アーカイブ\成果物\動画解析\dataset\G1_events_all.csv"
SESSION = 20261014
# detect_marks.FAMILY の最大先行 [秒]
LMAX = {"shakuri": 2.2, "accent": 2.2, "kobushi": 2.2, "vibrato": 3.0, "hammering": 2.2, "fall": 2.2}
IN_SHIFTS = [-1.50, -1.25, -1.00, 1.00, 1.25, 1.50]
JP = K.JP

def events(videos):
    """母数＝カウンターの増分1件＝1イベント。同一コマで複数カウンターが増えた場合も別々に数える＝171 件。"""
    g = pd.read_csv(G1_ALL)
    g = g[(g.session == SESSION) & (g.video.isin(videos))]
    return g.sort_values(["video", "video_time_sec"]).reset_index(drop=True)

def cand_times(video):
    """方式A が窓の中で見つけた候補（色ファミリは問わない＝発光系と同じ「印が出た時刻」の候補列）"""
    p = os.path.join(PARTIAL, video + "_cands.csv")
    if not os.path.exists(p): return np.array([])
    c = pd.read_csv(p)
    return np.sort(c["t"].astype(float).values)

def win_of(r):
    lmax = LMAX.get(r.counter, 2.2)
    return float(r.video_time_sec) - lmax, float(r.video_time_sec) + 0.5

def hit(ct, T, tol):
    if len(ct) == 0: return 0
    i = int(np.searchsorted(ct, T))
    for j in (i - 1, i):
        if 0 <= j < len(ct) and abs(ct[j] - T) <= tol: return 1
    return 0

def evaluate(videos, ct_of, label, ev=None, tols=(0.5, 1.0)):
    """窓の中だけで、実データと窓内ずらしの対照を測る"""
    ev = ev if ev is not None else events(videos)
    res = {t: dict(real=0, ctrl=0, nctrl=0) for t in tols}
    n = 0; n_cand_in_win = 0; win_sec = 0.0
    for v, g in ev.groupby("video"):
        ct = ct_of(v)
        for _, r in g.iterrows():
            lo, hi = win_of(r)
            T = float(r.video_time_sec)
            sub = ct[(ct >= lo) & (ct <= hi)]
            n += 1; n_cand_in_win += len(sub); win_sec += (hi - lo)
            for t in tols:
                res[t]["real"] += hit(sub, T, t)
                for sh in IN_SHIFTS:
                    T2 = T + sh
                    if T2 < lo + 0.25 or T2 > hi - 0.25: continue
                    res[t]["ctrl"] += hit(sub, T2, t); res[t]["nctrl"] += 1
    out = dict(手法=label, 件数=n, 窓内の候補数=n_cand_in_win,
               窓内の密度=round(n_cand_in_win / win_sec, 3))
    for t in tols:
        a = res[t]["real"] / n
        b = res[t]["ctrl"] / max(1, res[t]["nctrl"])
        out[f"当たり{int(t*10):02d}"] = res[t]["real"]
        out[f"r{int(t*10):02d}"] = round(a, 3)
        out[f"対照{int(t*10):02d}"] = round(b, 3)
        out[f"差{int(t*10):02d}"] = round(a - b, 3)
    return out

def lag_table(videos, ev=None):
    """方式A の抽出結果（partial/<video>.csv の mark_time_sec）から技法別の遅れを出す"""
    ev = ev if ev is not None else events(videos)
    parts = []
    for v in videos:
        p = os.path.join(PARTIAL, v + ".csv")
        if os.path.exists(p): parts.append(pd.read_csv(p))
    if not parts: return pd.DataFrame(), pd.DataFrame()
    m = pd.concat(parts, ignore_index=True)
    m["mark_time_sec"] = pd.to_numeric(m["mark_time_sec"], errors="coerce")
    m["lead_sec"] = pd.to_numeric(m["lead_sec"], errors="coerce")
    m["技法"] = m.counter.map(lambda c: JP.get(c, c))
    m.to_csv(os.path.join(P.OUT_DIR, "methodA_match.csv"), index=False, encoding="utf-8-sig")
    rows = []
    for c, g in m.groupby("counter"):
        lg = g.lead_sec.dropna()
        rows.append(dict(技法=JP.get(c, c), 処理した件数=len(g), 抽出=int(g.mark_time_sec.notna().sum()),
                         抽出率=round(float(g.mark_time_sec.notna().mean()), 3),
                         中央値=round(float(lg.median()), 3) if len(lg) else np.nan,
                         第3四分位=round(float(lg.quantile(.75)), 3) if len(lg) else np.nan,
                         p90=round(float(lg.quantile(.9)), 3) if len(lg) else np.nan,
                         最大=round(float(lg.max()), 3) if len(lg) else np.nan,
                         超2秒の割合=round(float((lg > 2.0).mean()), 3) if len(lg) else np.nan))
    ag = pd.DataFrame(rows).sort_values("p90", ascending=False)
    ag.to_csv(os.path.join(P.OUT_DIR, "methodA_lag.csv"), index=False, encoding="utf-8-sig")
    return m, ag

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--cfg", default="1/4縮小 96列 背景差分")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--gap", type=float, default=0.30)
    a = ap.parse_args()
    vids = a.videos or P.VIDEOS
    ev = events(vids)
    print(f"母数＝{len(ev)} 件（カウンターの増分1件＝1イベント・同一コマの複数増分も別々に数える）")
    rows = []
    rows.append(evaluate(vids, cand_times, "方式A 技法加点表示のテンプレート照合", ev))
    # 発光系を同じ窓・同じ対照で測り直す
    cache = {}
    def flash_of(v):
        if v not in cache:
            cache[v] = np.sort(K.candidates(v, a.cfg, a.k, a.gap))
        return cache[v]
    rows.append(evaluate(vids, flash_of, f"発光系 3段目（{a.cfg}・MAD×{a.k}・間隔{a.gap}s）", ev))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P.OUT_DIR, "methodA_compare.csv"), index=False, encoding="utf-8-sig")
    print()
    print("=== 同じ窓・同じ対照（窓の中で ±1.00〜±1.50 秒ずらし）で比べる ===")
    print(df.to_string(index=False))
    m, ag = lag_table(vids, ev)
    if len(ag):
        print()
        print("=== 方式A の技法別の遅れ（表示先行時間） ===")
        print(ag.to_string(index=False))
