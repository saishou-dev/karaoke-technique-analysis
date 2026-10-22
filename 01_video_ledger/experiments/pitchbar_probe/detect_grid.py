# -*- coding: utf-8 -*-
"""2段目：升目で閃光の局所ピークを検出し、1段目の短冊法と同じ手順で評価する。

スコアの作り方（升目 R×C・マスク k）:
  g(r, c, t)                       升目ごとの画素数
  d(r, c, t) = g(t) − g(t−2)       2 コマ差
  b(r, c, t) = d を直径 100 画素ぶんの箱で平滑化   ← 閃光の大きさに合わせて集める
  s(t)       = max_{r,c} [ b(r,c,t) − median_{r,c} b(r,c,t) ]
  中央値を引くのは、画面全体が明るくなる背景映像の変化を打ち消すため。

振る条件:
  升目の粗さ ＝ 192×16 ／ 96×8 ／ 48×4（保存した 192×16 を足し合わせて作る）
  平滑化     ＝ 閃光の大きさに合わせる（blob）／しない（raw）
  しきい値   ＝ 上位 5% ／ 1% ／ 0.5%

評価は 1段目と同じ:
  正解を使わない全長走査 → カウンター増分との突き合わせ → 時刻をずらした偽イベント列の対照
  近い対照（±1.5〜±3 秒）での時刻そろえ平均（z 値）

出力: <OUT_DIR>/grid_compare.csv ・ grid_eta.csv ・ grid_lag_by_counter.csv ・ match_grid.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P
from control import SHIFTS as FAR_SHIFTS, rate, nearest_lag

NEAR_SHIFTS = [-3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0]
KIND = {"white": 0, "gold": 1, "bright": 2}
BLOB_PX = 100.0          # 閃光の見かけの直径（原寸の画素）
GRIDS = [(16, 192), (8, 96), (4, 48)]
QS = [0.95, 0.99, 0.995]
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}

_cache = {}

def load(video):
    if video in _cache: return _cache[video]
    z = np.load(os.path.join(P.OUT_DIR, f"grid_{video}.npz"))
    g = z["grid"].astype(np.float32)      # (N, 3, 16, 192)
    t = z["t"]; crop = z["crop"]
    _cache[video] = (g, t, crop)
    return _cache[video]

def coarsen(g, nrow, ncol):
    """保存した 16×192 を足し合わせて粗い升目にする"""
    N, K, R, C = g.shape
    if (R, C) == (nrow, ncol): return g
    fr, fc = R // nrow, C // ncol
    return g.reshape(N, K, nrow, fr, ncol, fc).sum(axis=(3, 5))

def boxsum(x, k, axis):
    """幅 k の箱の合計（累積和で一気に計算する。畳み込みを 1 本ずつ回すと桁違いに遅い）"""
    if k <= 1: return x
    before, after = k // 2, k - 1 - k // 2
    pads = [(0, 0)] * x.ndim; pads[axis] = (before, after)
    c = np.cumsum(np.pad(x, pads, mode="constant"), axis=axis, dtype=np.float32)
    zs = list(c.shape); zs[axis] = 1
    c = np.concatenate([np.zeros(zs, np.float32), c], axis=axis)
    s1 = [slice(None)] * x.ndim; s1[axis] = slice(k, None)
    s0 = [slice(None)] * x.ndim; s0[axis] = slice(0, -k)
    return c[tuple(s1)] - c[tuple(s0)]

def blob_kernel(crop, nrow, ncol):
    """升目 1 つの原寸の大きさから、閃光の直径ぶんの箱の辺の数を決める"""
    kh = max(1, int(round(BLOB_PX / (crop[3] / nrow))))
    kw = max(1, int(round(BLOB_PX / (crop[2] / ncol))))
    return kh, kw

def score(video, kind, nrow, ncol, blob=True, chunk=4000):
    g, t, crop = load(video)
    a = coarsen(g, nrow, ncol)[:, KIND[kind]].astype(np.float32)   # (N, nrow, ncol)
    N = a.shape[0]
    kh, kw = blob_kernel(crop, nrow, ncol) if blob else (1, 1)
    s = np.full(N, np.nan, np.float32)
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        lo = max(0, i0 - 2)
        blk = a[lo:i1]
        d = blk[2:] - blk[:-2]            # 行 j は元のコマ lo+j+2 に対応
        if d.size == 0: continue
        if kh > 1: d = boxsum(d, kh, 1)
        if kw > 1: d = boxsum(d, kw, 2)
        f = d.reshape(d.shape[0], -1)
        v = f.max(axis=1) - np.median(f, axis=1)
        s[lo + 2:lo + 2 + len(v)] = v
    return t, s

def candidates(video, kind, nrow, ncol, blob, q):
    t, s = score(video, kind, nrow, ncol, blob)
    pk, thr = P.peaks(t, s, q=q)
    return np.array([p[0] for p in pk]), thr

def evaluate(videos, kinds=("white",), grids=GRIDS, blobs=(True, False), qs=QS,
             shifts=FAR_SHIFTS, win_lo=-3.0, win_hi=0.5):
    g1 = pd.read_csv(P.G1_CSV)
    rows = []
    for kind in kinds:
        for nrow, ncol in grids:
            for blob in blobs:
                for q in qs:
                    n_ev = n_cand = 0; r05 = r10 = 0.0
                    c05 = []; c10 = []; dur_sum = 0.0
                    for v in videos:
                        ct, thr = candidates(v, kind, nrow, ncol, blob, q)
                        _, t, _ = load(v); dur = float(t.max()); dur_sum += dur
                        Ts = np.sort(g1[g1.video == v].video_time_sec.astype(float).values)
                        n_ev += len(Ts); n_cand += len(ct)
                        r05 += rate(ct, Ts, 0.5) * len(Ts)
                        r10 += rate(ct, Ts, 1.0) * len(Ts)
                        for sh in shifts:
                            Ts2 = np.sort((Ts + sh) % dur)
                            c05.append(rate(ct, Ts2, 0.5) * len(Ts2))
                            c10.append(rate(ct, Ts2, 1.0) * len(Ts2))
                    ns = len(shifts)
                    a05 = r05 / n_ev; a10 = r10 / n_ev
                    b05 = sum(c05) / (n_ev * ns); b10 = sum(c10) / (n_ev * ns)
                    rows.append(dict(マスク=kind, 升目=f"{ncol}x{nrow}",
                                     平滑化=("閃光の大きさ" if blob else "なし"),
                                     しきい値=f"上位{(1-q)*100:.1f}%",
                                     候補数=n_cand, 密度=round(n_cand / dur_sum, 3),
                                     r05=round(a05, 3), 対照05=round(b05, 3), 差05=round(a05 - b05, 3),
                                     r10=round(a10, 3), 対照10=round(b10, 3), 差10=round(a10 - b10, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P.OUT_DIR, "grid_compare.csv"), index=False, encoding="utf-8-sig")
    return df

def eta(videos, kind, nrow, ncol, blob, shifts=NEAR_SHIFTS, pre=3.0, post=1.0, by_counter=True):
    """近い対照での時刻そろえ平均。閾値の取り方に依存しない確かめ方。"""
    g1 = pd.read_csv(P.G1_CSV)
    FPS = P.FPS_NOMINAL
    n_pre, n_post = int(pre * FPS), int(post * FPS)
    acc = {}
    for v in videos:
        t, s = score(v, kind, nrow, ncol, blob)
        nn = len(s); dur = float(t.max())
        ev = g1[g1.video == v]
        groups = [("全体", ev.video_time_sec.astype(float).values)]
        if by_counter:
            groups += [(JP.get(c, c), x.video_time_sec.astype(float).values) for c, x in ev.groupby("counter")]
        for grp, Ts in groups:
            for tag, offs in (("real", [0.0]), ("ctrl", shifts)):
                for off in offs:
                    for T in Ts:
                        T2 = (T + off) % dur if off else T
                        j = int(np.searchsorted(t, T2)); a, b = j - n_pre, j + n_post
                        if a < 0 or b > nn: continue
                        acc.setdefault((grp, tag), []).append(s[a:b])
    rel = (np.arange(n_pre + n_post) - n_pre) / FPS
    rows = []
    for (grp, tag), lst in acc.items():
        if tag != "real": continue
        R = np.vstack(lst); C = np.vstack(acc[(grp, "ctrl")])
        mr = np.nanmean(R, 0); mc = np.nanmean(C, 0)
        se = np.nanstd(C, 0) / np.sqrt(len(R))
        z = (mr - mc) / np.where(se > 0, se, np.nan)
        for i in range(0, len(rel), 3):
            rows.append(dict(group=grp, n_real=len(R), rel_sec=round(float(rel[i]), 3),
                             real=round(float(mr[i]), 4), ctrl=round(float(mc[i]), 4),
                             z=round(float(z[i]), 2)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P.OUT_DIR, "grid_eta.csv"), index=False, encoding="utf-8-sig")
    return df

def lag_table(videos, kind, nrow, ncol, blob, q, win_lo=-3.0, win_hi=0.5):
    g1 = pd.read_csv(P.G1_CSV)
    rows = []
    for v in videos:
        ct, thr = candidates(v, kind, nrow, ncol, blob, q)
        ct = np.sort(ct)
        ev = g1[g1.video == v].sort_values("video_time_sec")
        used = set()
        for _, r in ev.iterrows():
            T = float(r.video_time_sec)
            sel = np.where((ct >= T + win_lo) & (ct <= T + win_hi))[0]
            free = [j for j in sel if j not in used]
            pool = free if free else list(sel)
            best = min(pool, key=lambda j: abs(T - ct[j])) if pool else None
            if best is not None: used.add(best)
            rows.append(dict(video=v, counter=r.counter, 技法=JP.get(r.counter, r.counter),
                             count_value=int(r.count_value), T=round(T, 3),
                             flash_t=round(float(ct[best]), 3) if best is not None else np.nan,
                             lag=round(T - float(ct[best]), 3) if best is not None else np.nan,
                             n_in_win=len(sel)))
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(P.OUT_DIR, "match_grid.csv"), index=False, encoding="utf-8-sig")
    agg = []
    for c, g in m.groupby("counter"):
        lag = g.lag.dropna()
        agg.append(dict(技法=JP.get(c, c), 件数=len(g), 対応=len(lag),
                        中央値=round(float(lag.median()), 3) if len(lag) else np.nan,
                        第3四分位=round(float(lag.quantile(.75)), 3) if len(lag) else np.nan,
                        p90=round(float(lag.quantile(.9)), 3) if len(lag) else np.nan,
                        最大=round(float(lag.max()), 3) if len(lag) else np.nan,
                        超2秒の割合=round(float((lag > 2.0).mean()), 3) if len(lag) else np.nan))
    ag = pd.DataFrame(agg).sort_values("p90", ascending=False)
    ag.to_csv(os.path.join(P.OUT_DIR, "grid_lag_by_counter.csv"), index=False, encoding="utf-8-sig")
    return m, ag

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--stage", default="all", choices=["all", "compare", "eta", "lag"])
    ap.add_argument("--kinds", nargs="*", default=["white"])
    ap.add_argument("--best", nargs=4, metavar=("KIND", "NROW", "NCOL", "BLOB"), default=None)
    ap.add_argument("--q", type=float, default=0.99)
    a = ap.parse_args()
    vids = a.videos or P.VIDEOS
    if a.stage in ("all", "compare"):
        df = evaluate(vids, kinds=tuple(a.kinds))
        print("=== 升目の粗さ・平滑化・しきい値（対照＝±5〜±13秒のずらし） ===")
        print(df.sort_values("差05", ascending=False).to_string(index=False))
    if a.best:
        kind, nrow, ncol, blob = a.best[0], int(a.best[1]), int(a.best[2]), a.best[3].lower() in ("1", "true", "blob")
        if a.stage in ("all", "eta"):
            e = eta(vids, kind, nrow, ncol, blob)
            g = e[e.group == "全体"]
            w = g[(g.rel_sec >= -1.3) & (g.rel_sec <= 0.3)]
            print(f"\n=== 近い対照での時刻そろえ平均（{kind} {ncol}x{nrow} blob={blob}） ===")
            print(f"窓 -1.3〜+0.3 秒の平均 z ＝ {w.z.mean():.2f} ／ 最大 z ＝ {w.z.max():.2f}（相対時刻 {w.loc[w.z.idxmax(),'rel_sec']}）")
            h = e[e.group != "全体"]
            hw = h[(h.rel_sec >= -1.3) & (h.rel_sec <= 0.3)]
            print(hw.groupby("group").z.agg(["mean", "max"]).round(2).to_string())
        if a.stage in ("all", "lag"):
            m, ag = lag_table(vids, kind, nrow, ncol, blob, a.q)
            print(f"\n=== 閃光→加点の遅れ（技法別・しきい値 上位{(1-a.q)*100:.1f}%） ===")
            print(ag.to_string(index=False))
