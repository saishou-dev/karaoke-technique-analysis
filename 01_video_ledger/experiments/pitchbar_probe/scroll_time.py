# -*- coding: utf-8 -*-
"""3つ目の候補＝**位置から時刻を逆算する**（本人の着想・2026-10-22 16:26）。

本人の着想（原文の趣旨）＝「音程バーは歌唱バーの速度から逆算できない？ 歌い終わった区間のバーは
色が変わる。技法マークもバー上の位置に出る。だから画面上の x ずれは x÷v 秒のずれに直せる」。

画面の実際の仕組み（2026-10-08 設計メモ v0 の実測。本人の言い方を 1 点だけ直す）:
  バーは**静止**していて、**カーソル（再生位置）が左→右へ等速で動く**。右端でページが切り替わる。
  つまり「バーが流れる」のではなく「カーソルが走る」。
  速度はページごとに違う（9/24 lulu. で 152〜397 px/秒）。1 ページ 4〜11 秒。
  向きは本人の言う左流れではなく右進みだが、**x ずれ ÷ v ＝ 時刻ずれ**という関係は本人の言うとおり成り立つ。

この候補が効く理由（先生の指摘に直接効く）:
  カウンターも技法マークも「判定が終わってから」出るので、出た時刻には判定の遅れが乗る。
  ところが**出る場所**は、その技法を歌ったバーの位置そのもので、遅れの影響を受けない。
  だから位置を x÷v で時刻に直せば、判定の遅れを抜いた「歌った時刻」が取れるはずだ。
  ビブラートのように判定に時間がかかる技法ほど、この差が大きく出る。

実装（復号はしない。2段目で保存した升目 grid_<video>.npz をそのまま使う）:
  1. 速度 v の実測＝カーソルではなく**金色の進み先（歌い終わった区間の右端）**を追う。
     カーソルは明るい背景で読めず検出率 17〜25% だったが、金色は歌った後ずっと残るので追いやすい。
  2. 進み先 x(t) を区間（ページ）に切り、区間ごとに直線 x = v·(t − t0) を当てる。
  3. 候補の発光について、出た時刻 t と**出た列 c** の両方を記録する。
  4. 列 c を時刻に直す t_pos = t0 + x(c)/v（その時点のページの直線を使う）。
  5. t（出た時刻）と t_pos（位置から逆算した時刻）を、同じ対照つきの手順でカウンター増分と突き合わせる。

出力: <OUT_DIR>/scroll_speed.csv ・ scroll_compare.csv ・ scroll_lag_by_counter.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P
import detect_grid as D
from control import SHIFTS as FAR_SHIFTS, rate

NEAR_SHIFTS = [-3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0]
GOLD_MIN = 40           # 1 列が「歌い終わった」とみなす金色の画素数
V_LO, V_HI = 120.0, 520.0   # 速度の許容範囲 [px/秒]（設計メモ v0 の 152〜397 を少し広げた）
MIN_RAMP_SEC = 1.2
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}

FRONT_Q = 0.97          # 金色の量の累積がこの割合に達する列を「進み先」とする

def front(video, qmass=FRONT_Q):
    """金色の進み先＝歌い終わった区間の右端。
    「金色が立っているいちばん右の列」にすると背景映像の暖色のちらつきに引っ張られるので、
    **金色の量の累積が 97% に達する列**を使う（右端のわずかなちらつきは量が小さく効かない）。"""
    g, t, crop = D.load(video)
    gold = g[:, D.KIND["gold"]].sum(axis=1).astype(np.float32)   # (N, ncol)
    ncol = gold.shape[1]
    px_per_col = crop[2] / ncol
    tot = gold.sum(axis=1)
    cs = np.cumsum(gold, axis=1)
    thr = (qmass * tot)[:, None]
    idx = np.argmax(cs >= thr, axis=1).astype(np.float32)
    x = np.where(tot >= GOLD_MIN * 3, (idx + 0.5) * px_per_col, np.nan)
    return t, x, px_per_col, crop

def theilsen(t, x, nmax=400):
    """中央値の傾き（外れ値に強い）。点が多いときは間引く。"""
    n = len(t)
    if n < 4: return None
    k = np.linspace(0, n - 1, min(n, nmax)).astype(int)
    tt, xx = t[k], x[k]
    i, j = np.triu_indices(len(tt), 1)
    dt = tt[j] - tt[i]
    m = dt > 0.15
    if m.sum() < 5: return None
    v = np.median((xx[j][m] - xx[i][m]) / dt[m])
    b = np.median(xx - v * tt)
    return float(v), float(b)

def ramps(t, x, min_sec=MIN_RAMP_SEC):
    """進み先が右へ伸びる区間（＝1 ページ）に切り、区間ごとに直線を当てる。
    ページが替わると進み先は左へ大きく戻るので、その落ち込みを切れ目にする。"""
    good = np.isfinite(x)
    out = []
    i = 0
    N = len(x)
    while i < N:
        if not good[i]: i += 1; continue
        j = i + 1
        last = x[i]
        while j < N:
            if not good[j]: j += 1; continue
            if x[j] < last - 200:   # 200px 以上の戻り＝ページ切替
                break
            last = max(last, x[j]); j += 1
        seg = np.arange(i, j)[good[i:j]]
        if len(seg) >= 10 and (t[seg[-1]] - t[seg[0]]) >= min_sec:
            # 単調増加の包絡だけを使う（演出で一時的に消えた列を落とす）
            xs = np.maximum.accumulate(x[seg])
            fit = theilsen(t[seg], xs)
            if fit and V_LO <= fit[0] <= V_HI:
                v, b = fit
                out.append(dict(i0=int(seg[0]), i1=int(seg[-1]),
                                t0=float(t[seg[0]]), t1=float(t[seg[-1]]),
                                v=v, b=b, tcross=float(-b / v), n=len(seg)))
        i = j
    return out

def pos_to_time(rs, t_frame, x_px):
    """出た時刻 t_frame の時点で効いているページの直線を使い、列の位置 x_px を時刻に直す"""
    cand = [r for r in rs if r["t0"] - 0.5 <= t_frame <= r["t1"] + 0.5]
    if not cand:
        cand = sorted(rs, key=lambda r: min(abs(r["t0"] - t_frame), abs(r["t1"] - t_frame)))[:1]
    if not cand: return np.nan
    r = cand[0]
    return r["tcross"] + x_px / r["v"]

def score_with_pos(video, kind, nrow, ncol, blob=False, chunk=4000):
    """2段目のスコアに加えて、そのコマでいちばん立った升目の**列**も返す"""
    g, t, crop = D.load(video)
    a = D.coarsen(g, nrow, ncol)[:, D.KIND[kind]].astype(np.float32)
    N = a.shape[0]
    kh, kw = D.blob_kernel(crop, nrow, ncol) if blob else (1, 1)
    s = np.full(N, np.nan, np.float32); col = np.full(N, -1, np.int32)
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk); lo = max(0, i0 - 2)
        blk = a[lo:i1]
        d = blk[2:] - blk[:-2]
        if d.size == 0: continue
        if kh > 1: d = D.boxsum(d, kh, 1)
        if kw > 1: d = D.boxsum(d, kw, 2)
        f = d.reshape(d.shape[0], -1)
        am = f.argmax(axis=1)
        s[lo + 2:lo + 2 + len(am)] = f.max(axis=1) - np.median(f, axis=1)
        col[lo + 2:lo + 2 + len(am)] = am % ncol
    return t, s, col, crop

def candidates_with_pos(video, kind, nrow, ncol, blob, q):
    t, s, col, crop = score_with_pos(video, kind, nrow, ncol, blob)
    pk, thr = P.peaks(t, s, q=q)
    times = np.array([p[0] for p in pk])
    idx = np.searchsorted(t, times)
    idx = np.clip(idx, 0, len(t) - 1)
    px = (col[idx] + 0.5) * (crop[2] / ncol)
    return times, px

def evaluate(videos, kind="white", nrow=8, ncol=96, blob=False, qs=(0.95, 0.99, 0.995),
             shifts=FAR_SHIFTS, lag_cap=4.0):
    g1 = pd.read_csv(P.G1_CSV)
    speed_rows = []; rows = []
    per_video = {}
    for v in videos:
        t, x, ppc, crop = front(v)
        rs = ramps(t, x)
        vs = np.array([r["v"] for r in rs])
        speed_rows.append(dict(video=v, ページ数=len(rs),
                               速度の中央値=round(float(np.median(vs)), 1) if len(vs) else np.nan,
                               最小=round(float(vs.min()), 1) if len(vs) else np.nan,
                               最大=round(float(vs.max()), 1) if len(vs) else np.nan,
                               四分位範囲=round(float(np.percentile(vs, 75) - np.percentile(vs, 25)), 1) if len(vs) else np.nan,
                               進み先が読めたコマの割合=round(float(np.isfinite(x).mean()), 3)))
        per_video[v] = (rs, t, x)
    pd.DataFrame(speed_rows).to_csv(os.path.join(P.OUT_DIR, "scroll_speed.csv"),
                                    index=False, encoding="utf-8-sig")
    for q in qs:
        for mode in ("出た時刻", "位置から逆算"):
            n_ev = n_cand = 0; r05 = r10 = 0.0; c05 = []; c10 = []; dur_sum = 0.0
            for v in videos:
                rs, tt, xx = per_video[v]
                times, px = candidates_with_pos(v, kind, nrow, ncol, blob, q)
                if mode == "位置から逆算":
                    ct = np.array([pos_to_time(rs, a, b) for a, b in zip(times, px)])
                    keep = np.isfinite(ct) & (ct <= times + 0.2) & (ct >= times - lag_cap)
                    ct = np.sort(ct[keep])
                else:
                    ct = np.sort(times)
                dur = float(tt.max()); dur_sum += dur
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
            rows.append(dict(時刻の取り方=mode, しきい値=f"上位{(1-q)*100:.1f}%",
                             候補数=n_cand, 密度=round(n_cand / dur_sum, 3),
                             r05=round(a05, 3), 対照05=round(b05, 3), 差05=round(a05 - b05, 3),
                             r10=round(a10, 3), 対照10=round(b10, 3), 差10=round(a10 - b10, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P.OUT_DIR, "scroll_compare.csv"), index=False, encoding="utf-8-sig")
    return pd.DataFrame(speed_rows), df

def lag_by_counter(videos, kind="white", nrow=8, ncol=96, blob=False, q=0.99,
                   win_lo=-3.0, win_hi=0.5):
    """位置から逆算した時刻を使ったときの、技法ごとの遅れ（カウンター増分 − 逆算時刻）"""
    g1 = pd.read_csv(P.G1_CSV)
    rows = []
    for v in videos:
        t, x, ppc, crop = front(v)
        rs = ramps(t, x)
        times, px = candidates_with_pos(v, kind, nrow, ncol, blob, q)
        tp = np.array([pos_to_time(rs, a, b) for a, b in zip(times, px)])
        keep = np.isfinite(tp) & (tp <= times + 0.2) & (tp >= times - 4.0)
        ct = np.sort(tp[keep])
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
                             T=round(T, 3),
                             t_pos=round(float(ct[best]), 3) if best is not None else np.nan,
                             lag=round(T - float(ct[best]), 3) if best is not None else np.nan))
    m = pd.DataFrame(rows)
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
    ag.to_csv(os.path.join(P.OUT_DIR, "scroll_lag_by_counter.csv"), index=False, encoding="utf-8-sig")
    return m, ag

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--kind", default="white")
    ap.add_argument("--nrow", type=int, default=8)
    ap.add_argument("--ncol", type=int, default=96)
    ap.add_argument("--q", type=float, default=0.99)
    a = ap.parse_args()
    vids = a.videos or P.VIDEOS
    sp, df = evaluate(vids, a.kind, a.nrow, a.ncol, False)
    print("=== カーソル速度の実測（金色の進み先から） ===")
    print(sp.to_string(index=False))
    print()
    print("=== 出た時刻 と 位置から逆算した時刻（対照＝±5〜±13秒のずらし） ===")
    print(df.to_string(index=False))
    m, ag = lag_by_counter(vids, a.kind, a.nrow, a.ncol, False, a.q)
    print()
    print(f"=== 位置から逆算した時刻を使ったときの技法ごとの遅れ（上位{(1-a.q)*100:.1f}%） ===")
    print(ag.to_string(index=False))
