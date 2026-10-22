# -*- coding: utf-8 -*-
"""升目法（格子分割に基づく局所時間差分）の背景対策＝背景差分を 1 案として試す。

2段目までの背景対策＝**全升目の空間中央値を引く**（同じコマの中で、他の升目と比べて突出した升目だけを残す）。
今回足す案＝**背景差分**（直前 N 秒の升目ごとの中央値画像との差）。

標準名と出自（#40）:
  升目法        ＝ 格子分割に基づく局所時間差分（block-based temporal differencing）
                   Zhang, Kankanhalli & Smoliar 1993, DOI 10.1007/BF01210504
  背景差分      ＝ 背景モデルとの差分（background subtraction）
                   Stauffer & Grimson 1999, DOI 10.1109/CVPR.1999.784637
                   （本実装は混合ガウス分布ではなく、直前 N 秒の中央値を背景とする簡易版）

3 通りを同じ評価（対照つき）で比べる:
  A 空間中央値のみ   s(t) = max_cell[ d(t) − median_cell d(t) ]           ← 2段目まで
  B 背景差分のみ     s(t) = max_cell[ g(t) − 直前N秒の中央値 ]
  C 両方            B の値からさらに同じコマの空間中央値を引く

出力: <OUT_DIR>/bgsub_compare.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P
import detect_grid as D
from control import SHIFTS as FAR_SHIFTS, rate

NEAR_SHIFTS = [-3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0]

def bg_median(a, blk):
    """直前の塊（blk コマ）の升目ごとの中央値を背景とする。
    塊ごとにまとめて計算するので、1 コマずつ移動中央値を取るより桁違いに速い。"""
    N = a.shape[0]
    nb = int(np.ceil(N / blk))
    med = np.zeros((nb,) + a.shape[1:], np.float32)
    for b in range(nb):
        med[b] = np.median(a[b * blk:(b + 1) * blk], axis=0)
    bg = np.empty_like(a)
    for b in range(nb):
        src = med[b - 1] if b >= 1 else med[0]      # 直前の塊（先頭だけ自分自身）
        bg[b * blk:(b + 1) * blk] = src
    return bg

def score(video, kind, nrow, ncol, mode, bg_sec=2.0):
    g, t, crop = D.load(video)
    a = D.coarsen(g, nrow, ncol)[:, D.KIND[kind]].astype(np.float32)
    N = a.shape[0]
    if mode == "A_空間中央値":
        d = np.full_like(a, np.nan); d[2:] = a[2:] - a[:-2]
        f = d.reshape(N, -1)
        return t, f.max(axis=1) - np.nanmedian(f, axis=1)
    blk = max(2, int(bg_sec * P.FPS_NOMINAL))
    d = a - bg_median(a, blk)
    f = d.reshape(N, -1)
    if mode == "B_背景差分":
        return t, f.max(axis=1)
    return t, f.max(axis=1) - np.median(f, axis=1)      # C

def candidates(video, kind, nrow, ncol, mode, q, bg_sec=2.0):
    t, s = score(video, kind, nrow, ncol, mode, bg_sec)
    pk, thr = P.peaks(t, s, q=q)
    return np.sort(np.array([p[0] for p in pk]))

def eta(videos, kind, nrow, ncol, mode, bg_sec=2.0, shifts=NEAR_SHIFTS, pre=3.0, post=1.0):
    g1 = pd.read_csv(P.G1_CSV)
    FPS = P.FPS_NOMINAL
    n_pre, n_post = int(pre * FPS), int(post * FPS)
    R = []; C = []
    for v in videos:
        t, s = score(v, kind, nrow, ncol, mode, bg_sec)
        nn = len(s); dur = float(t.max())
        Ts = g1[g1.video == v].video_time_sec.astype(float).values
        for tag, offs in (("r", [0.0]), ("c", shifts)):
            for off in offs:
                for T in Ts:
                    T2 = (T + off) % dur if off else T
                    j = int(np.searchsorted(t, T2)); x0, x1 = j - n_pre, j + n_post
                    if x0 < 0 or x1 > nn: continue
                    (R if tag == "r" else C).append(s[x0:x1])
    R = np.vstack(R); C = np.vstack(C)
    mr = np.nanmean(R, 0); mc = np.nanmean(C, 0)
    se = np.nanstd(C, 0) / np.sqrt(len(R))
    z = (mr - mc) / np.where(se > 0, se, np.nan)
    rel = (np.arange(n_pre + n_post) - n_pre) / FPS
    w = (rel >= -1.3) & (rel <= 0.3)
    return float(np.nanmean(z[w])), float(np.nanmax(z[w])), float(rel[w][np.nanargmax(z[w])])

def main(videos, kind="white", grids=((8, 96), (1, 96)), qs=(0.95, 0.99, 0.995), bg_sec=2.0):
    g1 = pd.read_csv(P.G1_CSV)
    rows = []
    for nrow, ncol in grids:
        for mode in ("A_空間中央値", "B_背景差分", "C_両方"):
            zm, zx, zt = eta(videos, kind, nrow, ncol, mode, bg_sec)
            for q in qs:
                n_ev = n_cand = 0; r05 = r10 = 0.0; c05 = []; c10 = []; dur = 0.0
                for v in videos:
                    ct = candidates(v, kind, nrow, ncol, mode, q, bg_sec)
                    _, t, _ = D.load(v); dd = float(t.max()); dur += dd
                    Ts = np.sort(g1[g1.video == v].video_time_sec.astype(float).values)
                    n_ev += len(Ts); n_cand += len(ct)
                    r05 += rate(ct, Ts, 0.5) * len(Ts); r10 += rate(ct, Ts, 1.0) * len(Ts)
                    for sh in FAR_SHIFTS:
                        Ts2 = np.sort((Ts + sh) % dd)
                        c05.append(rate(ct, Ts2, 0.5) * len(Ts2)); c10.append(rate(ct, Ts2, 1.0) * len(Ts2))
                ns = len(FAR_SHIFTS)
                a05 = r05 / n_ev; a10 = r10 / n_ev
                b05 = sum(c05) / (n_ev * ns); b10 = sum(c10) / (n_ev * ns)
                rows.append(dict(背景対策=mode, 升目=f"{ncol}x{nrow}", しきい値=f"上位{(1-q)*100:.1f}%",
                                 候補数=n_cand, r05=round(a05, 3), 対照05=round(b05, 3),
                                 差05=round(a05 - b05, 3), 差10=round(a10 - b10, 3),
                                 近い対照の平均z=round(zm, 2), 最大z=round(zx, 2), 最大zの相対時刻=round(zt, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P.OUT_DIR, "bgsub_compare.csv"), index=False, encoding="utf-8-sig")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--kind", default="white")
    ap.add_argument("--bg_sec", type=float, default=2.0)
    a = ap.parse_args()
    df = main(a.videos or P.VIDEOS, a.kind, bg_sec=a.bg_sec)
    print(df.sort_values("差05", ascending=False).to_string(index=False))
