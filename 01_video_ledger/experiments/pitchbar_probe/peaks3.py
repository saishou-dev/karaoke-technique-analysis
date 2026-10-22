# -*- coding: utf-8 -*-
"""3段目：背景差分を当てた信号に対して極大値検出（ピーク検出）を作り直す。

2段目の追補で分かったこと＝背景モデルとの差分（background subtraction）は信号を良くする
（イベント時刻同期加算平均の z が 1.05→1.48）。しかし「上位 n% で切る」拾い方では候補が
464 件から 118 件に減り、1 件ごとの検出につながらなかった。信号ではなく拾い方の問題。

作り直す点:
  旧＝曲全体の分位点（上位 n%）で一律に切る。曲の中で背景が明るい区間だけから候補が出てしまう。
  新＝**局所の高さ**で切る。周囲 ±1 秒の中央値を引いてから閾値を当て、さらに**最小間隔**を課す。
      prom(t) = s(t) − median( s(t−1秒 〜 t+1秒) )
      閾値     = k × 1.4826 × MAD(prom)       k を 3 通り
      最小間隔 = 近い候補は高い方だけ残す      3 通り

標準名（材料40）:
  閃光法   ＝ フレーム間差分による変化検出（temporal differencing）＋差分時系列の極大値検出
  升目法   ＝ 格子分割に基づく局所時間差分（block-based temporal differencing）
  背景差分 ＝ 背景モデルとの差分（background subtraction・Stauffer & Grimson 1999 の簡易版）
  対照     ＝ 時刻をずらした対照列による偶然一致率（chance level）
  時間窓   ＝ 照合許容窓（tolerance window）

出力:
  <OUT_DIR>/peaks3_sweep.csv        しきい値×最小間隔×信号 の表（1段目・2段目の行も並べる）
  <OUT_DIR>/peaks3_lag.csv          技法の種類別の遅れ
  <OUT_DIR>/peaks3_match.csv        1 件ごとの対応
  <OUT_DIR>/peaks3_evidence/        当たり・外れの根拠画像
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P
import detect_grid as D
import bgsub as B
from control import SHIFTS as FAR_SHIFTS, rate

NEAR_SHIFTS = [-3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0]
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール",
      "vibrato": "ビブラート", "accent": "アクセント", "hammering": "ハンマリング"}
KS = [2.0, 3.0, 4.5]              # 閾値（MAD の何倍か）
GAPS = [0.30, 0.60, 1.00]         # 最小間隔 [秒]
LOCAL_SEC = 1.0                   # 局所の高さを測る窓（周囲 ±1 秒）

# ---- 信号の作り方（config 名 → (loader, kind, nrow, ncol, mode)）----
def _load_prof(v):
    """1段目の 1/4 縮小・96 列・縦 1 段（prof の ch 順は gold,white,bright）"""
    z = np.load(os.path.join(P.OUT_DIR, f"prof_{v}.npz"))
    pr = z["prof"].astype(np.float32)
    g = np.stack([pr[:, 1], pr[:, 0], pr[:, 2]], axis=1)      # white,gold,bright へ並べ替え
    return (g[:, :, None, :], z["t"], np.array([0, 0, 3190, 442]))

def _load_grid(v):
    z = np.load(os.path.join(P.OUT_DIR, f"grid_{v}.npz"))
    return (z["grid"].astype(np.float32), z["t"], z["crop"])

CONFIGS = {
    "1/4縮小 96列 背景差分":   (_load_prof, "white", 1, 96, "B_背景差分"),
    "1/2縮小 96列 背景差分":   (_load_grid, "white", 1, 96, "B_背景差分"),
    "1/2縮小 96x8 背景差分":   (_load_grid, "white", 8, 96, "B_背景差分"),
    "1/4縮小 96列 空間中央値": (_load_prof, "white", 1, 96, "A_空間中央値"),
    "1/2縮小 96x8 空間中央値": (_load_grid, "white", 8, 96, "A_空間中央値"),
}

def signal(video, cfg):
    loader, kind, nrow, ncol, mode = CONFIGS[cfg]
    D._cache.clear(); D.load = loader
    return B.score(video, kind, nrow, ncol, mode)

# ---- 新しい極大値検出 ----
def rolling_median(x, w):
    """幅 w の移動中央値（端は内側の値で埋める）"""
    n = len(x)
    w = int(w) | 1
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(xp, w)
    return np.median(sw, axis=1)

def prominence(t, s, local_sec=LOCAL_SEC):
    """局所の高さ＝周囲 ±local_sec 秒の中央値との差"""
    s = np.nan_to_num(np.asarray(s, float), nan=0.0)
    w = int(2 * local_sec * P.FPS_NOMINAL)
    return s - rolling_median(s, w)

def pick(t, prom, k, gap_sec):
    """中央絶対偏差の k 倍を超える極大値を、高い順に最小間隔を守って採る"""
    mad = np.median(np.abs(prom - np.median(prom))) * 1.4826
    thr = k * (mad if mad > 0 else 1.0)
    cand = np.where((prom[1:-1] >= thr) & (prom[1:-1] >= prom[:-2]) & (prom[1:-1] >= prom[2:]))[0] + 1
    if len(cand) == 0: return np.array([]), float(thr)
    order = cand[np.argsort(-prom[cand])]
    taken = []
    for i in order:
        ti = t[i]
        if all(abs(ti - t[j]) >= gap_sec for j in taken):
            taken.append(i)
    return np.sort(t[np.array(sorted(taken))]), float(thr)

def candidates(video, cfg, k, gap):
    t, s = signal(video, cfg)
    return pick(t, prominence(t, s), k, gap)[0]

# ---- 評価（1段目・2段目と同じ手順）----
def eval_rows(videos, cfg, k, gap, g1):
    n_ev = n_cand = 0; r05 = r10 = 0.0; c05 = []; c10 = []; dur = 0.0
    for v in videos:
        ct = candidates(v, cfg, k, gap)
        _, t, _ = D.load(v); dd = float(t.max()); dur += dd
        Ts = np.sort(g1[g1.video == v].video_time_sec.astype(float).values)
        n_ev += len(Ts); n_cand += len(ct)
        r05 += rate(ct, Ts, 0.5) * len(Ts); r10 += rate(ct, Ts, 1.0) * len(Ts)
        for sh in FAR_SHIFTS:
            T2 = np.sort((Ts + sh) % dd)
            c05.append(rate(ct, T2, 0.5) * len(T2)); c10.append(rate(ct, T2, 1.0) * len(T2))
    ns = len(FAR_SHIFTS)
    a05, a10 = r05 / n_ev, r10 / n_ev
    b05, b10 = sum(c05) / (n_ev * ns), sum(c10) / (n_ev * ns)
    return dict(信号=cfg, しきい値=f"MAD×{k}", 最小間隔=f"{gap:.2f}s",
                候補数=n_cand, 密度=round(n_cand / dur, 3),
                件数=n_ev, 当たり05=int(round(a05 * n_ev)),
                r05=round(a05, 3), 対照05=round(b05, 3), 差05=round(a05 - b05, 3),
                r10=round(a10, 3), 対照10=round(b10, 3), 差10=round(a10 - b10, 3))

def eta(videos, cfg, shifts=NEAR_SHIFTS, pre=3.0, post=1.0):
    g1 = pd.read_csv(P.G1_CSV)
    FPS = P.FPS_NOMINAL
    n_pre, n_post = int(pre * FPS), int(post * FPS)
    R = []; C = []
    for v in videos:
        t, s = signal(v, cfg)
        pr = prominence(t, s)
        nn = len(pr); dur = float(t.max())
        Ts = g1[g1.video == v].video_time_sec.astype(float).values
        for tag, offs in (("r", [0.0]), ("c", shifts)):
            for off in offs:
                for T in Ts:
                    T2 = (T + off) % dur if off else T
                    j = int(np.searchsorted(t, T2)); x0, x1 = j - n_pre, j + n_post
                    if x0 < 0 or x1 > nn: continue
                    (R if tag == "r" else C).append(pr[x0:x1])
    R = np.vstack(R); C = np.vstack(C)
    mr, mc = np.nanmean(R, 0), np.nanmean(C, 0)
    se = np.nanstd(C, 0) / np.sqrt(len(R))
    z = (mr - mc) / np.where(se > 0, se, np.nan)
    rel = (np.arange(n_pre + n_post) - n_pre) / FPS
    w = (rel >= -1.3) & (rel <= 0.3)
    return float(np.nanmean(z[w])), float(np.nanmax(z[w])), float(rel[w][np.nanargmax(z[w])])

def match(videos, cfg, k, gap, win_lo=-3.0, win_hi=0.5):
    g1 = pd.read_csv(P.G1_CSV)
    rows = []
    for v in videos:
        ct = np.sort(candidates(v, cfg, k, gap))
        ev = g1[g1.video == v].sort_values("video_time_sec")
        used = set()
        for _, r in ev.iterrows():
            T = float(r.video_time_sec)
            near = ct[(ct >= T - 0.5) & (ct <= T + 0.5)]
            sel = np.where((ct >= T + win_lo) & (ct <= T + win_hi))[0]
            free = [j for j in sel if j not in used]
            pool = free if free else list(sel)
            best = min(pool, key=lambda j: abs(T - ct[j])) if pool else None
            if best is not None: used.add(best)
            rows.append(dict(video=v, counter=r.counter, 技法=JP.get(r.counter, r.counter),
                             count_value=int(r.count_value), multi=int(r.multi_increment),
                             T=round(T, 3),
                             当たり05=int(len(near) > 0),
                             flash_t=round(float(ct[best]), 3) if best is not None else np.nan,
                             lag=round(T - float(ct[best]), 3) if best is not None else np.nan))
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(P.OUT_DIR, "peaks3_match.csv"), index=False, encoding="utf-8-sig")
    agg = []
    for c, g in m.groupby("counter"):
        lag = g.lag.dropna()
        agg.append(dict(技法=JP.get(c, c), 件数=len(g), 当たり05=int(g.当たり05.sum()),
                        当たり率05=round(float(g.当たり05.mean()), 3), 対応=len(lag),
                        中央値=round(float(lag.median()), 3) if len(lag) else np.nan,
                        第3四分位=round(float(lag.quantile(.75)), 3) if len(lag) else np.nan,
                        p90=round(float(lag.quantile(.9)), 3) if len(lag) else np.nan,
                        最大=round(float(lag.max()), 3) if len(lag) else np.nan,
                        超2秒の割合=round(float((lag > 2.0).mean()), 3) if len(lag) else np.nan))
    ag = pd.DataFrame(agg).sort_values("p90", ascending=False)
    ag.to_csv(os.path.join(P.OUT_DIR, "peaks3_lag.csv"), index=False, encoding="utf-8-sig")
    return m, ag

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--stage", default="sweep", choices=["sweep", "match"])
    ap.add_argument("--cfg", default="1/4縮小 96列 背景差分")
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--gap", type=float, default=0.6)
    a = ap.parse_args()
    vids = a.videos or P.VIDEOS
    g1 = pd.read_csv(P.G1_CSV)
    if a.stage == "sweep":
        rows = []
        for cfg in CONFIGS:
            zm, zx, zt = eta(vids, cfg)
            for k in KS:
                for gap in GAPS:
                    r = eval_rows(vids, cfg, k, gap, g1)
                    r.update(近い対照の平均z=round(zm, 2), 最大z=round(zx, 2), 最大zの相対時刻=round(zt, 3))
                    rows.append(r)
            P.log(f"[sweep] {cfg} 済")
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(P.OUT_DIR, "peaks3_sweep.csv"), index=False, encoding="utf-8-sig")
        print(df.sort_values("差05", ascending=False).head(20).to_string(index=False))
    else:
        m, ag = match(vids, a.cfg, a.k, a.gap)
        print(ag.to_string(index=False))
        print()
        print("当たり（±0.5秒）", int(m.当たり05.sum()), "/", len(m))
