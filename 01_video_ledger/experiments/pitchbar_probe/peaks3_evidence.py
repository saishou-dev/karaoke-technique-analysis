# -*- coding: utf-8 -*-
"""3段目：当たり・外れの根拠画像と、技法別の遅れの対照（偶然あたり）。

根拠画像＝当たり（±0.5 秒以内に候補があった加点）3 枚・外れ（窓 [T−3, T+0.5] に候補が 1 つも無かった加点）3 枚。
対照＝技法別の遅れの表を、時刻をずらした偽イベント列でも作る。実データの表と並べて、
      技法ごとの違いが偶然で説明できるかどうかを見る。

出力:
  <OUT_DIR>/peaks3_evidence/hit_*.png / miss_*.png
  <OUT_DIR>/peaks3_lag_control.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P
import detect_grid as D
import peaks3 as K

JP = K.JP

def tile(video, center_t, label, band, color=(0, 255, 255)):
    """音程ガイド表示の領域から、中心時刻の前後 4 コマを縦に並べた 1 枚を作る"""
    crop = P.pitchbar_roi(video, band)
    path = os.path.join(P.VID_DIR, video + ".mp4")
    fps = P.FPS_NOMINAL
    frames = []
    for i, bgr in P.decode(path, max(0.0, center_t - 10 / fps), 32, crop, div=2, tag="ev3"):
        frames.append(bgr)
    pts = P.read_pts(P.decode.last_errpath)
    nn = min(len(frames), len(pts))
    if nn == 0: return None
    j = int(np.argmin([abs(p - center_t) for p in pts[:nn]]))
    tiles = []
    for k in (j - 8, j, j + 8, j + 20):
        k = min(max(0, k), nn - 1)
        im = cv2.resize(frames[k], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"{pts[k]:.3f}s ({k-j:+d}f)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if k == j: cv2.rectangle(im, (2, 2), (im.shape[1] - 3, im.shape[0] - 3), color, 4)
        tiles.append(im)
    img = np.vstack(tiles)
    cv2.putText(img, label, (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

def lag_control(videos, cfg, k, gap, win_lo=-3.0, win_hi=0.5, shifts=K.NEAR_SHIFTS):
    """技法別の遅れを、時刻をずらした偽イベント列でも作る（偶然あたりの形）"""
    g1 = pd.read_csv(P.G1_CSV)
    rows = []
    for v in videos:
        ct = np.sort(K.candidates(v, cfg, k, gap))
        _, t, _ = D.load(v); dur = float(t.max())
        ev = g1[g1.video == v]
        for _, r in ev.iterrows():
            for sh in shifts:
                T = (float(r.video_time_sec) + sh) % dur
                near = ct[(ct >= T - 0.5) & (ct <= T + 0.5)]
                sel = ct[(ct >= T + win_lo) & (ct <= T + win_hi)]
                lag = (T - sel[np.argmin(np.abs(T - sel))]) if len(sel) else np.nan
                rows.append(dict(counter=r.counter, 当たり05=int(len(near) > 0), lag=lag))
    m = pd.DataFrame(rows)
    agg = []
    for c, g in m.groupby("counter"):
        lg = g.lag.dropna()
        agg.append(dict(技法=JP.get(c, c), 対照の件数=len(g),
                        対照の当たり率05=round(float(g.当たり05.mean()), 3),
                        対照の中央値=round(float(lg.median()), 3) if len(lg) else np.nan,
                        対照のp90=round(float(lg.quantile(.9)), 3) if len(lg) else np.nan,
                        対照の超2秒の割合=round(float((lg > 2.0).mean()), 3) if len(lg) else np.nan))
    ag = pd.DataFrame(agg)
    ag.to_csv(os.path.join(P.OUT_DIR, "peaks3_lag_control.csv"), index=False, encoding="utf-8-sig")
    return ag

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="1/4縮小 96列 背景差分")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--gap", type=float, default=0.30)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    vids = P.VIDEOS
    ag = lag_control(vids, a.cfg, a.k, a.gap)
    print("=== 技法別の遅れ（対照＝時刻をずらした偽イベント列） ===")
    print(ag.to_string(index=False))

    m = pd.read_csv(os.path.join(P.OUT_DIR, "peaks3_match.csv"))
    band = pd.read_csv(P.BAND_CSV)
    d = os.path.join(P.OUT_DIR, "peaks3_evidence")
    hits = m[m.当たり05 == 1].copy()
    hits["d"] = hits.lag.abs()
    hits = hits.sort_values("d").drop_duplicates("技法").head(a.n)
    miss = m[m.flash_t.isna()].drop_duplicates("技法").head(a.n)
    for _, r in hits.iterrows():
        lab = f"HIT {r['技法']} #{int(r.count_value)} flash={r.flash_t:.3f} counter={r['T']:.3f} lag={r.lag:+.3f}s"
        img = tile(r.video, float(r.flash_t), lab, band, (0, 255, 0))
        if img is not None:
            P.imwrite_u(os.path.join(d, f"hit_{r.video}_{r.counter}_{int(r.count_value):03d}.png"), img)
            P.log(f"当たりの根拠画像 {r.video} {r.counter} #{int(r.count_value)}")
    for _, r in miss.iterrows():
        lab = f"MISS {r['技法']} #{int(r.count_value)} counter={r['T']:.3f} (no candidate in -3.0..+0.5s)"
        img = tile(r.video, float(r["T"]) - 0.8, lab, band, (0, 0, 255))
        if img is not None:
            P.imwrite_u(os.path.join(d, f"miss_{r.video}_{r.counter}_{int(r.count_value):03d}.png"), img)
            P.log(f"外れの根拠画像 {r.video} {r.counter} #{int(r.count_value)}")
