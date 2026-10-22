# -*- coding: utf-8 -*-
"""発光の瞬間の根拠画像（手順4の裏付け）。
match.csv で対応が付いた発光の時刻 t について、t−6f / t / t+6f / t+18f の 4 コマを
音程バー領域から原寸で切り出し、横に並べた 1 枚にする。黄枠＝発光のフレーム。
出力: <OUT_DIR>/evidence/<video>_<技法>_<count>_F<発光時刻>_T<カウンター時刻>.png
"""
import os, sys, argparse
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P

JP = {"shakuri": "shakuri", "kobushi": "kobushi", "fall": "fall",
      "vibrato": "vibrato", "accent": "accent", "hammering": "hammering"}

def tile(video, flash_t, T, counter, cv_, band, method):
    crop = P.pitchbar_roi(video, band)
    path = os.path.join(P.VID_DIR, video + ".mp4")
    fps = P.FPS_NOMINAL
    t0 = max(0.0, flash_t - 8 / fps)
    frames = []
    for i, bgr in P.decode(path, t0, 28, crop, div=2, tag="ev"):
        frames.append(bgr)
    pts = P.read_pts(P.decode.last_errpath)
    nn = min(len(frames), len(pts))
    if nn == 0: return None
    j = int(np.argmin([abs(p - flash_t) for p in pts[:nn]]))
    tiles = []
    for k in (j - 6, j, j + 6, j + 18):
        k = min(max(0, k), nn - 1)
        t = frames[k].copy()
        t = cv2.resize(t, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{pts[k]:.3f}s ({k-j:+d}f)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if k == j:
            cv2.rectangle(t, (2, 2), (t.shape[1] - 3, t.shape[0] - 3), (0, 255, 255), 4)
        tiles.append(t)
    img = np.vstack(tiles)
    cv2.putText(img, f"{method}  flash={flash_t:.3f}  counter={T:.3f}  lag={T-flash_t:+.3f}s  {counter} #{cv_}",
                (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None)
    ap.add_argument("--per_video", type=int, default=2)
    ap.add_argument("--match_file", default="match_local.csv")
    a = ap.parse_args()
    m = pd.read_csv(os.path.join(P.OUT_DIR, a.match_file))
    meth = a.method or m.method.iloc[0]
    band = pd.read_csv(P.BAND_CSV)
    sel = m[(m.method == meth) & m.lag.notna()]
    for v, g in sel.groupby("video"):
        med = g.lag.median()
        # 遅れが中央値に近い（＝典型的な）ものを選ぶ。技法はなるべく散らす。
        g = g.assign(d=(g.lag - med).abs()).sort_values("d")
        picked = []
        seen = set()
        for _, r in g.iterrows():
            if r.counter in seen and len(picked) < a.per_video: continue
            picked.append(r); seen.add(r.counter)
            if len(picked) >= a.per_video: break
        for r in picked:
            img = tile(v, float(r.flash_t), float(r["T"]), r.counter, int(r.count_value), band, meth)
            if img is None: continue
            fn = f"{v}_{JP.get(r.counter, r.counter)}_{int(r.count_value):03d}_F{float(r.flash_t):.3f}_T{float(r['T']):.3f}.png"
            P.imwrite_u(os.path.join(P.OUT_DIR, "evidence", fn), img)
            P.log(f"[{v}] 根拠画像 {fn}")
