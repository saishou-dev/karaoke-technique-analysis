# -*- coding: utf-8 -*-
"""音程バー領域の位置合わせ確認（手順1）。
3 曲それぞれについて、カウンター増分の時刻の前後から数コマを原寸で取り出し、
音程バー領域（黄）とカウンター帯（赤）を枠で描いた画像を書き出す。
曲ごとに位置が変わるかどうかを band_probe_v2 の値と並べて記録する。
出力: <OUT_DIR>/roi_check/<video>_T<時刻>.png ・ roi_check/roi_geometry.csv
"""
import os, sys, argparse
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P

def dump(video, times, band):
    crop = P.pitchbar_roi(video, band)
    bx, by, bw, bh = P.band_roi(video, band)
    # 両方が入る大枠（原寸・等倍で切り出す）
    x0 = min(crop[0], bx) - 20
    y0 = crop[1] - 20
    x1 = max(crop[0] + crop[2], bx + bw) + 20
    y1 = by + bh + 20
    x0, y0 = max(0, x0), max(0, y0)
    cw, ch = (x1 - x0) & ~1, (y1 - y0) & ~1
    path = os.path.join(P.VID_DIR, video + ".mp4")
    for T in times:
        got = None
        for i, bgr in P.decode(path, max(0.0, T - 0.05), 4, (x0, y0, cw, ch), div=1, tag="roi"):
            got = bgr
            break
        if got is None: continue
        img = got.copy()
        cv2.rectangle(img, (crop[0] - x0, crop[1] - y0),
                      (crop[0] + crop[2] - x0, crop[1] + crop[3] - y0), (0, 255, 255), 4)
        cv2.rectangle(img, (bx - x0, by - y0), (bx + bw - x0, by + bh - y0), (0, 0, 255), 4)
        cv2.putText(img, f"{video}  T={T:.3f}  pitchbar(yellow)={crop}  band(red)=({bx},{by},{bw},{bh})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        P.imwrite_u(os.path.join(P.OUT_DIR, "roi_check", f"{video}_T{T:.3f}.png"), img)
        P.log(f"[{video}] ROI 確認画像 T={T:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    vids = a.videos or P.VIDEOS
    band = pd.read_csv(P.BAND_CSV)
    g1 = pd.read_csv(P.G1_CSV)
    geo = []
    for v in vids:
        crop = P.pitchbar_roi(v, band); b = P.band_roi(v, band)
        r = band[band.video == v].iloc[0]
        geo.append(dict(video=v, xL=int(r.xL), yL=int(r.yL), w=int(r.w), scale=float(r.scale),
                        angle_deg=float(r.angle_deg),
                        pb_x=crop[0], pb_y=crop[1], pb_w=crop[2], pb_h=crop[3]))
        ts = g1[g1.video == v].sort_values("video_time_sec").video_time_sec.astype(float).values
        pick = ts[len(ts) // 4], ts[len(ts) // 2], ts[3 * len(ts) // 4]
        dump(v, list(pick)[:a.n], band)
    df = pd.DataFrame(geo)
    df.to_csv(os.path.join(P.OUT_DIR, "roi_check", "roi_geometry.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string())
