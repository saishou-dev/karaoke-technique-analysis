# -*- coding: utf-8 -*-
"""2段目：音程バー領域を升目に割って閃光を追うための復号（1 巡だけ）。

1段目の結論＝発光は白い閃光として加点の約 0.3〜1.3 秒前に出るが、横 96 本の短冊では
縦 442 画素を全部足し込むため、閃光（直径 100 画素ほど）が背景映像の明滅に薄められた。

本スクリプトは**いちばん細かい升目（縦 16 × 横 192）だけを保存する**。
粗い升目（96×8・48×4）は保存した升目を足し合わせて作れるので、復号は 1 回で済む。

1段目との違い:
  - 縮小率を 1/4 から **1/2** に上げた（1/4 では細い閃光が周りの暗い画素と平均されて消えるため）
  - 縦を 16 段に割った（1段目は縦を割らず 1 本に潰していた）

保存＝<OUT_DIR>/grid_<video>.npz
  grid ＝ (コマ数, 3, 16, 192) の uint8。3 種＝[白い画素, 金色の画素, 明るい画素]
  t    ＝ コマごとの pts 秒
升目 1 つの大きさ＝原寸で約 16.6 × 27.6 画素（1/2 縮小で 8.3 × 13.8 画素）。
升目の中の画素数は最大 115 なので uint8 に収まる。
"""
import os, sys, time, argparse, traceback
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitchbar_flash_probe as P

NROW, NCOL = 16, 192
DIV = 2

def masks(bgr):
    """白い閃光・金色・明るい の 3 種。白＝高輝度かつ低彩度（1段目で唯一、近い対照に差を残した条件）"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (S < 60) & (V >= 230)
    gold = (H >= 10) & (H <= 40) & (S >= 140) & (V >= 80)
    bright = (V >= 200)
    return white, gold, bright

def to_grid(m, redges, cedges):
    c = np.add.reduceat(m.astype(np.uint16), cedges, axis=1)
    return np.add.reduceat(c, redges, axis=0)

def scan_grid(video, hwaccel="", div=DIV, nrow=NROW, ncol=NCOL, prefix="grid"):
    t0 = time.time()
    path = os.path.join(P.VID_DIR, video + ".mp4")
    crop = P.pitchbar_roi(video)
    P.log(f"[{video}] {prefix}_scan 開始 ROI={crop} 升目{nrow}x{ncol} 縮小1/{div}")
    out = []
    redges = cedges = None
    for i, bgr in P.decode(path, 0.0, 0, crop, div=div, tag=prefix):
        if redges is None:
            h, w = bgr.shape[:2]
            redges = np.linspace(0, h, nrow + 1).astype(int)[:-1]
            cedges = np.linspace(0, w, ncol + 1).astype(int)[:-1]
            P.log(f"[{video}] 切り出し後の大きさ {w}x{h} 升目1つ {w/ncol:.1f}x{h/nrow:.1f}画素（1/{div}縮小）")
        g = np.stack([to_grid(m, redges, cedges) for m in masks(bgr)])
        out.append(np.minimum(g, 255).astype(np.uint8))
    pts = P.read_pts(P.decode.last_errpath)
    nn = min(len(out), len(pts))
    arr = np.stack(out[:nn])
    np.savez_compressed(os.path.join(P.OUT_DIR, f"{prefix}_{video}.npz"),
                        grid=arr, t=np.array(pts[:nn], np.float64),
                        crop=np.array(crop), div=div, nrow=nrow, ncol=ncol)
    P.log(f"[{video}] {prefix}_scan 完了 {nn}枚 {arr.nbytes/1e6:.0f}MB {time.time()-t0:.0f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--hwaccel", default="")
    ap.add_argument("--div", type=int, default=DIV)
    ap.add_argument("--nrow", type=int, default=NROW)
    ap.add_argument("--ncol", type=int, default=NCOL)
    ap.add_argument("--prefix", default="grid")
    a = ap.parse_args()
    for v in (a.videos or P.VIDEOS):
        try: scan_grid(v, a.hwaccel, a.div, a.nrow, a.ncol, a.prefix)
        except Exception as e: P.log(f"[{v}] {a.prefix}_scan 失敗 {e}\n{traceback.format_exc()}")
