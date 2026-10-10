# -*- coding: utf-8 -*-
"""目視サンプル用のコンタクトシートを作る。
入力: evidence/samples/<video>/<tech>_<n>_T<T>_M<mark>.png（検出時に保存した 4 コマ帯：出現6f前／出現／8f後／20f後）
出力: evidence/sheet_samples_<tech>.png（技法別に等間隔サンプル N 枚）, evidence/sheet_outliers.png（外れ値）"""
import os, sys, glob, re, cv2, numpy as np, pandas as pd

OUT_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
EV = os.path.join(OUT_DIR, "evidence")
N_PER_SHEET = 12

def imread_u(p): return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
def imwrite_u(p, img): open(p, "wb").write(cv2.imencode(".png", img)[1].tobytes())

def label_tile(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img

def sheet_from(rows, out, title):
    tiles = []
    for _, r in rows.iterrows():
        pat = os.path.join(EV, "samples", r.video, f"{r.counter}_{int(r.count_value):03d}_T{r.video_time_sec:.3f}_M*.png")
        fs = glob.glob(pat)
        if not fs: continue
        img = imread_u(fs[0])
        txt = f"{r.video[:18]} {r.counter}#{int(r.count_value)} T={r.video_time_sec:.2f} mark={r.mark_time_sec:.2f} lead={r.lead_sec:+.2f} conf={r.conf:.2f}"
        tiles.append(label_tile(img, txt))
    if not tiles: return 0
    W = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in tiles]
    sheet = np.vstack(tiles)
    hdr = np.zeros((30, W, 3), np.uint8); cv2.putText(hdr, title, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    imwrite_u(out, np.vstack([hdr, sheet]))
    return len(tiles)

def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "marks_20260924.csv"))
    ok = df[df.mark_time_sec.notna()].copy()
    for tech, g in ok.groupby("counter"):
        g = g.sort_values(["video", "video_time_sec"])
        idx = np.linspace(0, len(g) - 1, min(N_PER_SHEET, len(g))).round().astype(int)
        n = sheet_from(g.iloc[idx], os.path.join(EV, f"sheet_samples_{tech}.png"),
                       f"{tech}: equally spaced samples {len(idx)}/{len(g)}  (tiles: -6f | onset | +8f | +20f, yellow box = onset frame)")
        print(tech, "sheet tiles", n)
    op = os.path.join(OUT_DIR, "outliers.csv")
    if os.path.exists(op):
        o = pd.read_csv(op)
        n = sheet_from(o.head(N_PER_SHEET), os.path.join(EV, "sheet_outliers.png"), f"outliers (|lead - median| > 0.6s or lead < 0): first {min(N_PER_SHEET, len(o))}/{len(o)}")
        print("outliers sheet tiles", n)

if __name__ == "__main__":
    main()
