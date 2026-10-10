# -*- coding: utf-8 -*-
"""精緻化後の目視サンプルシート。evidence/refined/<video>/<tech>_<n>_T<T>.png（初出−6f｜初出｜v4出現｜v4+8f）を
技法別に等間隔で N 枚並べる → evidence/sheet_refined_<tech>.png、外れ値 → evidence/sheet_refined_outliers.png"""
import os, glob, cv2, numpy as np, pandas as pd
OUT_DIR = r"G:\研究アーカイブ\成果物\動画解析\marks_20260928"
EV = os.path.join(OUT_DIR, "evidence")
N = 12
def imread_u(p): return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
def imwrite_u(p, img): open(p, "wb").write(cv2.imencode(".png", img)[1].tobytes())
def sheet(rows, out, title):
    tiles = []
    for _, r in rows.iterrows():
        p = os.path.join(EV, "refined", r.video, f"{r.counter}_{int(r.count_value):03d}_T{r.video_time_sec:.3f}.png")
        if not os.path.exists(p): continue
        img = imread_u(p)
        bar = np.zeros((22, img.shape[1], 3), np.uint8)
        txt = f"{r.video[:16]} {r.counter}#{int(r.count_value)} T={r.video_time_sec:.2f} onset={r.mark_time_sec:.2f} (v4 {r.mark_time_raw:.2f}, back {int(r.back_frames)}f) lead={r.lead_sec:+.2f} selfncc={r.ncc_at_v4:.2f}"
        cv2.putText(bar, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(np.vstack([bar, img]))
    if not tiles: return 0
    W = max(t.shape[1] for t in tiles)
    sheet_ = np.vstack([np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in tiles])
    hdr = np.zeros((30, W, 3), np.uint8); cv2.putText(hdr, title, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    imwrite_u(out, np.vstack([hdr, sheet_])); return len(tiles)
df = pd.read_csv(os.path.join(OUT_DIR, "marks_20260924.csv"))
ok = df[df.mark_time_sec.notna() & df.back_frames.notna()]
for tech, g in ok.groupby("counter"):
    g = g.sort_values(["video", "video_time_sec"]); idx = np.linspace(0, len(g) - 1, min(N, len(g))).round().astype(int)
    print(tech, sheet(g.iloc[idx], os.path.join(EV, f"sheet_refined_{tech}.png"), f"{tech} refined: {len(idx)}/{len(g)}  tiles: onset-6f | ONSET(yellow box) | v4 onset | v4+8f"))
op = os.path.join(OUT_DIR, "outliers.csv")
if os.path.exists(op):
    o = pd.read_csv(op); o = o[o.back_frames.notna()] if "back_frames" in o else o.iloc[0:0]
    print("outliers", sheet(o.head(N), os.path.join(EV, "sheet_refined_outliers.png"), f"outliers after refinement: first {min(N, len(o))}/{len(o)}"))
