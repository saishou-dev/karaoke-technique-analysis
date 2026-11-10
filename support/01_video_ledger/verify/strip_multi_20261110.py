# -*- coding: utf-8 -*-
r"""複数の区間を1枚にまとめた帯状写真（逐次デコード1周・D-2 の 7(c) 用）。scripts は変更しない補助。2026-11-09＝found_20261021\strip_multi_20261022.py の 11/10 版（結果ディレクトリと出力先だけ変更）。

使い方: python strip_multi_20261110.py "<video>" <out_tag> <counter>:<a>:<b>:<step> [...] [--per-row=5] [--padx=6]
  --padx=N ... ROI の左右の余白(px)。2桁目が ROI で切れる曲で広げる（既定 6＝元と同じ）
出力: dataset/found_20261110/evidence/multi_<video>_<out_tag>.png
"""
import os, sys, json, cv2, numpy as np

ROOT = r"G:\研究アーカイブ\成果物\動画解析"
OUT = os.path.join(ROOT, "dataset", "found_20261110", "evidence")
os.makedirs(OUT, exist_ok=True)

name, tag = sys.argv[1], sys.argv[2]
per_row = 5
padx = 6
segs = []
for a in sys.argv[3:]:
    if a.startswith("--per-row="):
        per_row = int(a.split("=")[1]); continue
    if a.startswith("--padx="):
        padx = int(a.split("=")[1]); continue
    c, f0, f1, st = a.split(":")
    segs.append((c, int(f0), int(f1), int(st)))

roi = json.load(open(os.path.join(ROOT, "results_wao_20261110", name, "roi.json"), encoding="utf-8"))
need_hi = max(s[2] for s in segs)
cap = cv2.VideoCapture(os.path.join(ROOT, "bands_wao", name + ".mp4"))
buf = {i: [] for i in range(len(segs))}
f = 0
while f <= need_hi:
    ok, img = cap.read()
    if not ok:
        break
    for i, (c, f0, f1, st) in enumerate(segs):
        if f0 <= f <= f1 and (f - f0) % st == 0:
            x1, y1, x2, y2 = roi[c]
            t = img[max(0, y1 - 6):y2 + 6, max(0, x1 - padx):x2 + padx]
            t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            t = cv2.copyMakeBorder(t, 24, 0, 0, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cv2.putText(t, f"{c[:4]} f{f}", (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            buf[i].append(t)
    f += 1
cap.release()

blocks = []
for i, (c, f0, f1, st) in enumerate(segs):
    ts = buf[i]
    assert ts, f"{c} {f0}-{f1} でタイル0枚"
    h = max(t.shape[0] for t in ts); w = max(t.shape[1] for t in ts)
    ts = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in ts]
    rows = [np.hstack(ts[j:j + per_row]) for j in range(0, len(ts), per_row)]
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    b = cv2.copyMakeBorder(np.vstack(rows), 26, 6, 0, 0, cv2.BORDER_CONSTANT, value=(70, 70, 70))
    cv2.putText(b, f"[{i+1}] {c} f{f0}-{f1} step{st}", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    blocks.append(b)
W = max(b.shape[1] for b in blocks)
blocks = [cv2.copyMakeBorder(b, 0, 0, 0, W - b.shape[1], cv2.BORDER_CONSTANT) for b in blocks]
m = cv2.copyMakeBorder(np.vstack(blocks), 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
cv2.putText(m, f"{name[:46]} {tag}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
p = os.path.join(OUT, f"multi_{name}_{tag}.png")
cv2.imencode(".png", m)[1].tofile(p)
print("WROTE", p, m.shape)
