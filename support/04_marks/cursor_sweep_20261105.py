# -*- coding: utf-8 -*-
"""案D（mark-pos-time）手順3＝再生カーソル x の曲中の動きを見る・2026-11-05 Opus
detect_marks.py と同じ ROI・同じ座標系（4K 原画素・原点＝4K 画面の左上）で、step 秒ごとに1枚復号し、
detect_marks.find_cursor のカーソル x（4K 座標）を CSV に、縮小した帯を縦に8枚ずつ並べた PNG を出す。
既存スクリプトは import して関数を使うだけ（本体は無変更）。
使い方: python cursor_sweep_20261105.py SESSION VIDEO T0 T1 STEP OUTDIR
"""
import sys, os, subprocess, re, csv
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_marks as dm

ses, video, t0, t1, step, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
dm.configure(ses); crop = dm.roi_for(video, pd.read_csv(dm.BAND_CSV)); cx, cy, cw, ch = crop
cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-ss", f"{t0}", "-t", f"{t1 - t0}", "-copyts",
       "-i", os.path.join(dm.VID_DIR, video + ".mp4"), "-an",
       "-vf", f"fps={1 / step},crop={cw}:{ch}:{cx}:{cy},showinfo", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
fs = cw * ch * 3; n = len(p.stdout) // fs
pts = [float(x) for x in re.findall(rb"pts_time:([0-9.]+)", p.stderr)]
os.makedirs(out, exist_ok=True)
tag = f"{ses}_{re.sub(r'[^0-9A-Za-z]+', '_', video)}_{t0:g}-{t1:g}"
tiles = []
with open(os.path.join(out, f"cursor_{tag}.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["session", "video", "pts_sec", "cursor_x_4k", "roi_x0", "roi_y0"])
    for i in range(n):
        f = np.frombuffer(p.stdout[i * fs:(i + 1) * fs], np.uint8).reshape(ch, cw, 3)
        c = dm.find_cursor(dm.classify(f))
        t = pts[i] if i < len(pts) else ""
        w.writerow([ses, video, t, "" if c is None else c + cx, cx, cy])
        print(f"{i} pts={t} cursor_x4k={None if c is None else c + cx}")
        tiles.append(cv2.resize(f, None, fx=0.25, fy=0.25))
for k in range(0, n, 8):
    cv2.imencode(".png", np.vstack(tiles[k:k + 8]))[1].tofile(os.path.join(out, f"sheet_{tag}_{k:03d}.png"))   # 日本語パス対策
