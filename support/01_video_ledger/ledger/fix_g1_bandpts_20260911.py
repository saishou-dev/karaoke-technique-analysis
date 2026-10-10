# -*- coding: utf-8 -*-
"""9/11 G1 の時刻を「帯クロップ動画のフレーム番号→帯クロップ動画自身のpts」で引き直す（2026-09-14）。
   背景: results_wao_20260911/<video>/counter_events.csv の frame・event_frame_est は bands_wao/<video>.mp4（wincrop_20260911.py で再エンコード）の
         フレーム番号。再エンコードで ffmpeg が同一時刻に丸まるフレームを落としたため（例 CHEERS 元11,477→帯11,413）、元動画の pts 表
         （dataset/pts/<video>.csv）を同じ番号で引くと曲の後半ほど早い時刻になる（検収クリップで上昇が0.5秒位置より0.7秒以上遅れて見えた）。
         帯クロップ動画は vfr で時刻を保っている（1/59秒に量子化）ので、帯動画の moov から pts を読めば正しい提示時刻になる。
   処理: ① bands_wao/<video>.mp4 → dataset/pts_bands_20260911/<video>.csv（index,pts_sec・scripts/mp4_pts.py の video_pts を流用）
         ② dataset/G1_events_20260911.csv を読み、video_time_sec / stable_read_time_sec / audio_time_sec を帯ptsで引き直した作業用CSV
            dataset/G1_events_20260911_bandpts_20260914.csv を書く（元の値は *_srcpts_sec 列に保存）。元CSV・G1_events_all は書き換えない。
   使い方: python scripts/fix_g1_bandpts_20260911.py"""
import os, sys, csv
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mp4_pts import video_pts
os.chdir(ROOT); OUT = "dataset/pts_bands_20260911"; os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open("dataset/G1_events_20260911.csv", encoding="utf-8-sig")))
vids = sorted(set(r["video"] for r in rows)); bpts = {}
print("video | src_frames | band_frames | diff")
for v in vids:
    p = os.path.join(OUT, v + ".csv")
    if not os.path.exists(p):
        pts, _ = video_pts(os.path.join("bands_wao", v + ".mp4"))
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["index", "pts_sec"]); [w.writerow([i, f"{x:.4f}"]) for i, x in enumerate(pts)]
    bpts[v] = [float(r["pts_sec"]) for r in csv.DictReader(open(p, encoding="utf-8"))]
    nsrc = sum(1 for _ in open(os.path.join("dataset", "pts", v + ".csv"), encoding="utf-8")) - 1
    print(f"{v[:40]:42s} {nsrc:6d} {len(bpts[v]):6d} {len(bpts[v]) - nsrc:+5d}")
fields = list(rows[0].keys()) + ["video_time_srcpts_sec", "stable_read_srcpts_sec", "audio_time_srcpts_sec"]
mx = 0.0; n_big = 0
for r in rows:
    p = bpts[r["video"]]; fi = min(int(r["event_frame_est"]), len(p) - 1); si = min(int(r["frame"]), len(p) - 1)
    t0 = float(r["video_time_sec"]); t1 = p[fi]; sh = t1 - t0; mx = max(mx, abs(sh)); n_big += abs(sh) > 0.5
    r["video_time_srcpts_sec"] = r["video_time_sec"]; r["stable_read_srcpts_sec"] = r["stable_read_time_sec"]; r["audio_time_srcpts_sec"] = r["audio_time_sec"]
    r["video_time_sec"] = f"{t1:.3f}"; r["stable_read_time_sec"] = f"{p[si]:.3f}"
    if r["audio_time_sec"]: r["audio_time_sec"] = f"{float(r['audio_time_sec']) + sh:.3f}"
out = "dataset/G1_events_20260911_bandpts_20260914.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"WROTE {out} rows={len(rows)} max|shift|={mx:.3f}s shift>0.5s={n_big}")
for r in rows:
    if r["video"].startswith("CHEERS") and r["counter"] == "shakuri" and r["count_value"] in ("10", "14"):
        print("check", r["video"], r["counter"], r["count_value"], "src", r["video_time_srcpts_sec"], "-> band", r["video_time_sec"])
