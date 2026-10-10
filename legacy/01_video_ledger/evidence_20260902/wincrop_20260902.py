# -*- coding: utf-8 -*-
"""Windowsネイティブ帯クロップランナー v2 (傾き補正版): band_fix_final_20260902.csv の vf 列
(crop上下pad→rotate微小角→crop→scale=2480:120) をそのままffmpegに渡す。他の引数は crop_bands_20260818.sh と同一。
usage: python wincrop_20260902.py [workers]   (完成済み final は skip・.part→rename)"""
import csv, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

FF = r"C:\Users\conan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffmpeg.exe"
SRC = r"D:\研究アーカイブ\ビデオ素材\20260902"
OUT = r"D:\研究アーカイブ\成果物\動画解析\bands_wao"
SP = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(SP, "band_fix_final_20260902.csv")
LOG = os.path.join(SP, "wincrop_20260902.log")
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")

def crop(row):
    name, vf = row["video"], row["vf"]
    final = os.path.join(OUT, name + ".mp4")
    part = os.path.join(OUT, name + ".part.mp4")
    if os.path.exists(final):
        return f"skip {name}"
    log(f"CROP {name} vf={vf}")
    cmd = [FF, "-nostdin", "-v", "error", "-y", "-i", os.path.join(SRC, name + ".mp4"),
           "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an", part]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode == 0:
        os.replace(part, final)
        log(f"DONE {name}")
        return f"done {name}"
    log(f"FAIL {name} rc={p.returncode} {p.stderr.decode('utf-8', 'replace')[-300:]}")
    return f"fail {name}"

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
rows.sort(key=lambda r: 0 if r["video"] == "エルダーフラワー_20260902_121130" else 1)
log(f"start v2(roll-fix) workers={workers} rows={len(rows)}")
with ThreadPoolExecutor(max_workers=workers) as ex:
    for r in ex.map(crop, rows):
        pass
log("CROP_ALL_DONE")
