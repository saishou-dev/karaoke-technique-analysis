# -*- coding: utf-8 -*-
"""検収クリップ v2 (gen_clips_v2_20260817.py の Windowsネイティブ移植・引数/式/ffmpeg引数は同一・D27)
窓 [T-1.0, max(S-T,0.2)cap12 + 0.8]・640x360・libx264 crf24・無音・eid={video}_{counter}_{value}
usage: python winclips_20260902.py <G1_csv> <out_dir> [max_count]   既存(>=20KB)はスキップ=再開可能
"""
import csv, os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor

FFMPEG = r"C:\Users\conan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffmpeg.exe"
PRE = 1.0
g1, out_dir = sys.argv[1], sys.argv[2]
max_count = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winclips_20260902.log")

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")

def src(video, session):
    assert session == "20260902", session
    return rf"D:\研究アーカイブ\ビデオ素材\{session}\{video}.mp4"

rows = list(csv.DictReader(open(g1, encoding="utf-8-sig")))
todo = []
for r in rows:
    eid = f'{r["video"]}_{r["counter"]}_{r["count_value"]}'
    p = os.path.join(out_dir, eid + ".mp4")
    if not (os.path.exists(p) and os.path.getsize(p) >= 20 * 1024):
        todo.append((eid, r))
total_missing = len(todo)
todo = todo[:max_count]
log(f"missing={total_missing} this_chunk={len(todo)}")

def dur_of(r):
    T = float(r["video_time_sec"]); S = float(r["stable_read_time_sec"])
    lag = min(max(S - T, 0.2), 12.0)
    return PRE + lag + 0.8

def make(job):
    eid, r = job
    T = float(r["video_time_sec"])
    t0 = max(0.0, T - PRE)
    out = os.path.join(out_dir, eid + ".mp4")
    cmd = [FFMPEG, "-nostdin", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-i", src(r["video"], r["session"]),
           "-t", f"{dur_of(r):.2f}", "-vf", "scale=640:360", "-c:v", "libx264", "-crf", "24",
           "-preset", "veryfast", "-an", "-movflags", "+faststart", out]
    p = subprocess.run(cmd, capture_output=True)
    return eid, p.returncode

ok = ng = 0
with ThreadPoolExecutor(max_workers=10) as ex:
    for eid, rc in ex.map(make, todo):
        if rc == 0:
            ok += 1
        else:
            ng += 1
            log(f"FAIL {eid}")
        if (ok + ng) % 100 == 0:
            log(f"progress {ok+ng}/{len(todo)}")
log(f"chunk done ok={ok} ng={ng} remaining={total_missing - len(todo)}")
log("CLIPS_DONE")
