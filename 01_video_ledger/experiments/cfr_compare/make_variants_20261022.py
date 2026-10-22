# -*- coding: utf-8 -*-
r"""手順2＝3方式で帯動画を作り、変換の数字を記録する（2026-10-22）。

帯クロップのフィルタ・コーデック・画質は wincrop_20260915.py と同一にして、フレームレートの扱いだけを変える。
  A_passthrough  -fps_mode passthrough -enc_time_base 1/90000 -video_track_timescale 90000（現行）
  B_cfr60        -fps_mode cfr -r 60
  C_cfr5994      -fps_mode cfr -r 60000/1001

記録する数字＝出力フレーム数・ffmpeg の dup／drop・元録画とのフレーム数差・最終 pts の差・ファイルサイズ・変換時間。
各変換の pts 表（index,pts_sec）も OUT\pts\ に作る（A は帯pts表と同じ作り方＝moov の stts/ctts から）。

出力＝dataset\cfr_compare_2026-10-22\bands\<variant>__<video>.mp4 ＋ pts\<variant>__<video>.csv ＋ report\variants.csv
使い方＝python make_variants_20261022.py [--only A_passthrough,B_cfr60] [--workers 3]
"""
import argparse
import concurrent.futures as cf
import csv
import math
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfr_common_20261022 import (OUT, SRC_DIR, FFMPEG, FFPROBE, PROBE_CSV, VARIANTS, VIDEOS,
                                 band_path, ensure_out, log, pts_path, video_pts)

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--workers", type=int, default=3)
A = ap.parse_args()

ensure_out()
REP = os.path.join(OUT, "report")
LOG = os.path.join(REP, "make_variants.log")
WANT = [k for k in VARIANTS if not A.only or k in A.only.split(",")]

probe = {r["video"]: r for r in csv.DictReader(open(PROBE_CSV, encoding="utf-8-sig"))}


def nb_frames(path):
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        return int(p.stdout.strip())
    except Exception:
        return None


def write_pts_csv(path, pts):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "pts_sec"])
        for i, p in enumerate(pts):
            w.writerow([i, "%.4f" % p])


def one(job):
    variant, video = job
    out = band_path(variant, video)
    src = os.path.join(SRC_DIR, video + ".mp4")
    r = probe[video]
    xL, yL, w, h = int(r["xL"]), int(r["yL"]), int(r["w"]), int(r["h"])
    ang = float(r["angle_deg"])
    mx, my = 20, 48
    vf = ("crop=%d:%d:%d:%d,rotate=%.6f:c=black,crop=%d:%d:%d:%d,scale=2480:120"
          % (w + 2 * mx, h + 2 * my, xL - mx, yL - my, -math.radians(ang), w, h, mx, my)
          + VARIANTS[variant].get("vf_suffix", ""))
    cmd = ([FFMPEG, "-y", "-v", "warning", "-stats", "-threads", "4", "-i", src, "-vf", vf]
           + VARIANTS[variant]["args"]
           + ["-c:v", "libx264", "-crf", "15", "-preset", "fast", "-an", out + ".part.mp4"])
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    if p.returncode != 0:
        try:
            os.remove(out + ".part.mp4")
        except OSError:
            pass
        return dict(variant=variant, video=video, status="ERR",
                    reason="ffmpeg rc=%d %s" % (p.returncode, (p.stderr or "")[-300:].replace("\n", " ")))
    os.replace(out + ".part.mp4", out)
    se = p.stderr or ""
    dup = drop = None
    m = re.findall(r"dup=\s*(\d+)", se)
    if m:
        dup = int(m[-1])
    m = re.findall(r"drop=\s*(\d+)", se)
    if m:
        drop = int(m[-1])
    bp, _ = video_pts(out)
    sp, _ = video_pts(src)
    write_pts_csv(pts_path(variant, video), bp)
    nbf = nb_frames(out)
    bd = [bp[i + 1] - bp[i] for i in range(len(bp) - 1)]
    bds = sorted(bd)
    return dict(
        variant=variant, video=video, status="OK", reason="",
        ffmpeg_args=" ".join(VARIANTS[variant]["args"]),
        elapsed_sec=round(dt, 1), bytes=os.path.getsize(out),
        n_src=len(sp), n_band=len(bp), nb_read_frames=nbf,
        diff_frames=len(bp) - len(sp), dup=dup, drop=drop,
        last_pts_band=round(bp[-1], 4), last_pts_src=round(sp[-1], 4),
        last_pts_diff=round(bp[-1] - sp[-1], 4),
        band_d_min=round(bds[0], 6), band_d_median=round(bds[len(bds) // 2], 6), band_d_max=round(bds[-1], 6),
        band_d_unique=len(set(round(x, 6) for x in bds)),
        band_is_vfr=("YES" if len(set(round(x, 6) for x in bds)) > 1 else "NO"),
    )


jobs = [(v, s) for v in WANT for s in VIDEOS if not os.path.exists(band_path(v, s))]
skipped = [(v, s) for v in WANT for s in VIDEOS if os.path.exists(band_path(v, s))]
for v, s in skipped:
    log("[skip] 既にある %s__%s" % (v, s), LOG)
log("[make] jobs=%d workers=%d" % (len(jobs), A.workers), LOG)

rows = []
if jobs:
    with cf.ThreadPoolExecutor(max_workers=A.workers) as ex:
        for res in ex.map(one, jobs):
            rows.append(res)
            if res["status"] == "OK":
                log("[ok] %-14s %-24s %5.0fs %6.1fMB frames=%d(src %d diff%+d) dup=%s drop=%s last_pts差=%+.4fs 可変=%s"
                    % (res["variant"], res["video"][:24], res["elapsed_sec"], res["bytes"] / 1e6,
                       res["n_band"], res["n_src"], res["diff_frames"], res["dup"], res["drop"],
                       res["last_pts_diff"], res["band_is_vfr"]), LOG)
            else:
                log("[ERR] %s %s %s" % (res["variant"], res["video"], res["reason"]), LOG)

# 既にあった分も表に載せ直す（再変換はしない＝時間は空欄）
for v, s in skipped:
    bp, _ = video_pts(band_path(v, s))
    sp, _ = video_pts(os.path.join(SRC_DIR, s + ".mp4"))
    if not os.path.exists(pts_path(v, s)):
        write_pts_csv(pts_path(v, s), bp)
    bd = sorted(bp[i + 1] - bp[i] for i in range(len(bp) - 1))
    rows.append(dict(variant=v, video=s, status="OK(existing)", reason="", ffmpeg_args=" ".join(VARIANTS[v]["args"]),
                     elapsed_sec="", bytes=os.path.getsize(band_path(v, s)), n_src=len(sp), n_band=len(bp),
                     nb_read_frames="", diff_frames=len(bp) - len(sp), dup="", drop="",
                     last_pts_band=round(bp[-1], 4), last_pts_src=round(sp[-1], 4),
                     last_pts_diff=round(bp[-1] - sp[-1], 4), band_d_min=round(bd[0], 6),
                     band_d_median=round(bd[len(bd) // 2], 6), band_d_max=round(bd[-1], 6),
                     band_d_unique=len(set(round(x, 6) for x in bd)),
                     band_is_vfr=("YES" if len(set(round(x, 6) for x in bd)) > 1 else "NO")))

cols = ["variant", "video", "status", "reason", "ffmpeg_args", "elapsed_sec", "bytes", "n_src", "n_band",
        "nb_read_frames", "diff_frames", "dup", "drop", "last_pts_band", "last_pts_src", "last_pts_diff",
        "band_d_min", "band_d_median", "band_d_max", "band_d_unique", "band_is_vfr"]
path = os.path.join(REP, "variants.csv")
old = []
if os.path.exists(path):
    old = [x for x in csv.DictReader(open(path, encoding="utf-8"))
           if (x["variant"], x["video"]) not in {(r["variant"], r["video"]) for r in rows}]
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in old + rows:
        w.writerow({k: r.get(k, "") for k in cols})
log("DONE variants.csv rows=%d" % (len(old) + len(rows)), LOG)
