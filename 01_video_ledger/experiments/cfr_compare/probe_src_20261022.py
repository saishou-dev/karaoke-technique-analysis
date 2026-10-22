# -*- coding: utf-8 -*-
r"""手順1＝元録画の実態を測る（2026-10-22）。

「スマホの録画は本当に可変フレームレートなのか・どれだけ揺れているのか」を数字で出す。
moov の stts/ctts から全フレームの提示時刻（pts）を読み（mp4_pts.video_pts・デコードなし）、
フレーム間隔の最小／中央／最大・平均fps・1秒を超える空白の有無を集計する。

出力＝dataset\cfr_compare_2026-10-22\report\src_probe.csv ＋ 同 _intervals_<video>.csv（間隔のヒストグラム）
使い方＝python probe_src_20261022.py
"""
import csv
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfr_common_20261022 import (OUT, SRC_DIR, FFPROBE, VIDEOS, ensure_out, log, video_pts)

ensure_out()
REP = os.path.join(OUT, "report")
LOG = os.path.join(REP, "probe_src.log")


def ffprobe_stream(path):
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,time_base,duration",
                        "-of", "json", path], capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        return json.loads(p.stdout)["streams"][0]
    except Exception:
        return {}


def frac(s):
    try:
        a, b = s.split("/")
        return float(a) / float(b)
    except Exception:
        try:
            return float(s)
        except Exception:
            return float("nan")


rows = []
for v in VIDEOS:
    src = os.path.join(SRC_DIR, v + ".mp4")
    st = ffprobe_stream(src)
    pts, timescale = video_pts(src)
    n = len(pts)
    d = [pts[i + 1] - pts[i] for i in range(n - 1)]
    ds = sorted(d)
    span = pts[-1] - pts[0]
    avg_fps = (n - 1) / span if span > 0 else float("nan")
    # 1秒超の空白＝連続フレームの間隔が 1.0 秒を超えた箇所
    gaps = [(i, d[i]) for i in range(len(d)) if d[i] > 1.0]
    # 「公称60fpsの格子」からのずれ＝index/60 と実pts の差（第1種の時刻ずれの大きさ）
    drift60 = [pts[i] - i / 60.0 for i in range(n)]
    drift5994 = [pts[i] - i * 1001.0 / 60000.0 for i in range(n)]
    # 落ちフレームの推定＝間隔が公称1フレーム(1/60)の1.5倍を超えた回数と、そのとき失われたフレーム数
    nominal = 1.0 / 60.0
    missing = sum(int(round(x / nominal)) - 1 for x in d if x > nominal * 1.5)
    r = dict(
        video=v, bytes=os.path.getsize(src), codec=st.get("codec_name", ""),
        size="%sx%s" % (st.get("width"), st.get("height")),
        timescale=timescale, n_frames=n, nb_frames_ffprobe=st.get("nb_frames"),
        r_frame_rate=st.get("r_frame_rate"), avg_frame_rate_ffprobe=st.get("avg_frame_rate"),
        avg_frame_rate_num=round(frac(st.get("avg_frame_rate", "nan")), 4),
        duration_pts=round(pts[-1], 4), avg_fps_from_pts=round(avg_fps, 4),
        d_min=round(ds[0], 6), d_p1=round(ds[max(0, len(ds) // 100)], 6),
        d_median=round(statistics.median(ds), 6), d_p99=round(ds[min(len(ds) - 1, len(ds) * 99 // 100)], 6),
        d_max=round(ds[-1], 6), d_unique=len(set(round(x, 6) for x in ds)),
        n_gap_over_1s=len(gaps), gap_list=";".join("%d:%.3f" % g for g in gaps[:10]),
        est_missing_frames=missing,
        drift60_end=round(drift60[-1], 4), drift60_max_abs=round(max(abs(x) for x in drift60), 4),
        drift5994_end=round(drift5994[-1], 4), drift5994_max_abs=round(max(abs(x) for x in drift5994), 4),
        is_vfr=("YES" if len(set(round(x, 6) for x in ds)) > 1 else "NO"),
    )
    rows.append(r)
    log("[src] %s" % v, LOG)
    log("      %s %s ts=%d frames=%d dur=%.3fs avg_fps(pts)=%.4f r_frame_rate=%s avg(ffprobe)=%s"
        % (r["codec"], r["size"], timescale, n, pts[-1], avg_fps, r["r_frame_rate"], r["avg_frame_rate_ffprobe"]), LOG)
    log("      間隔 min=%.6f p1=%.6f 中央=%.6f p99=%.6f max=%.6f 種類=%d 可変=%s"
        % (r["d_min"], r["d_p1"], r["d_median"], r["d_p99"], r["d_max"], r["d_unique"], r["is_vfr"]), LOG)
    log("      1秒超の空白=%d件 %s / 落ちフレーム推定=%d枚" % (r["n_gap_over_1s"], r["gap_list"], missing), LOG)
    log("      index/60 との差 末尾=%+.4fs 最大絶対=%.4fs ／ index/59.94 との差 末尾=%+.4fs 最大絶対=%.4fs"
        % (r["drift60_end"], r["drift60_max_abs"], r["drift5994_end"], r["drift5994_max_abs"]), LOG)
    # 間隔の分布（ミリ秒を0.5ms刻みで丸めた度数）
    hist = {}
    for x in d:
        k = round(x * 1000 * 2) / 2.0
        hist[k] = hist.get(k, 0) + 1
    with open(os.path.join(REP, "src_intervals_%s.csv" % v.replace("/", "_")), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["interval_ms", "count"])
        for k in sorted(hist):
            w.writerow([k, hist[k]])

with open(os.path.join(REP, "src_probe.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
log("DONE src_probe.csv", LOG)
