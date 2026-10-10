#!/usr/bin/env python3
"""9/2欠落増分の調査ドライバ(選択版・2026-09-03): 引数で指定した動画のみ、verify_20260902.csvの不足カウンターを
最終確定読取フレームから終盤走査(rescan_tail)。ログ= ~/karaoke-vision/rescan_20260902/<video>.log"""
import csv, os, subprocess, sys
OUT = os.path.expanduser("~/karaoke-vision/out_wao")
BANDS = "/mnt/d/研究アーカイブ/成果物/動画解析/bands_wao"
LOGD = os.path.expanduser("~/karaoke-vision/rescan_20260902")
os.makedirs(LOGD, exist_ok=True)
COUNTERS = ["shakuri", "kobushi", "fall", "vibrato", "accent", "hammering"]
ONLY = set(sys.argv[1:])
rows = {r["video"]: r for r in csv.DictReader(open(os.path.join(OUT, "verify_20260902.csv"), encoding="utf-8-sig"))}
for video in sys.argv[1:]:
    r = rows[video]
    deficit = [k for k in COUNTERS if int(r[f"diff_{k}"]) < 0]
    if not deficit:
        print("no deficit", video); continue
    evp = os.path.join(OUT, video, "counter_events.csv")
    last_t = {}
    for e in csv.DictReader(open(evp)):
        if e["counter"] in deficit:
            last_t[e["counter"]] = float(e["time_sec"])
    t0 = min(last_t.get(c, 0.0) for c in deficit)
    f0 = max(0, int(t0 * 59.94) - 60)
    log = os.path.join(LOGD, video + ".log")
    print(f"=== {video} deficit={deficit} t0={t0:.1f}s f0={f0}", flush=True)
    with open(log, "w") as lf:
        subprocess.run([sys.executable, os.path.expanduser("~/karaoke-vision/rescan_tail.py"),
                        os.path.join(BANDS, video + ".mp4"), str(f0), ",".join(deficit)], stdout=lf, stderr=subprocess.STDOUT)
    print(open(log).read()[-600:], flush=True)
print("ALL_RESCANS_DONE")
