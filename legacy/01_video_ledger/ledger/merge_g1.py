#!/usr/bin/env python3
"""G1データセット統合: 各曲のカウンターイベント(録画時間軸) + sync_map -> 録音時間軸付き統合CSV"""
import csv
import os
import sys
import glob

BASE = sys.argv[1]  # 動画解析 dir
sync = {}
with open(os.path.join(BASE, "sync_work", "sync_map.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        sync[r["video"]] = r

out_rows = []
for d in sorted(glob.glob(os.path.join(BASE, "results", "*"))):
    video = os.path.basename(d)
    ev_path = os.path.join(d, "counter_events.csv")
    if not os.path.exists(ev_path):
        continue
    s = sync.get(video, {})
    take = s.get("take", "")
    off = s.get("offset_sec", "")
    with open(ev_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["delta"] == "0":
                continue  # 初期値行はイベントでない
            vt = float(r.get("event_time_est") or r["time_sec"])
            at = ""
            if take not in ("", "UNMATCHED") and off != "":
                at = f"{float(off) + vt:.3f}"
            out_rows.append({
                "session": "20260628",
                "video": video,
                "counter": r["counter"],
                "count_value": r["value"],
                "delta": r["delta"],
                "video_time_sec": f"{vt:.3f}",
                "stable_read_time_sec": r["time_sec"],
                "take": take if take != "UNMATCHED" else "",
                "audio_time_sec": at,
            })

out = os.path.join(BASE, "dataset")
os.makedirs(out, exist_ok=True)
path = os.path.join(out, "G1_events_20260628.csv")
with open(path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print(f"rows={len(out_rows)} -> {path}")
from collections import Counter
print(Counter(r["counter"] for r in out_rows))
print("with_audio:", sum(1 for r in out_rows if r["audio_time_sec"]))
