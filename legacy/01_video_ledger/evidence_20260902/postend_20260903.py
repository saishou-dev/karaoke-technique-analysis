#!/usr/bin/env python3
"""演奏終了(DAM playDate)後の増分イベント検出: video開始時刻(ファイル名)とplayDateから終了t_endを求め、
event_time_est > t_end-3 の増分を列挙(偽イベント候補)。"""
import csv, os, glob, json
from datetime import datetime
OUT = os.path.expanduser("~/karaoke-vision/out_wao")
dam = [r for r in csv.DictReader(open("/home/toshiaki/karaoke-api/output/dam_ai-hearts_full_2026-09-02T05-29-56.csv", encoding="utf-8-sig")) if r["playDate"].startswith("2026-09-02")]
for d in sorted(glob.glob(os.path.join(OUT, "*_20260902_*"))):
    v = os.path.basename(d); p = os.path.join(d, "counter_events.csv")
    if not os.path.exists(p): continue
    vs = datetime.strptime("2026-09-02 " + v.rsplit("_", 1)[1], "%Y-%m-%d %H%M%S")
    c = [r for r in dam if 0 < (datetime.strptime(r["playDate"], "%Y-%m-%d %H:%M:%S") - vs).total_seconds() < 480]
    if not c: print(v, "NO DAM"); continue
    b = min(c, key=lambda r: datetime.strptime(r["playDate"], "%Y-%m-%d %H:%M:%S"))
    t_end = (datetime.strptime(b["playDate"], "%Y-%m-%d %H:%M:%S") - vs).total_seconds()
    ev = [r for r in csv.DictReader(open(p)) if r["delta"] != "0"]
    last = max(float(r["event_time_est"]) for r in ev) if ev else 0
    late = [(r["counter"], r["value"], r["delta"], r["event_time_est"]) for r in ev if float(r["event_time_est"]) > t_end - 3]
    print(f"{v}: t_end={t_end:.0f}s last_ev={last:.1f}s n_ev={len(ev)} late={late if late else '-'}")
