#!/usr/bin/env python3
"""ダンスホール_20260902_113928: fall 1->17 (f11165, t187.7) は t188-195 の大型テロップ(白抜き「V」字グラフィック)がフォールROIを覆った際の
lowconf-rescue誤読(values=[17] 121 frames)。目視 m_dance_fall_184_208: t184-187=1, t188-195=テロップ, t196以降=1 に復帰。偽イベント除去。"""
import csv, json, os, shutil
d = os.path.expanduser("~/karaoke-vision/out_wao/ダンスホール_20260902_113928")
p = os.path.join(d, "counter_events.csv"); shutil.copy(p, p + ".pre_fix_20260903")
rows = list(csv.reader(open(p))); keep = [r for r in rows if not (r[0] == "fall" and r[3] == "17")]
print("removed", len(rows) - len(keep)); csv.writer(open(p, "w", newline="")).writerows(keep)
j = os.path.join(d, "counter_events.json"); shutil.copy(j, j + ".pre_fix_20260903")
jj = json.load(open(j)); b = len(jj["fall"]); jj["fall"] = [e for e in jj["fall"] if e[1] != 17]; print("json fall", b, "->", len(jj["fall"]))
json.dump(jj, open(j, "w"))
with open(os.path.join(d, "read_log.txt"), "a") as f:
    f.write("\n[manual-fix 2026-09-03] removed false event fall 1->17 f11165 (t187.7): telop graphic covering fall ROI t188-195 was lowconf-rescued as '17'. Visual m_dance_fall_184_208.png shows fall=1 before (t184-187) and after (t196-202). Final fall=1 = DAM CSV.\n")
print("ok")
