#!/usr/bin/env python3
"""lulu._20260902_111702 shakuri=11 行の修正: 読取器は t254.6(f15184)の「17」を「11」と誤読して安定化した(stable_read=254.620)。
目視 m_lulu_sh_152_154: 10 last f9094(t152.5), 11 from f9109(t152.75)。規約どおり event_frame_est=last_old+1=9095, stable=9109。"""
import csv, json, os, shutil
d = os.path.expanduser("~/karaoke-vision/out_wao/lulu._20260902_111702")
p = os.path.join(d, "counter_events.csv"); shutil.copy(p, p + ".pre_fix_20260903")
fps = 59.63
rows = list(csv.reader(open(p))); n = 0
for r in rows[1:]:
    if r[0] == "shakuri" and r[3] == "11" and r[1] == "15184":
        r[1], r[2], r[5], r[6] = "9109", f"{9109/fps:.3f}", "9095", f"{9095/fps:.3f}"; n += 1
csv.writer(open(p, "w", newline="")).writerows(rows)
j = os.path.join(d, "counter_events.json"); shutil.copy(j, j + ".pre_fix_20260903")
jj = json.load(open(j)); m = 0
for e in jj["shakuri"]:
    if e[1] == 11 and e[0] == 15184: e[0] = 9109; m += 1
json.dump(jj, open(j, "w"))
with open(os.path.join(d, "read_log.txt"), "a") as f:
    f.write("\n[manual-fix 2026-09-03] shakuri 11: stable frame 15184 (t254.6) was a misread of '17' -> corrected to stable=9109 (t152.75), event_frame_est=9095 (10 last f9094 t152.5). Evidence m_lulu_sh_152_154.png\n")
print("fixed csv rows", n, "json", m)
