#!/usr/bin/env python3
"""ケセラセラ kobushi=7 行修正: 帯は実デコード16177フレーム(f0..16176, t276.58)で終端。逐次読み m_kesera_tail_16160_end.png:
6 last f16173(t276.52), f16174-16175 fade transition, 7 visible at f16176(final frame). 当初のstable=16200は終端超え(seek clampの見誤り)→ stable=16176, event_frame_est=16174"""
import csv, json, os, shutil
d = os.path.expanduser("~/karaoke-vision/out_wao/ケセラセラ_20260902_123805"); fps = 58.487
p = os.path.join(d, "counter_events.csv"); shutil.copy(p, p + ".pre_fix7_20260903")
rows = list(csv.reader(open(p))); n = 0
for r in rows[1:]:
    if r[0] == "kobushi" and r[3] == "7":
        r[1], r[2], r[5], r[6] = "16176", f"{16176/fps:.3f}", "16174", f"{16174/fps:.3f}"; n += 1
csv.writer(open(p, "w", newline="")).writerows(rows)
j = os.path.join(d, "counter_events.json"); jj = json.load(open(j)); m = 0
for e in jj["kobushi"]:
    if e[1] == 7: e[0] = 16176; m += 1
json.dump(jj, open(j, "w"))
with open(os.path.join(d, "read_log.txt"), "a") as f:
    f.write("\n[manual-fix 2026-09-03] kobushi 7: corrected stable 16200->16176 (final decodable frame, t276.56), event_frame_est=16174 (6 last f16173). Evidence m_kesera_tail_16160_end.png. NOTE vibrato 24 (DAM) never displayed before recording end (video ends t276.6, playDate t283) = partial (recording cut).\n")
print("fixed", n, m)
