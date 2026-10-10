#!/usr/bin/env python3
"""Answer_20260902_121900: 帯消滅後(t>=246s・演奏終了12:23:11=t251)の誤読2行(fall 0->7 @f16649, kobushi 6->9 @f19453)を除去。
根拠: m_answer_fall_240_251.png / m_answer_kob_240_251.png(t=245.5まで fall=0, kobushi=6・t=246.0以降は帯なし)。
偽イベント0原則。counter_events.csv/.json から該当行を除去し read_log に注記。旧ファイルは .pre_fix_20260903 で保存"""
import csv, json, os, shutil
d = os.path.expanduser("~/karaoke-vision/out_wao/Answer_20260902_121900")
BAD = {("fall", "7", "16649"), ("kobushi", "9", "19453")}
p = os.path.join(d, "counter_events.csv"); shutil.copy(p, p + ".pre_fix_20260903")
rows = list(csv.reader(open(p)))
keep = [r for r in rows if not (r[0], r[3], r[1]) in BAD]
print("removed rows:", len(rows) - len(keep))
csv.writer(open(p, "w", newline="")).writerows(keep)
j = os.path.join(d, "counter_events.json"); shutil.copy(j, j + ".pre_fix_20260903")
jj = json.load(open(j))
for c in ("fall", "kobushi"):
    before = len(jj[c]); jj[c] = [e for e in jj[c] if not (c, str(e[1]), str(e[0])) in BAD]; print(c, before, "->", len(jj[c]))
json.dump(jj, open(j, "w"))
with open(os.path.join(d, "read_log.txt"), "a") as f:
    f.write("\n[manual-fix 2026-09-03] removed false events after band disappearance (song end t=246s, playDate t=251s): fall 0->7 f16649 (t279), kobushi 6->9 f19453 (t326). Evidence: m_answer_fall_240_251.png, m_answer_kob_240_251.png, answer_band_t246/t280.png (no counter band; MV only). Final fall=0 kobushi=6 = DAM CSV.\n")
print("ok")
