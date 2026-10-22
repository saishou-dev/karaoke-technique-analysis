# -*- coding: utf-8 -*-
r"""手順3の理論値＝フレーム番号→時刻の対応表だけで固定fps化の時刻差を予測する（2026-10-22）。

実測（read_variants→compare_times）は重いので、まず対応表だけで予測を出して実測と突き合わせる。

ffmpeg の fps 変換（-fps_mode cfr -r R）は、入力フレーム i（提示時刻 pts_i）を
出力の格子番号 s_i = round(pts_i * R) に置く。足りない格子は直前のフレームを複製し、
同じ格子に2枚以上来た入力は捨てる（＝重複フレームと落ちフレーム）。
よって固定化した映像で「フレーム番号÷R」で引いた時刻は round(pts_i*R)/R になり、
誤差は最大 ±1/(2R) 秒（R=60 なら ±8.33ms）に収まる、というのが理論値。

入力＝既存の帯pts表（dataset\pts_bands_20261014\<video>.csv・読むだけ）と
      既存の台帳 G1_events_20261014_wao.csv の event_frame_est（読むだけ）。
出力＝report\theory.csv（曲×レートごとの予測）／report\theory_events.csv（イベントごとの予測差）
使い方＝python theory_cfr_20261022.py
"""
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfr_common_20261022 import G1_EXIST, OUT, PTS_EXIST, VIDEOS, ensure_out, log

ensure_out()
REP = os.path.join(OUT, "report")
LOG = os.path.join(REP, "theory.log")
# D＝固定化するなら元録画の実測レートに合わせた場合（60 や 59.94 ではなく）。
# 実測＝フレーム間隔の 99.9% が 90000 分の 1516 か 1517 刻みで、その比が約 1:4＝平均 1516.8 刻み
# → 90000/1516.8 = 59.3354 fps（probe_src_20261022.py の tick 集計が根拠）。
RATES = {"B_cfr60": 60.0, "C_cfr5994": 60000.0 / 1001.0, "D_cfr_measured": 90000.0 / 1516.8}

g1 = {}
for r in csv.DictReader(open(G1_EXIST, encoding="utf-8-sig")):
    if r["video"] in VIDEOS:
        g1.setdefault(r["video"], []).append(
            (r["counter"], int(r["count_value"]), int(r["event_frame_est"]), float(r["video_time_sec"])))

rows = []
ev_rows = []
for video in VIDEOS:
    pts = [float(x["pts_sec"]) for x in csv.DictReader(
        open(os.path.join(PTS_EXIST, video + ".csv"), encoding="utf-8"))]
    n = len(pts)
    for name, R in RATES.items():
        slot = [round(p * R) for p in pts]
        # 同じ格子に2枚以上＝落ちフレーム／格子が飛ぶ＝その差だけ複製
        kept = len(set(slot))
        dropped = n - kept
        n_out = slot[-1] - slot[0] + 1
        dup = n_out - kept
        err = [slot[i] / R - pts[i] for i in range(n)]
        aerr = sorted(abs(e) for e in err)
        d = []
        for counter, cv, fr, vt in g1.get(video, []):
            i = min(fr, n - 1)
            t_cfr = slot[i] / R
            d.append(t_cfr - pts[i])
            ev_rows.append(dict(video=video, variant=name, counter=counter, count_value=cv,
                                event_frame_est=fr, pts_A=round(pts[i], 4), slot=slot[i],
                                time_cfr_pred=round(t_cfr, 4), diff_pred=round(t_cfr - pts[i], 4),
                                g1_video_time_sec=vt))
        ad = sorted(abs(x) for x in d)
        rows.append(dict(
            video=video, variant=name, rate=round(R, 4), n_src_band=n,
            n_out_pred=n_out, dup_pred=dup, dropped_pred=dropped,
            err_all_median=round(statistics.median(aerr), 5), err_all_max=round(aerr[-1], 5),
            n_events=len(d),
            ev_abs_median=round(statistics.median(ad), 5) if ad else "",
            ev_abs_mean=round(statistics.fmean(ad), 5) if ad else "",
            ev_abs_max=round(ad[-1], 5) if ad else "",
            ev_over_0_1s=sum(1 for x in ad if x > 0.1), ev_over_1s=sum(1 for x in ad if x > 1.0),
            bound_sec=round(1.0 / (2 * R), 5)))
        log("[理論] %-26s %-10s 予測 出力%d枚（複製%+d・落ち%d）｜全フレームの|誤差| 中央%.5fs 最大%.5fs"
            % (video[:26], name, n_out, dup, dropped, rows[-1]["err_all_median"], rows[-1]["err_all_max"]), LOG)
        log("        加点%d件の|予測差| 中央%.5fs 平均%.5fs 最大%.5fs（理論上限 %.5fs）｜>0.1s %d件・>1s %d件"
            % (len(d), rows[-1]["ev_abs_median"], rows[-1]["ev_abs_mean"], rows[-1]["ev_abs_max"],
               rows[-1]["bound_sec"], rows[-1]["ev_over_0_1s"], rows[-1]["ev_over_1s"]), LOG)

with open(os.path.join(REP, "theory.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
with open(os.path.join(REP, "theory_events.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(ev_rows[0].keys()))
    w.writeheader()
    for r in ev_rows:
        w.writerow(r)
log("DONE theory.csv / theory_events.csv", LOG)
