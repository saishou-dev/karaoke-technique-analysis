# -*- coding: utf-8 -*-
r"""手順3後半＋手順4＝加点イベントの時刻を方式どうしで突き合わせる（2026-10-22）。

時刻の引き方（方式ごとに「その方式が本来使う引き方」で引く）:
  A_passthrough … 帯動画自身の pts 表 pts_A[event_frame_est]（＝現行の G1 の video_time_sec と同じ引き方）
  B_cfr60       … event_frame_est / 60           （固定fpsなら「フレーム番号÷fps」で引けるのが固定化の利点）
  C_cfr5994     … event_frame_est * 1001 / 60000
B・C は自分の pts 表でも引いて、index 式と一致することを確認する（固定化されていれば一致する）。

突合の鍵＝(counter, count_value)。カウンターは減らない通し番号なので、同じカウンターの同じ値は同じ加点を指す。

出力＝report\events_long.csv（全方式の全イベント）／report\pairs_<X>.csv（Aとの対応・時刻差）／
      report\summary.csv・summary.md（方式ごとの表）／report\dup_runs.csv（重複フレームと安定ランへの影響）
使い方＝python compare_times_20261022.py
"""
import csv
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfr_common_20261022 import (G1_EXIST, OUT, PTS_EXIST, RESULTS_EXIST, SRC_DIR, VARIANTS, VIDEOS,
                                 ensure_out, log, pts_path, result_dir, video_pts)

ensure_out()
REP = os.path.join(OUT, "report")
LOG = os.path.join(REP, "compare_times.log")
BASE = "A_passthrough"


def load_pts(path):
    return [float(r["pts_sec"]) for r in csv.DictReader(open(path, encoding="utf-8"))]


def load_events(variant, video):
    f = os.path.join(result_dir(variant, video), "counter_events.csv")
    if not os.path.exists(f):
        return None
    out = []
    for r in csv.DictReader(open(f, encoding="utf-8")):
        out.append(dict(counter=r["counter"], frame=int(r["frame"]), value=int(r["value"]),
                        delta=int(r["delta"]), event_frame_est=int(r["event_frame_est"])))
    return out


def t_of(variant, pts, frame):
    """その方式が本来使う引き方で時刻を出す。"""
    mode = VARIANTS[variant]["time_mode"]
    if mode == "pts":
        return pts[min(frame, len(pts) - 1)]
    return frame / VARIANTS[variant]["rate"]


def read_log_stats(variant, video):
    f = os.path.join(result_dir(variant, video), "read_log.txt")
    s = dict(fps_cv2=None, frames=None, readable={}, runs={}, off_path={}, final={}, increments={}, band="")
    if not os.path.exists(f):
        return s
    for line in open(f, encoding="utf-8"):
        m = re.match(r"fps=([\d.]+) frames=(\d+)", line)
        if m:
            s["fps_cv2"] = float(m.group(1)); s["frames"] = int(m.group(2))
        m = re.match(r"(\w+): readable (\d+)/(\d+)", line)
        if m:
            s["readable"][m.group(1)] = int(m.group(2))
        m = re.match(r"(\w+): initial=(\S+) final=(\S+) increments=(\d+) runs=(\d+) off_path=(\d+)", line)
        if m:
            s["final"][m.group(1)] = m.group(3)
            s["increments"][m.group(1)] = int(m.group(4))
            s["runs"][m.group(1)] = int(m.group(5))
            s["off_path"][m.group(1)] = int(m.group(6))
        if line.startswith("band: "):
            s["band"] = line.strip()
    return s


# ---- 既存 G1（現行の正本）の時刻＝A の再現を検算するための基準 ----
g1 = {}
for r in csv.DictReader(open(G1_EXIST, encoding="utf-8-sig")):
    if r["video"] in VIDEOS:
        g1[(r["video"], r["counter"], int(r["count_value"]))] = float(r["video_time_sec"])

verify = {}
for r in csv.DictReader(open(os.path.join(os.path.dirname(G1_EXIST), "verify_20261014_wao.csv"), encoding="utf-8-sig")):
    if r["video"] in VIDEOS:
        verify[r["video"]] = r

# ---- 全イベントを長形式で ----
long_rows = []
ev = {}
pts_by = {}
for video in VIDEOS:
    for variant in VARIANTS:
        e = load_events(variant, video)
        if e is None:
            log("[warn] 読取結果が無い %s %s" % (variant, video), LOG)
            continue
        pts = load_pts(pts_path(variant, video))
        pts_by[(variant, video)] = pts
        ev[(variant, video)] = e
        for x in e:
            t_own = t_of(variant, pts, x["event_frame_est"])
            t_pts = pts[min(x["event_frame_est"], len(pts) - 1)]
            long_rows.append(dict(video=video, variant=variant, counter=x["counter"], count_value=x["value"],
                                  delta=x["delta"], frame=x["frame"], event_frame_est=x["event_frame_est"],
                                  time_own_method=round(t_own, 4), time_from_pts_table=round(t_pts, 4),
                                  time_own_minus_pts=round(t_own - t_pts, 4),
                                  g1_video_time_sec=g1.get((video, x["counter"], x["value"]), "")))

with open(os.path.join(REP, "events_long.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
    w.writeheader()
    for r in long_rows:
        w.writerow(r)

# ---- A の再現が既存 G1 と一致しているか（検算）----
rep_diff = []
for video in VIDEOS:
    for x in ev.get((BASE, video), []):
        k = (video, x["counter"], x["value"])
        if k in g1:
            t = t_of(BASE, pts_by[(BASE, video)], x["event_frame_est"])
            rep_diff.append(abs(t - g1[k]))
if rep_diff:
    log("[検算] A の再現 vs 既存 G1：対応 %d 件・時刻差 最大 %.4fs 中央 %.4fs"
        % (len(rep_diff), max(rep_diff), statistics.median(rep_diff)), LOG)

# ---- A と各方式の対応・時刻差 ----
summary = []
for variant in VARIANTS:
    if variant == BASE:
        continue
    pair_rows = []
    for video in VIDEOS:
        a = ev.get((BASE, video))
        b = ev.get((variant, video))
        if a is None or b is None:
            continue
        ka = {(x["counter"], x["value"]): x for x in a}
        kb = {(x["counter"], x["value"]): x for x in b}
        for k in sorted(set(ka) | set(kb)):
            xa, xb = ka.get(k), kb.get(k)
            ta = t_of(BASE, pts_by[(BASE, video)], xa["event_frame_est"]) if xa else None
            tb = t_of(variant, pts_by[(variant, video)], xb["event_frame_est"]) if xb else None
            pair_rows.append(dict(video=video, counter=k[0], count_value=k[1],
                                  in_A=int(xa is not None), in_X=int(xb is not None),
                                  frame_A=xa["event_frame_est"] if xa else "",
                                  frame_X=xb["event_frame_est"] if xb else "",
                                  time_A=round(ta, 4) if ta is not None else "",
                                  time_X=round(tb, 4) if tb is not None else "",
                                  diff_sec=(round(tb - ta, 4) if (ta is not None and tb is not None) else "")))
    if not pair_rows:
        log("[skip] %s の読取結果が無いので突合しない" % variant, LOG)
        continue
    with open(os.path.join(REP, "pairs_%s.csv" % variant), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)
    d = [abs(r["diff_sec"]) for r in pair_rows if r["diff_sec"] != ""]
    sd = [r["diff_sec"] for r in pair_rows if r["diff_sec"] != ""]
    both = len(d)
    only_a = sum(1 for r in pair_rows if r["in_A"] and not r["in_X"])
    only_x = sum(1 for r in pair_rows if r["in_X"] and not r["in_A"])
    summary.append(dict(
        variant=variant, label=VARIANTS[variant]["label"], matched=both, only_A=only_a, only_X=only_x,
        abs_median=round(statistics.median(d), 4) if d else "",
        abs_mean=round(statistics.fmean(d), 4) if d else "",
        abs_p95=round(sorted(d)[min(len(d) - 1, len(d) * 95 // 100)], 4) if d else "",
        abs_max=round(max(d), 4) if d else "",
        n_over_0_1s=sum(1 for x in d if x > 0.1), n_over_0_5s=sum(1 for x in d if x > 0.5),
        n_over_1s=sum(1 for x in d if x > 1.0),
        signed_median=round(statistics.median(sd), 4) if sd else "",
        signed_min=round(min(sd), 4) if sd else "", signed_max=round(max(sd), 4) if sd else "",
    ))
    log("[%s] 対応 %d 件（A のみ %d・%s のみ %d）｜|差| 中央 %.4fs 平均 %.4fs p95 %.4fs 最大 %.4fs｜>0.1s %d 件・>1s %d 件"
        % (variant, both, only_a, variant, only_x, summary[-1]["abs_median"], summary[-1]["abs_mean"],
           summary[-1]["abs_p95"], summary[-1]["abs_max"], summary[-1]["n_over_0_1s"], summary[-1]["n_over_1s"]), LOG)

with open(os.path.join(REP, "summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    for r in summary:
        w.writerow(r)

# ---- 重複フレーム・安定ランへの影響（6フレーム安定ルールの副作用）----
var_rows = {(r["variant"], r["video"]): r for r in csv.DictReader(open(os.path.join(REP, "variants.csv"), encoding="utf-8"))}
dupf = {}
_dp = os.path.join(REP, "dupframe.csv")
if os.path.exists(_dp):
    dupf = {(r["variant"], r["video"]): r for r in csv.DictReader(open(_dp, encoding="utf-8"))}
dup_rows = []
for video in VIDEOS:
    vr = verify.get(video, {})
    for variant in VARIANTS:
        if not os.path.exists(os.path.join(result_dir(variant, video), "counter_events.csv")):
            continue
        s = read_log_stats(variant, video)
        mv = var_rows.get((variant, video), {})
        n_band = int(mv.get("n_band") or 0)
        n_src = int(mv.get("n_src") or 0)
        row = dict(video=video, variant=variant, n_src=n_src, n_band=n_band, diff_frames=n_band - n_src,
                   dup=mv.get("dup", ""), drop=mv.get("drop", ""),
                   bytes=mv.get("bytes", ""), elapsed_sec=mv.get("elapsed_sec", ""),
                   fps_cv2=s["fps_cv2"], band_range=s["band"],
                   readable_total=sum(s["readable"].values()) if s["readable"] else "",
                   runs_total=sum(s["runs"].values()) if s["runs"] else "",
                   off_path_total=sum(s["off_path"].values()) if s["off_path"] else "",
                   increments_total=sum(s["increments"].values()) if s["increments"] else "",
                   unique_frames=dupf.get((variant, video), {}).get("unique_lt_0.05", ""),
                   same_as_prev=dupf.get((variant, video), {}).get("same_lt_0.05", ""))
        for c in ("shakuri", "kobushi", "fall", "vibrato", "accent", "hammering"):
            row["final_" + c] = s["final"].get(c, "")
            row["dam_" + c] = vr.get("csv_" + c, "")
        row["final_vs_dam"] = ",".join(
            "%s:%s!=%s" % (c, s["final"].get(c), vr.get("csv_" + c))
            for c in ("shakuri", "kobushi", "fall", "vibrato", "accent", "hammering")
            if vr and str(s["final"].get(c)) != str(vr.get("csv_" + c))) or "一致"
        dup_rows.append(row)
with open(os.path.join(REP, "dup_runs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(dup_rows[0].keys()))
    w.writeheader()
    for r in dup_rows:
        w.writerow(r)

# ---- まとめ（Markdown）----
md = []
md.append("# 固定フレームレート（CFR）変換との比較・2026-10-22")
md.append("")
md.append("対象＝10/14 収録の3曲（怪獣の花唄・RPG・共犯 10:24:28 テイク）。ffmpeg 9.0.1。読取器 v0.8e・ROI は曲ごとの較正値。")
md.append("")
md.append("## 方式ごとの時刻差（基準＝A 現行）")
md.append("")
md.append("| 方式 | 対応イベント | A のみ | その方式のみ | 時刻差 中央 | 平均 | p95 | 最大 | >0.1s | >1s |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for r in summary:
    md.append("| %s | %d | %d | %d | %.4f s | %.4f s | %.4f s | %.4f s | %d | %d |"
              % (r["label"], r["matched"], r["only_A"], r["only_X"], r["abs_median"], r["abs_mean"],
                 r["abs_p95"], r["abs_max"], r["n_over_0_1s"], r["n_over_1s"]))
md.append("")
md.append("## 変換そのものの数字")
md.append("")
md.append("| 曲 | 方式 | 元フレーム | 出力フレーム | 差 | 中身が違うフレーム | 直前と同じ | dup | drop | 容量 | 変換時間 | 曲末の読値 vs DAM |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in dup_rows:
    md.append("| %s | %s | %d | %d | %+d | %s | %s | %s | %s | %.0f MB | %s s | %s |"
              % (r["video"][:14], r["variant"], r["n_src"], r["n_band"], r["diff_frames"],
                 r["unique_frames"], r["same_as_prev"],
                 r["dup"], r["drop"], float(r["bytes"] or 0) / 1e6, r["elapsed_sec"], r["final_vs_dam"]))
md.append("")
md.append("## 6フレーム安定ルールへの影響（安定ラン数・採用外ラン数・読めたフレーム数）")
md.append("")
md.append("| 曲 | 方式 | 読めたフレーム(6カウンター合計) | 安定ラン | 採用外 | 増分合計 |")
md.append("|---|---|---|---|---|---|")
for r in dup_rows:
    md.append("| %s | %s | %s | %s | %s | %s |" % (r["video"][:14], r["variant"], r["readable_total"],
                                                   r["runs_total"], r["off_path_total"], r["increments_total"]))
open(os.path.join(REP, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
log("DONE summary.csv / summary.md / pairs_*.csv / events_long.csv / dup_runs.csv", LOG)
