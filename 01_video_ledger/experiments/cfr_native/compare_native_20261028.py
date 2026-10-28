# -*- coding: utf-8 -*-
r"""手順3＝候補ごとの指標を1枚の表にする（2026-10-28）。

指標（指示のとおり5つ）
  ① 曲末の読値と DAM の採点結果の一致（3曲×6カウンター＝18 枠）
  ② 現行 A との加点時刻の差＝中央値・1秒超の件数・最大
  ③ 加点の件数（増分の合計・単独増分の件数）
  ④ 処理時間（1,000 フレームあたり秒。1フレームごとの読み＋規則の後処理）
  ⑤ フレーム番号からの頭出しが当たるか（10/22 の seek_check.csv の実測を引き継ぐ）

突合の鍵＝(曲, カウンター, 通し番号の値)。カウンターは減らないので同じ値は同じ加点を指す。

出力＝cfr_native_20261028\report\ の summary.csv・summary.md・pairs_<cand>.csv・events_long.csv・final_vs_dam.csv
使い方＝python compare_native_20261028.py
"""
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_common_20261028 import (COUNTERS, G1_EXIST, OLD, OUT, SOURCES, VERIFY_EXIST, VIDEOS,
                                    ensure_out, log)
from rules_20261028 import CANDS

ensure_out()
REP = os.path.join(OUT, "report")
LOG = os.path.join(REP, "compare.log")
BASE = "A_現行"


def ev_path(cand, video):
    return os.path.join(OUT, "results", "%s__%s" % (cand, video), "counter_events.csv")


def load_events(path):
    if not os.path.exists(path):
        return None
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.append(dict(counter=r["counter"], frame=int(r["frame"]), value=int(r["value"]),
                        delta=int(r["delta"]), event_frame_est=int(r["event_frame_est"]),
                        t=float(r["event_time_est"])))
    return out


def load_stats(cand, video):
    p = os.path.join(OUT, "results", "%s__%s" % (cand, video), "stats.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


# ---- DAM の採点結果（正解）----
dam = {}
for r in csv.DictReader(open(VERIFY_EXIST, encoding="utf-8-sig")):
    if r["video"] in VIDEOS:
        dam[r["video"]] = {c: r.get("csv_" + c) for c in COUNTERS}

# ---- 検算1＝自分で読み直した A が 10/22 の A_passthrough と同じ加点を出すか ----
old_pts = {}
for video in VIDEOS:
    p = os.path.join(OLD, "pts", "A_passthrough__%s.csv" % video)
    old_pts[video] = [float(r["pts_sec"]) for r in csv.DictReader(open(p, encoding="utf-8"))]
chk = dict(n=0, same_frame=0, maxdt=0.0, only_new=0, only_old=0)
for video in VIDEOS:
    new = load_events(ev_path(BASE, video)) or []
    old = load_events(os.path.join(OLD, "results", "A_passthrough__%s" % video, "counter_events.csv")) or []
    kn = {(x["counter"], x["value"]): x for x in new}
    ko = {(x["counter"], x["value"]): x for x in old}
    for k in set(kn) | set(ko):
        if k in kn and k in ko:
            chk["n"] += 1
            if kn[k]["event_frame_est"] == ko[k]["event_frame_est"]:
                chk["same_frame"] += 1
            to = old_pts[video][min(ko[k]["event_frame_est"], len(old_pts[video]) - 1)]
            chk["maxdt"] = max(chk["maxdt"], abs(kn[k]["t"] - to))
        elif k in kn:
            chk["only_new"] += 1
        else:
            chk["only_old"] += 1
log("[検算1] 自分の A vs 10/22 の A：対応 %d 件・フレーム一致 %d 件・時刻差 最大 %.4f s・片方だけ 新%d／旧%d"
    % (chk["n"], chk["same_frame"], chk["maxdt"], chk["only_new"], chk["only_old"]), LOG)

# ---- 検算3＝足場の再現（P0 vs 10/22 の B2・P3b vs 10/22 の D2）----
for cand, oldv in (("P0_60_現行規則", "B2_fps60_filter"), ("P3b_5933_現行規則", "D2_fps5933_filter")):
    s = dict(n=0, same=0, only_new=0, only_old=0)
    for video in VIDEOS:
        new = load_events(ev_path(cand, video)) or []
        old = load_events(os.path.join(OLD, "results", "%s__%s" % (oldv, video), "counter_events.csv")) or []
        kn = {(x["counter"], x["value"]): x for x in new}
        ko = {(x["counter"], x["value"]): x for x in old}
        for k in set(kn) | set(ko):
            if k in kn and k in ko:
                s["n"] += 1
                s["same"] += int(kn[k]["event_frame_est"] == ko[k]["event_frame_est"])
            elif k in kn:
                s["only_new"] += 1
            else:
                s["only_old"] += 1
    log("[検算3] %s vs 10/22 の %s：対応 %d 件・フレーム一致 %d 件・片方だけ 新%d／旧%d"
        % (cand, oldv, s["n"], s["same"], s["only_new"], s["only_old"]), LOG)

# ---- 検算2＝既存台帳 G1 の時刻との差 ----
g1 = {}
for r in csv.DictReader(open(G1_EXIST, encoding="utf-8-sig")):
    if r["video"] in VIDEOS:
        g1[(r["video"], r["counter"], int(r["count_value"]))] = float(r["video_time_sec"])
dif = []
for video in VIDEOS:
    for x in load_events(ev_path(BASE, video)) or []:
        k = (video, x["counter"], x["value"])
        if k in g1:
            dif.append(abs(x["t"] - g1[k]))
if dif:
    log("[検算2] 自分の A vs 既存台帳 G1：対応 %d 件・時刻差 最大 %.4f s・中央 %.4f s"
        % (len(dif), max(dif), statistics.median(dif)), LOG)

# ---- 頭出しの実測（10/22 の計測を引き継ぐ）----
seek = {}
sp = os.path.join(OLD, "report", "seek_check.csv")
if os.path.exists(sp):
    for r in csv.DictReader(open(sp, encoding="utf-8")):
        seek[(r["variant"], r["video"])] = r
SEEK_MAP = {"vfr": "A_passthrough", "g60": "B2_fps60_filter", "g5933": "D2_fps5933_filter"}
# 今回書き出した帯動画の実測（seek_check_native_20261028.py）
seek_new = {}
spn = os.path.join(OUT, "report", "seek_check_native.csv")
if os.path.exists(spn):
    for r in csv.DictReader(open(spn, encoding="utf-8")):
        src = r["video"].split("__")[0]
        seek_new.setdefault(src, []).append(r)
SEEK_NEW_MAP = {"blend60file": "blend60", "gapblend60file": "gapblend60"}

# ---- 全イベント＋突合 ----
long_rows = []
summary = []
for cand, C in CANDS.items():
    rows_all = {}
    for video in VIDEOS:
        e = load_events(ev_path(cand, video))
        if e is not None:
            rows_all[video] = e
            for x in e:
                long_rows.append(dict(cand=cand, video=video, counter=x["counter"], count_value=x["value"],
                                      delta=x["delta"], frame=x["frame"],
                                      event_frame_est=x["event_frame_est"], time_sec=round(x["t"], 4)))
    if not rows_all:
        log("[skip] %s は結果が無い" % cand, LOG)
        continue
    # ① 曲末 vs DAM
    n_cells = n_hit = 0
    miss = []
    for video in VIDEOS:
        st = load_stats(cand, video)
        if not st or video not in dam:
            continue
        for c in COUNTERS:
            n_cells += 1
            got = st["stats"][c]["final"]
            want = dam[video][c]
            if str(got) == str(want):
                n_hit += 1
            else:
                miss.append("%s/%s %s→%s" % (video[:6], c, want, got))
    # ② A との時刻差
    pair_rows = []
    for video in VIDEOS:
        a = load_events(ev_path(BASE, video))
        b = rows_all.get(video)
        if a is None or b is None:
            continue
        ka = {(x["counter"], x["value"]): x for x in a}
        kb = {(x["counter"], x["value"]): x for x in b}
        for k in sorted(set(ka) | set(kb)):
            xa, xb = ka.get(k), kb.get(k)
            pair_rows.append(dict(video=video, counter=k[0], count_value=k[1],
                                  in_A=int(xa is not None), in_X=int(xb is not None),
                                  delta_A=xa["delta"] if xa else "", delta_X=xb["delta"] if xb else "",
                                  frame_A=xa["event_frame_est"] if xa else "",
                                  frame_X=xb["event_frame_est"] if xb else "",
                                  time_A=round(xa["t"], 4) if xa else "",
                                  time_X=round(xb["t"], 4) if xb else "",
                                  diff_sec=(round(xb["t"] - xa["t"], 4) if (xa and xb) else "")))
    with open(os.path.join(REP, "pairs_%s.csv" % cand), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)
    d = [abs(r["diff_sec"]) for r in pair_rows if r["diff_sec"] != ""]
    # ③ 件数 ④ 時間
    n_events = n_single = nframes = 0
    ocr = 0.0
    rule = 0.0
    for video in VIDEOS:
        st = load_stats(cand, video)
        if not st:
            continue
        n_events += sum(st["stats"][c]["increments"] for c in COUNTERS)
        n_single += sum(st["stats"][c]["n_single"] for c in COUNTERS)
        nframes += st["info"]["nframes"]
        ocr += float(st["info"]["ocr_sec"] or 0)
    for r in csv.DictReader(open(os.path.join(REP, "rules_meta.csv"), encoding="utf-8")):
        if r["cand"] == cand:
            rule += float(r["rule_sec"])
    # ⑤ 頭出し
    sv = SEEK_MAP.get(C["src"])
    snv = SEEK_NEW_MAP.get(C["src"])
    if snv and snv in seek_new:
        rs = seek_new[snv]
        seek_s = "%d/%d ぴったり・最大 %d フレーム外れ（実測）" % (
            sum(int(x["exact_hits"]) for x in rs), sum(int(x["n_probe"]) for x in rs),
            max(int(x["abs_max"]) for x in rs))
    elif sv:
        hits = sum(int(seek[(sv, v)]["exact_hits"]) for v in VIDEOS if (sv, v) in seek)
        smax = max([int(seek[(sv, v)]["abs_max"]) for v in VIDEOS if (sv, v) in seek] or [0])
        seek_s = "%d/60 ぴったり・最大 %d フレーム外れ" % (hits, smax)
    else:
        seek_s = "格子の定義どおり0フレーム（その場で作るためファイル無し・書き出した版の実測は 4f/7f の行）"
    summary.append(dict(
        cand=cand, label=C["label"], src=C["src"],
        dam_hit="%d/%d" % (n_hit, n_cells), dam_miss="／".join(miss) or "なし",
        matched=len(d), only_A=sum(1 for r in pair_rows if r["in_A"] and not r["in_X"]),
        only_X=sum(1 for r in pair_rows if r["in_X"] and not r["in_A"]),
        abs_median=round(statistics.median(d), 4) if d else "",
        abs_p95=round(sorted(d)[min(len(d) - 1, len(d) * 95 // 100)], 4) if d else "",
        abs_max=round(max(d), 4) if d else "",
        n_over_0_1s=sum(1 for x in d if x > 0.1), n_over_1s=sum(1 for x in d if x > 1.0),
        # 1秒超のうち「増分の刻み方が A と違う」もの＝同じ画面の加点を 4 と 3+1 のように違う刻みで
        # 拾ったために通し番号の対応がずれた分（時刻そのものの狂いではない）。残りが本当の時刻差。
        n_over_1s_valuealign=sum(1 for r in pair_rows if r["diff_sec"] != "" and abs(r["diff_sec"]) > 1.0
                                 and r["delta_A"] != r["delta_X"]),
        n_over_1s_pure=sum(1 for r in pair_rows if r["diff_sec"] != "" and abs(r["diff_sec"]) > 1.0
                           and r["delta_A"] == r["delta_X"]),
        n_events=n_events, n_single=n_single, nframes=nframes,
        sec_per_1k=round((ocr + rule) / max(1, nframes) * 1000, 2),
        ocr_sec=round(ocr, 1), rule_sec=round(rule, 2), seek=seek_s))
    log("[%s] DAM %s｜時刻差 中央 %s s・>1s %d 件・最大 %s s｜加点 %d（単独 %d）｜%s s/1000f"
        % (cand, summary[-1]["dam_hit"], summary[-1]["abs_median"], summary[-1]["n_over_1s"],
           summary[-1]["abs_max"], n_events, n_single, summary[-1]["sec_per_1k"]), LOG)

with open(os.path.join(REP, "events_long.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
    w.writeheader()
    for r in long_rows:
        w.writerow(r)
with open(os.path.join(REP, "summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    for r in summary:
        w.writerow(r)

# 曲末の値を1枠ずつ
with open(os.path.join(REP, "final_vs_dam.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cand", "video", "counter", "dam", "read", "match"])
    for cand in CANDS:
        for video in VIDEOS:
            st = load_stats(cand, video)
            if not st or video not in dam:
                continue
            for c in COUNTERS:
                got = st["stats"][c]["final"]
                want = dam[video][c]
                w.writerow([cand, video, c, want, got, int(str(got) == str(want))])

# ---- まとめ（Markdown）----
ORDER = ["A_現行", "A2_現行_時間安定", "P1_60_複製除去", "P2_60_時間安定", "P4_混ぜ60", "P3_5933_複製除去",
         "P7_抜けだけ中間", "P7t_抜けだけ中間_時間", "P8_混ぜ5933", "P4f_混ぜ60_ファイル", "P4ft_混ぜ60_ファイル_時間", "P7f_作り直し60_ファイル", "P7ft_作り直し60_ファイル_時間", "P9_格子番号化",
         "P5_60_除去＋時間", "P6_混ぜ60_時間", "P2b_60_時間101", "P2s_60_時間長重み",
         "P4s_混ぜ60_時間長重み", "P0_60_現行規則", "P3b_5933_現行規則"]
byc = {r["cand"]: r for r in summary}
md = []
md.append("# 60fps 専用の読み取り規則と現行方式の比較・2026-10-28")
md.append("")
md.append("対象＝10/14 収録の3曲（怪獣の花唄・RPG・共犯 10:24:28 テイク）。加点162件。")
md.append("読取（数字の切り出し・照合）は現行と同じ v0.8e。違うのは入力の作り方と「安定の数え方」だけ。")
md.append("")
md.append("| 候補 | ① 曲末 vs DAM | ② 時刻差 中央 | >1s | うち値のずれ由来 | 最大 | ③ 加点 | 単独 | ④ 秒/1000f | ⑤ 頭出し |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for c in ORDER:
    r = byc.get(c)
    if not r:
        continue
    md.append("| %s | %s | %s s | %d | %d | %s s | %d | %d | %s | %s |"
              % (r["label"], r["dam_hit"], r["abs_median"], r["n_over_1s"], r["n_over_1s_valuealign"],
                 r["abs_max"], r["n_events"], r["n_single"], r["sec_per_1k"], r["seek"]))
md.append("")
md.append("## 曲末の値が DAM と違った枠")
md.append("")
md.append("| 候補 | 違った枠（正解→読値） |")
md.append("|---|---|")
for c in ORDER:
    r = byc.get(c)
    if r:
        md.append("| %s | %s |" % (c, r["dam_miss"]))
md.append("")
md.append("## 複製と見なす画素差のしきい値を振った結果（入力＝固定60fps）")
md.append("")
sw = os.path.join(REP, "dedup_sweep.csv")
if os.path.exists(sw):
    md.append("| 曲 | しきい値 | 外したフレーム | 安定ラン | 加点 | しゃくり | こぶし | フォール | ビブラート | アクセント | ハンマリング |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in csv.DictReader(open(sw, encoding="utf-8")):
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                  % (r["video"][:14], r["theta"], r["removed"], r["runs"], r["n_events"],
                     r["final_shakuri"], r["final_kobushi"], r["final_fall"], r["final_vibrato"],
                     r["final_accent"], r["final_hammering"]))
md.append("")
md.append("## 入力ごとの数字")
md.append("")
md.append("| 候補 | 入力 | レート | フレーム数 | 外したフレーム | 1フレーム読み(秒) | 後処理(秒) |")
md.append("|---|---|---|---|---|---|---|")
for r in csv.DictReader(open(os.path.join(REP, "rules_meta.csv"), encoding="utf-8")):
    md.append("| %s | %s | %s | %s | %s | %s | %s |"
              % (r["cand"], r["video"][:14], r["rate"], r["nframes"], r["removed"], r["ocr_sec"], r["rule_sec"]))
open(os.path.join(REP, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
log("DONE summary.csv / summary.md / pairs_*.csv / events_long.csv / final_vs_dam.csv", LOG)
