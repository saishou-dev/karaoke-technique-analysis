#!/usr/bin/env python3
"""grade_v08f_all.py ＝ 読取器 v0.8f（見本 k＝3＋規則(c)）で読み直した全収録の npz から加点列を作り、
公式採点データ・現行 G1 と突き合わせる（2026-10-31）

入力（読むだけ）
  走査＝reread_v08f_20261031/fullscan/<収録>/<曲>.npz（v3_fullscan_all.py の出力）
  公式採点データ＝verify_<収録>.csv の csv_* 列（6技法・Ai は4技法）
    8/18・9/2 は dataset に無い＝正式結果フォルダ results_wao_<収録>/verify_<収録>.csv（同じ形・同じ csv_* 列）を使う
    6/28（Ai）は verify が無い＝DAM 精密採点Ai の履歴の書き出し（dam_api/dam_ai_2026-08-18T07-25-54.csv）から
      曲名（(生音)・[生音] を除き大文字小文字を無視）と演奏日時（動画名の時刻に最も近い・±180秒以内）で引く＝「別印」
  現行 G1＝dataset/G1_events_all.csv（正本・書き換えない）
    フレーム列（frame・event_frame_est）は収録ごとの G1 から（9/11＝bandpts_rerun_20260914・9/15＝rescan）
    フレーム列の無い収録（8/12・8/18・9/2・6/28）は G1_events_all の stable_read_time_idx_sec・video_time_idx_sec
      （＝帯のフレーム番号÷fps）に npz の fps を掛けて丸める
後処理＝a4_rules.py と同じ関数（values_rule・detect(rule_c=True)・gc.band_trim）。
  event_frame_est＝v2_fullscan.events_from と同じ求め方（前の確定値が最後に読めたコマ＋1）。
条件＝基準（見本追加なし）・k=3・k=3+(c)（＝v0.8f）。尺度は a4_rules.py と同じ。
出力＝reread_v08f_20261031/grade_c/
  <収録>/A4_table.csv・A4_per_counter.csv・events_v08f.csv・base_identity.csv・_inputs/（a4_rules.py 用の正規化入力）
  summary_by_session.csv・diff_vs_G1.csv・G1_events_all_v08f_candidate.csv・official_0628_dam.csv
--check-a4＝a4_apply_all.sh が書いた grade_c/<収録>/a4/A4_table.csv と本スクリプトの値（3条件）が同じか確かめる。
"""
import argparse
import collections
import csv
import datetime
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glyph_core as gc
import score_compare as sc
from a4_rules import values_rule, detect
from v2_summarize import match_interval

ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"
SCAN = ROOT + "/reread_v08f_20261031/fullscan"
OUT = ROOT + "/reread_v08f_20261031/grade_c"
G1_ALL = ROOT + "/dataset/G1_events_all.csv"
DAM_AI = "/mnt/g/研究アーカイブ/成果物/dam_api/dam_ai_2026-08-18T07-25-54.csv"
C6 = gc.COUNTERS
C4 = C6[:4]

# 収録キー → (verify・frame 列つき G1（None＝G1_events_all から換算）・技法・G1_events_all の session)
SESS = {
    "20261014": ("dataset/verify_20261014_wao.csv", "dataset/G1_events_20261014_wao.csv", C6, "20261014"),
    "20260924": ("dataset/verify_20260924.csv", "dataset/G1_events_20260924.csv", C6, "20260924"),
    "20261006": ("dataset/verify_20261006_wao.csv", "dataset/G1_events_20261006_wao.csv", C6, "20261006"),
    "20260911": ("dataset/verify_20260911.csv", "dataset/G1_events_20260911_bandpts_rerun_20260914.csv", C6, "20260911"),
    "20260915": ("dataset/verify_20260915_rescan.csv", "dataset/G1_events_20260915_rescan.csv", C6, "20260915"),
    "20261021": ("dataset/verify_20261021_wao.csv", "dataset/G1_events_20261021_wao.csv", C6, "20261021"),
    "20260902": ("results_wao_20260902/verify_20260902.csv", None, C6, "20260902"),
    "20260818": ("results_wao_20260818/verify_20260818.csv", None, C6, "20260818"),
    "20260812": ("dataset/verify_20260812.csv", None, C6, "20260812"),
    "20261006_ai": ("dataset/verify_20261006_ai.csv", "dataset/G1_events_20261006_ai.csv", C4, "20261006"),
    "20260628_ai": (None, None, C4, "20260628"),
}
ORDER = list(SESS)
RESULTS = {  # base_identity 用の正式結果（v3_fullscan_all.SESS と同じ）
    "20260924": "results_wao_20260924", "20261006": "results_wao_20261006", "20260911": "results_wao_20260911",
    "20260915": "results_wao_20260915_rescan", "20261021": "results_wao_20261021", "20260902": "results_wao_20260902",
    "20260818": "results_wao_20260818", "20260812": "results_wao_20260812", "20261014": "results_wao_20261014",
    "20261006_ai": "results_ai_20261006", "20260628_ai": "results",
}
CONDS = [("基準", False, False), ("k=3", True, False), ("k=3+(c)", True, True)]
G1_COLS = ["session", "video", "counter", "count_value", "delta", "video_time_sec", "stable_read_time_sec", "take",
           "audio_time_sec", "video_time_idx_sec", "stable_read_time_idx_sec", "multi_increment"]


def rcsv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def wcsv(p, rows, cols=None):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cols = cols or (list(rows[0].keys()) if rows else ["empty"])
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def norm_name(s):
    s = s.replace("(生音)", "").replace("[生音]", "").replace("（生音）", "")
    return re.sub(r"\s+", "", s).lower()


def video_dt(video):
    m = re.search(r"_(\d{8})_(\d{6})$", video)
    return datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def official_0628(songs):
    """6/28＝DAM 精密採点Ai の履歴（和名の列）から曲名＋時刻で引く。"""
    rows = rcsv(DAM_AI)
    out, meta = {}, []
    for s in songs:
        name = norm_name(s.rsplit("_", 2)[0])
        t = video_dt(s)
        best = None
        for r in rows:
            if norm_name(r["曲名"]) != name:
                continue
            d = (datetime.datetime.strptime(r["演奏日時"], "%Y-%m-%d %H:%M:%S") - t).total_seconds()
            if abs(d) <= 180 and (best is None or abs(d) < abs(best[0])):
                best = (d, r)
        if best is None:
            meta.append({"video": s, "dam_song": "", "dam_playDate": "", "dt_sec": "", "note": "DAM に該当なし"})
            continue
        d, r = best
        out[s] = {"csv_shakuri": r["しゃくり"], "csv_kobushi": r["こぶし"], "csv_fall": r["フォール"],
                  "csv_vibrato": r["ビブラート回数"]}
        meta.append({"video": s, "dam_song": r["曲名"], "dam_playDate": r["演奏日時"], "dt_sec": int(d),
                     "scoringId": r["採点ID"], **out[s], "note": "別印（verify 無し・DAM 履歴から曲名＋時刻で照合）"})
    return out, meta


def g1_frames(key, songs, fps_of, g1all):
    """G1 の加点（frame・event_frame_est つき）を曲ごとに返す。"""
    _v, g1p, _c, _s = SESS[key]
    out = collections.defaultdict(list)
    if g1p:
        for r in rcsv(os.path.join(ROOT, g1p)):
            if r["video"] in songs:
                out[r["video"]].append(dict(counter=r["counter"], count_value=int(r["count_value"]),
                                            delta=int(r["delta"]), frame=int(r["frame"]),
                                            event_frame_est=int(r["event_frame_est"]),
                                            video_time_sec=r["video_time_sec"], stable_read_time_sec=r["stable_read_time_sec"]))
    else:
        for s in songs:
            fps = fps_of[s]
            for r in g1all.get(s, []):
                out[s].append(dict(counter=r["counter"], count_value=int(r["count_value"]), delta=int(r["delta"]),
                                   frame=int(round(float(r["stable_read_time_idx_sec"]) * fps)),
                                   event_frame_est=int(round(float(r["video_time_idx_sec"]) * fps)),
                                   video_time_sec=r["video_time_sec"], stable_read_time_sec=r["stable_read_time_sec"]))
    return out


def events_c(z, ext, rule_c, counters):
    """values_rule → band_trim → detect（a4 と同じ）→ event_frame_est（events_from と同じ求め方）。"""
    vals = values_rule(z, ext, False)
    n = vals.shape[0]
    values = {}
    for ci, name in enumerate(C6):
        idx = np.nonzero(vals[:, ci] >= 0)[0]
        values[name] = [(int(i), int(vals[i, ci])) for i in idx]
    values, _band = gc.band_trim(values, n, gc.BAND_W_FRAMES)
    accs, evs = {}, {}
    for name in counters:
        acc, _sk = detect(values[name], False, rule_c, False)
        accs[name] = acc
        seq = values[name]
        rows = []
        for i, (fr, v, d) in enumerate(acc):
            ev = fr
            if i > 0:
                pv = acc[i - 1][1]
                last_old = None
                for f2, v2 in seq:
                    if f2 >= fr:
                        break
                    if v2 == pv:
                        last_old = f2
                if last_old is not None:
                    ev = last_old + 1
            rows.append((fr, v, d, ev))
        evs[name] = rows
    return vals, accs, evs


def time_map(g1rows, fps):
    """曲ごとの 帯フレーム→動画時刻（G1 の video_time_sec 系）の写像。範囲内は折れ線・範囲外は直線あてはめ。"""
    pts = collections.defaultdict(list)
    for r in g1rows:
        pts[r["event_frame_est"]].append(float(r["video_time_sec"]))
        pts[r["frame"]].append(float(r["stable_read_time_sec"]))
    if len(pts) < 2:
        return (lambda f: f / fps), "fps", 0.0
    xs = np.array(sorted(pts), float)
    ys = np.array([np.mean(pts[int(x)]) for x in xs])
    a, b = np.polyfit(xs, ys, 1)
    res = float(np.max(np.abs(ys - (a * xs + b))))

    def f(fr):
        if xs[0] <= fr <= xs[-1]:
            return float(np.interp(fr, xs, ys))
        return float(a * fr + b)
    return f, "interp_G1", res


def grade(only=None):
    g1all = collections.defaultdict(list)
    for r in rcsv(G1_ALL):
        g1all[r["video"]].append(r)
    summary, diff_rows, cand_rows, scanned = [], [], [], set()
    for key in ORDER:
        if only and key not in only:
            continue
        vp, _g1p, counters, sess = SESS[key]
        songs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SCAN, key, "*.npz")))
        od = os.path.join(OUT, key)
        os.makedirs(os.path.join(od, "_inputs"), exist_ok=True)
        if vp:
            ver = {r["video"]: r for r in rcsv(os.path.join(ROOT, vp))}
            mark = ""
        else:
            ver, meta = official_0628(songs)
            wcsv(os.path.join(OUT, "official_0628_dam.csv"), meta)
            mark = "別印"
        zs = {s: np.load(os.path.join(SCAN, key, s + ".npz")) for s in songs}
        fps_of = {s: float(zs[s]["fps"]) for s in songs}
        g1f = g1_frames(key, set(songs), fps_of, g1all)
        # a4_rules.py 用の正規化入力（6技法の収録だけ）
        if counters == C6:
            wcsv(os.path.join(od, "_inputs", "verify_norm.csv"),
                 [{"video": s, **{"csv_" + c: ver[s]["csv_" + c] for c in C6}} for s in songs if s in ver],
                 ["video"] + ["csv_" + c for c in C6])
            wcsv(os.path.join(od, "_inputs", "G1_frames.csv"),
                 [dict(video=s, **r) for s in songs for r in g1f.get(s, [])],
                 ["video", "counter", "count_value", "delta", "frame", "event_frame_est"])
        per_rows, ev_rows, ident = [], [], []
        for s in songs:
            z = zs[s]
            fps = fps_of[s]
            n = z["nb"].shape[0]
            truth = collections.defaultdict(list)
            truth_iv = collections.defaultdict(list)
            for r in g1f.get(s, []):
                for _ in range(r["delta"]):
                    truth[r["counter"]].append(r["frame"])
                    truth_iv[r["counter"]].append((r["event_frame_est"], r["frame"]))
            for cond, ext, rc in CONDS:
                vals, accs, evs = events_c(z, ext, rc, counters)
                for c in counters:
                    ci = C6.index(c)
                    acc = accs[c]
                    fin = acc[-1][1] if acc else 0
                    dam = int(ver[s]["csv_" + c]) if s in ver else None
                    pf = sc.expand_events(acc, list(range(n)))
                    fp, fn = sc.match_events(pf, truth[c], sc.MATCH_SEC * fps)
                    fpi, fni = match_interval(pf, truth_iv[c], sc.MATCH_SEC * fps)
                    per_rows.append({"video": s, "cond": cond, "counter": c, "final": fin,
                                     "dam": "" if dam is None else dam, "diff": "" if dam is None else fin - dam,
                                     "hit": "" if dam is None else int(fin == dam),
                                     "false_inc": fp, "missed_inc": fn, "false_inc_iv": fpi, "missed_inc_iv": fni,
                                     "readable": int((vals[:, ci] >= 0).sum()), "frames": n, "mark": mark})
                    if cond == "k=3+(c)":
                        for fr, v, d, ev in evs[c]:
                            ev_rows.append({"video": s, "counter": c, "count_value": v, "delta": d, "frame": fr,
                                            "event_frame_est": ev, "time_sec": round(fr / fps, 3),
                                            "event_time_est": round(ev / fps, 3)})
                if cond == "基準":
                    ident.append(base_identity(key, s, evs, counters))
            # 差分（加点の単位ごと）と候補
            vev = [r for r in ev_rows if r["video"] == s]
            diff_rows += diff_song(key, s, fps, counters, vev, g1f.get(s, []))
            cand_rows += cand_song(sess, s, fps, vev, g1f.get(s, []), g1all.get(s, []))
            scanned.add(s)
        wcsv(os.path.join(od, "A4_per_counter.csv"), per_rows)
        wcsv(os.path.join(od, "events_v08f.csv"), ev_rows,
             ["video", "counter", "count_value", "delta", "frame", "event_frame_est", "time_sec", "event_time_est"])
        wcsv(os.path.join(od, "base_identity.csv"), ident)
        table = []
        for cond, _e, _r in CONDS:
            rr = [r for r in per_rows if r["cond"] == cond and r["hit"] != ""]
            hits = collections.Counter()
            for r in rr:
                hits[r["video"]] += r["hit"]
            over = [r for r in rr if r["diff"] > 0]
            under = [r for r in rr if r["diff"] < 0]
            rall = [r for r in per_rows if r["cond"] == cond]
            table.append({"session": key, "cond": cond, "songs": len(songs), "songs_official": len(hits),
                          "counters": len(counters), "final_hit": sum(r["hit"] for r in rr), "final_total": len(rr),
                          "perfect_songs": sum(1 for v in hits.values() if v == len(counters)),
                          "over_cells": len(over), "over_sum": sum(r["diff"] for r in over),
                          "under_cells": len(under), "under_sum": -sum(r["diff"] for r in under),
                          "readable_pct": round(100.0 * sum(r["readable"] for r in rall) / max(1, sum(r["frames"] for r in rall)), 2),
                          "false_inc": sum(r["false_inc"] for r in rall), "missed_inc": sum(r["missed_inc"] for r in rall),
                          "false_inc_iv": sum(r["false_inc_iv"] for r in rall), "missed_inc_iv": sum(r["missed_inc_iv"] for r in rall),
                          "base_identity_relaxed": sum(1 for x in ident if x["relaxed"] == "SAME"),
                          "mark": mark})
        wcsv(os.path.join(od, "A4_table.csv"), table)
        summary += table
        print(key, [(t["cond"], t["final_hit"], t["final_total"], t["perfect_songs"]) for t in table], flush=True)
    if not only:
        summary.append({"session": "20260807", "cond": "対象外", "songs": 0,
                        "mark": "帯の切り出し映像が bands・bands_ai・bands_wao のどこにも無い＝v0.8f で読み直せない"})
        wcsv(os.path.join(OUT, "summary_by_session.csv"), summary, list(summary[0].keys()))
        wcsv(os.path.join(OUT, "diff_vs_G1.csv"), diff_rows)
        kept = 0
        for s, rows in g1all.items():
            if s in scanned:
                continue
            for r in rows:
                cand_rows.append({**{k: r[k] for k in G1_COLS}, "frame": "", "event_frame_est": "",
                                  "cand_source": "G1_kept（v0.8f の走査なし）", "time_src": ""})
                kept += 1
        cand_rows.sort(key=lambda r: (r["session"], r["video"], r["counter"], int(r["count_value"])))
        wcsv(os.path.join(OUT, "G1_events_all_v08f_candidate.csv"), cand_rows,
             G1_COLS + ["frame", "event_frame_est", "cand_source", "time_src"])
        print("diff rows", len(diff_rows), "cand rows", len(cand_rows), "kept from G1", kept)


def base_identity(key, s, evs, counters):
    """基準（見本追加なし）の加点列を正式結果の counter_events.csv と比べる。
    strict＝(技法・frame・値・増分・event_frame_est) の集合一致／relaxed＝(技法・frame・値・増分)（並び順と event_frame_est は無視）。"""
    p = os.path.join(ROOT, RESULTS[key], s, "counter_events.csv")
    if not os.path.exists(p):
        return {"video": s, "strict": "NO_REF", "relaxed": "NO_REF", "n_ref": 0, "n_got": 0, "only_ref": "", "only_got": ""}
    ref = [r for r in rcsv(p) if r["counter"] in counters]
    rs = {(r["counter"], int(r["frame"]), int(r["value"]), int(r["delta"])) for r in ref}
    rse = {(r["counter"], int(r["frame"]), int(r["value"]), int(r["delta"]), int(r["event_frame_est"])) for r in ref}
    gs, gse = set(), set()
    for c in counters:
        for fr, v, d, ev in evs[c]:
            gs.add((c, fr, v, d))
            gse.add((c, fr, v, d, ev))
    return {"video": s, "strict": "SAME" if rse == gse else "DIFF", "relaxed": "SAME" if rs == gs else "DIFF",
            "n_ref": len(ref), "n_got": len(gs), "only_ref": len(rs - gs), "only_got": len(gs - rs)}


def units(rows, fkey="frame"):
    """加点の行 → 値ごとの単位 {値: (frame, 行の増分, event_frame_est)}（delta＝2 なら2つの値が同じフレーム）。"""
    out = collections.defaultdict(dict)
    for r in rows:
        v, d = int(r["count_value"]), int(r["delta"])
        for k in range(v - d + 1, v + 1):
            out[r["counter"]][k] = (int(r[fkey]), d, int(r["event_frame_est"]))
    return out


def diff_song(key, s, fps, counters, vev, g1rows):
    uv = units([r for r in vev if int(r["delta"]) > 0])
    ug = units(g1rows)
    out = []
    for c in counters:
        for k in sorted(set(uv[c]) | set(ug[c])):
            a, b = ug[c].get(k), uv[c].get(k)
            if a and b:
                sh = b[0] - a[0]
                if sh == 0 and a[1] == b[1]:
                    st = "一致"
                elif abs(sh) <= fps * sc.MATCH_SEC:
                    st = "ずれ1秒以内"
                else:
                    st = "ずれ1秒超"
                if sh == 0 and a[1] != b[1]:
                    st = "増分ちがい"
            elif b:
                st, sh = "増えた", ""
            else:
                st, sh = "消えた", ""
            out.append({"session": key, "video": s, "counter": c, "count_value": k, "status": st,
                        "g1_frame": a[0] if a else "", "g1_delta": a[1] if a else "",
                        "v08f_frame": b[0] if b else "", "v08f_delta": b[1] if b else "",
                        "shift_frames": sh, "shift_sec": round(sh / fps, 3) if sh != "" else ""})
    return out


def cand_song(sess, s, fps, vev, g1rows, g1all_rows):
    f, src, _res = time_map(g1rows, fps)
    off = [float(r["audio_time_sec"]) - float(r["video_time_sec"]) for r in g1all_rows if r.get("audio_time_sec")]
    aoff = float(np.median(off)) if off else None
    takes = collections.Counter(r["take"] for r in g1all_rows)
    take = takes.most_common(1)[0][0] if takes else ""
    out = []
    for r in vev:
        if int(r["delta"]) <= 0:
            continue
        fr, ev = int(r["frame"]), int(r["event_frame_est"])
        vt = f(ev)
        out.append({"session": sess, "video": s, "counter": r["counter"], "count_value": r["count_value"],
                    "delta": r["delta"], "video_time_sec": f"{vt:.3f}", "stable_read_time_sec": f"{f(fr):.3f}",
                    "take": take, "audio_time_sec": "" if aoff is None else f"{vt + aoff:.3f}",
                    "video_time_idx_sec": f"{ev / fps:.3f}", "stable_read_time_idx_sec": f"{fr / fps:.3f}",
                    "multi_increment": int(int(r["delta"]) > 1), "frame": fr, "event_frame_est": ev,
                    "cand_source": "v0.8f(k=3+(c))", "time_src": src})
    return out


def check_a4():
    ok = True
    for key in ORDER:
        p = os.path.join(OUT, key, "a4", "A4_table.csv")
        if not os.path.exists(p):
            print(key, "a4 なし")
            continue
        mine = {r["cond"]: r for r in rcsv(os.path.join(OUT, key, "A4_table.csv"))}
        a4 = {r["cond"]: r for r in rcsv(p) if r["group"] == "全30曲"}
        for cond in ("基準", "k=3", "k=3+(c)"):
            keys = ["final_hit", "final_total", "perfect_songs", "over_cells", "over_sum", "under_cells", "under_sum",
                    "readable_pct", "false_inc", "missed_inc", "false_inc_iv", "missed_inc_iv"]
            bad = [k for k in keys if float(mine[cond][k]) != float(a4[cond][k])]
            ok &= not bad
            print(key, cond, "OK" if not bad else f"NG {bad}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-a4", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.check_a4:
        return check_a4()
    grade(a.only.split(",") if a.only else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
