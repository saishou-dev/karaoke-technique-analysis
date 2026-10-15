# -*- coding: utf-8 -*-
"""「観測できた区間だけで評価する」を入れた前後の precision/recall/F値を、**既に作ってある検出照合台帳から**
   組み直して並べる（2026-10-15）。CRNN の推論はやり直さない（予測の時刻と確率は台帳に全部残っているため、
   照合だけをやり直せば同じ数字が再現できる＝再現の検算も出す）。

区間＝dataset/song_valid_range.csv の [first_frame_pts, last_frame_pts]（帯動画の時刻）を
      + offset（index.json）で録音WAV時刻に直し、評価区間 span と重ねたもの。
      録画開始前／帯が映っていない区間には正解が存在しない（台帳に入れようがない）ので、
      その区間の予測を誤検出（FP）として数えないための措置。

使い方: python eval_valid_range_20261015.py [--ledger <検出照合台帳.csv>] [--tol 1.0]
出力: dataset/ledgers_20261008/検出照合_観測区間_前後_<test>.csv / .md
"""
import os, sys, csv, json, argparse, collections

BASE = r"G:\研究アーカイブ\成果物\動画解析"
OUT = os.path.join(BASE, "dataset", "ledgers_20261008")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from song_valid_range_20261015 import clip_span, load  # noqa: E402

TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JA = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール",
      "accent": "アクセント", "hammering": "ハンマリング"}

ap = argparse.ArgumentParser()
ap.add_argument("--ledger", default=os.path.join(OUT, "検出照合台帳_test20260924.csv"))
ap.add_argument("--tol", type=float, default=1.0)
ap.add_argument("--truth", default="ledger", choices=["ledger", "g1"],
                help="ledger＝台帳に入っている正解（＝index.json を作った時点）／g1＝いまの G1_events_all から組み直す")
a = ap.parse_args()


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def match(pred, truth, tol):
    """build_ledgers_20261008.py / render_pitchbar.py と同じ貪欲1対1照合。"""
    used = set(); pairs = []
    for pi in sorted(range(len(pred)), key=lambda i: pred[i]):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - pred[pi]) > tol:
                continue
            if best is None or abs(t - pred[pi]) < abs(truth[best] - pred[pi]):
                best = k
        if best is not None:
            used.add(best); pairs.append((pi, best))
    return pairs


rows = list(csv.DictReader(open(a.ledger, encoding="utf-8-sig")))
test = rows[0]["session"]
index = {d["video"]: d for d in json.load(open(os.path.join(BASE, "g3v2_20260928", "index.json"), encoding="utf-8"))}
P = collections.defaultdict(list); T = collections.defaultdict(list)
for r in rows:
    k = (r["video"], r["technique"])
    if r["pred_time_sec"] != "":
        P[k].append(float(r["pred_time_sec"]))
    if r["truth_time_sec"] != "":
        T[k].append(float(r["truth_time_sec"]))
videos = sorted({r["video"] for r in rows})
if a.truth == "g1":
    # いまの統合台帳から正解を組み直す（build_index.py と同じ規則＝multi_increment==0・wav時刻＝video_time_sec+offset）
    n_led = sum(len(v) for v in T.values())
    T = collections.defaultdict(list)
    n = 0
    for r in csv.DictReader(open(os.path.join(BASE, "dataset", "G1_events_all.csv"), encoding="utf-8-sig")):
        if r["session"] != test or r["video"] not in index or int(r["multi_increment"]) != 0:
            continue
        T[(r["video"], r["counter"])].append(float(r["video_time_sec"]) + float(index[r["video"]].get("offset", 0.0)))
        n += 1
    print(f"正解＝G1_events_all から組み直し: {n} 件（台帳に入っている正解は {n_led} 件）")
vr = load()
print(f"台帳 {len(rows)} 行 / {len(videos)} 曲 / tol ±{a.tol}s")

res = {}
spans = []
for mode in ("keep", "exclude"):
    tech = {c: [0, 0, 0] for c in TYPES}
    for v in videos:
        d = index[v]; off = float(d.get("offset", 0.0))
        span, clipped = clip_span(v, list(d["span"]), off, mode)
        if mode == "exclude":
            r0 = vr.get(v, {})
            spans.append(dict(video=v, span_lo=round(d["span"][0], 3), span_hi=round(d["span"][1], 3),
                              valid_lo=round(span[0], 3), valid_hi=round(span[1], 3),
                              offset=off, first_frame_pts=r0.get("first_frame_pts", ""),
                              last_frame_pts=r0.get("last_frame_pts", ""),
                              pre_record_total=r0.get("pre_record_total", ""), clipped=int(clipped)))
        for c in TYPES:
            pr = [t for t in P[(v, c)] if span[0] <= t <= span[1]]
            tr = [t for t in T[(v, c)] if span[0] <= t <= span[1]]
            h = len(match(pr, tr, a.tol))
            tech[c][0] += h; tech[c][1] += len(pr) - h; tech[c][2] += len(tr) - h
    res[mode] = tech

L = [f"# 検出器の評価：観測できた区間だけで測った前後（{test}・{len(videos)}曲・±{a.tol}s・2026-10-15）", "",
     f"- 台帳＝{os.path.basename(a.ledger)}（予測の時刻・確率はそのまま・照合だけやり直し）／正解＝{a.truth}",
     "- keep＝従来（評価区間＝[最初の正解−5s, 最後の正解+2s]）／exclude＝観測できた区間と重ねたもの（既定）",
     "- 区間表＝dataset/song_valid_range.csv", "",
     "| 技法 | TP | FP | FN | 適合率 | 再現率 | F値 | TP | FP | FN | 適合率 | 再現率 | F値 | ΔF |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
tot = {m: [0, 0, 0] for m in res}
for c in TYPES:
    a0 = res["keep"][c]; a1 = res["exclude"][c]
    for i in range(3):
        tot["keep"][i] += a0[i]; tot["exclude"][i] += a1[i]
    p0, r0, f0 = prf(*a0); p1, r1, f1 = prf(*a1)
    L.append(f"| {JA[c]} | {a0[0]} | {a0[1]} | {a0[2]} | {p0:.3f} | {r0:.3f} | {f0:.3f} | "
             f"{a1[0]} | {a1[1]} | {a1[2]} | {p1:.3f} | {r1:.3f} | {f1:.3f} | {f1-f0:+.3f} |")
p0, r0, f0 = prf(*tot["keep"]); p1, r1, f1 = prf(*tot["exclude"])
L.append(f"| **全体** | **{tot['keep'][0]}** | **{tot['keep'][1]}** | **{tot['keep'][2]}** | **{p0:.3f}** | **{r0:.3f}** | **{f0:.3f}** | "
         f"**{tot['exclude'][0]}** | **{tot['exclude'][1]}** | **{tot['exclude'][2]}** | **{p1:.3f}** | **{r1:.3f}** | **{f1:.3f}** | **{f1-f0:+.3f}** |")
tag = "" if a.truth == "ledger" else "_truth-g1"
nclip = sum(s["clipped"] for s in spans)
L += ["", f"- 区間が狭まった曲 = {nclip}/{len(videos)}", ""]
if nclip:
    L += ["| 曲 | 従来の区間 | 観測区間と重ねた区間 | 差(前) | 差(後) |", "|---|---|---|---|---|"]
    for s in spans:
        if s["clipped"]:
            L.append(f"| {s['video']} | {s['span_lo']}〜{s['span_hi']} | {s['valid_lo']}〜{s['valid_hi']} | "
                     f"{s['valid_lo']-s['span_lo']:+.3f}s | {s['valid_hi']-s['span_hi']:+.3f}s |")
md = os.path.join(OUT, f"検出照合_観測区間_前後_{test}{tag}.md")
open(md, "w", encoding="utf-8").write("\n".join(L) + "\n")
with open(os.path.join(OUT, f"検出照合_観測区間_前後_{test}{tag}.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(spans[0].keys())); w.writeheader(); w.writerows(spans)
print("\n".join(L))
print("WROTE", md)
