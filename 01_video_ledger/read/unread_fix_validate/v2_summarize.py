#!/usr/bin/env python3
"""v2_summarize.py ＝ 検証2の集計＝全フレーム走査の保存（<曲>.npz）から2条件の曲末の値を作り、公式採点データと突合（2026-10-30）

条件
  基準＝見本追加なし（v0.8e と同じ＝正式結果 results_wao_20261014 と counter_events が一致することを走査時に確認済み）
  k＝3＝追加見本30個（10/14 の4曲から作った templates_k3_all4.npy）を足した読み
後処理は v0.8e と同じ（帯消失ガード 150・安定判定6・減らない制約）＝v2_fullscan.events_from。
尺度
  曲末一致＝曲×技法で、曲末の値が公式採点データ（verify_20261014_wao.csv の csv_*）と同じ件数（180 中）
  完全一致の曲＝6技法すべて一致した曲の数（30 中）
  過検出＝曲末の値が公式より多い曲×技法の件数と、その差の合計
  不足＝曲末の値が公式より少ない曲×技法の件数と、その差の合計
  読めたコマの率＝全フレーム×6技法のうち値を出したコマの割合（帯消失ガードの前）
  偽の増分・取り落とし＝増分の位置を正解データ（G1_events_20261014_wao.csv）と ±1秒で1対1に対応づけた時の余りと不足
    （10/18 と同じ＝正解の位置は G1 の frame 列＝正式結果の安定読み取りの位置）
  偽の増分・取り落とし（区間）＝正解の位置を G1 の [event_frame_est − 1秒, frame ＋ 1秒] の区間にした版
    （G1 の frame 列は見本追加なしの読みで決まった位置なので、早く読めた増分を「ずれ」と数えないための補助）
曲の組＝全30曲／見本作りに使っていない26曲（10/14 の4曲を除く）
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyph_core as gc
import score_compare as sc
from v2_fullscan import events_from, values_from

ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"
COUNTERS = gc.COUNTERS
TRAIN4 = ["サイレントマジョリティー_20261014_105318", "スターダスト_20261014_120507",
          "共犯_20261014_111506", "恋人ごっこ(生音)_20261014_124950"]


def match_interval(pred_frames, ivs, tol):
    """正解の区間 [a−tol, b＋tol] に入る増分を、区間までの距離の近い順に1対1で対応づける。"""
    used = [False] * len(ivs)
    fp = 0
    for pf in sorted(pred_frames):
        best, bestd = -1, None
        for j, (a_, b_) in enumerate(ivs):
            if used[j]:
                continue
            if a_ - tol <= pf <= b_ + tol:
                d = 0 if a_ <= pf <= b_ else min(abs(pf - a_), abs(pf - b_))
                if bestd is None or d < bestd:
                    best, bestd = j, d
        if best < 0:
            fp += 1
        else:
            used[best] = True
    return fp, sum(1 for u in used if not u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default="dataset/verify_20261014_wao.csv")
    ap.add_argument("--g1", default="dataset/G1_events_20261014_wao.csv")
    a = ap.parse_args()
    ver = {}
    with open(os.path.join(ROOT, a.verify), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ver[r["video"]] = r
    truth = collections.defaultdict(list)
    truth_iv = collections.defaultdict(list)
    with open(os.path.join(ROOT, a.g1), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            for _ in range(int(r["delta"])):
                truth[(r["video"], r["counter"])].append(int(r["frame"]))
                truth_iv[(r["video"], r["counter"])].append((int(r["event_frame_est"]), int(r["frame"])))
    scanlog = {}
    p = os.path.join(a.scan, "scan_log.csv")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            x = line.rstrip("\n").split(",")
            scanlog[x[0]] = x
    songs = [s for s in ver if os.path.exists(os.path.join(a.scan, s + ".npz"))]
    per_rows, song_rows = [], []
    for s in songs:
        z = np.load(os.path.join(a.scan, s + ".npz"))
        fps = float(z["fps"])
        n = z["nb"].shape[0]
        for cond, use_ext in [("基準", False), ("k=3", True)]:
            vals = values_from(z["nb"], z["bs"], z["bl"], z["es"], z["el"], use_ext)
            _rows, accs, band = events_from(vals, fps)
            hits = 0
            for ci, c in enumerate(COUNTERS):
                acc = accs[c]
                fin = acc[-1][1] if acc else 0
                dam = int(ver[s]["csv_" + c])
                off = int(ver[s]["read_" + c])
                pf = sc.expand_events(acc, list(range(n)))
                fp, fn = sc.match_events(pf, truth[(s, c)], sc.MATCH_SEC * fps)
                fpi, fni = match_interval(pf, truth_iv[(s, c)], sc.MATCH_SEC * fps)
                readable = int((vals[:, ci] >= 0).sum())
                hits += fin == dam
                per_rows.append({"video": s, "cond": cond, "counter": c, "final": fin, "dam": dam, "official_read": off,
                                 "diff": fin - dam, "hit": int(fin == dam), "false_inc": fp, "missed_inc": fn,
                                 "false_inc_iv": fpi, "missed_inc_iv": fni, "readable": readable, "frames": n, "train4": int(s in TRAIN4)})
            song_rows.append({"video": s, "cond": cond, "hits6": hits, "perfect": int(hits == 6), "frames": n,
                              "band": f"{band[0]}..{band[1]}", "scan_sec": scanlog.get(s, ["", "", "", ""])[3],
                              "identity": scanlog.get(s, ["", "", "", "", ""])[4], "train4": int(s in TRAIN4)})
    table = []
    for grp, gs in [("全30曲", songs), ("見本作りに使っていない26曲", [s for s in songs if s not in TRAIN4]),
                    ("見本作りに使った4曲", [s for s in songs if s in TRAIN4])]:
        for cond in ["基準", "k=3"]:
            rr = [r for r in per_rows if r["cond"] == cond and r["video"] in gs]
            sr = [r for r in song_rows if r["cond"] == cond and r["video"] in gs]
            over = [r for r in rr if r["diff"] > 0]
            under = [r for r in rr if r["diff"] < 0]
            table.append({
                "group": grp, "cond": cond, "songs": len(gs),
                "final_hit": sum(r["hit"] for r in rr), "final_total": len(rr),
                "perfect_songs": sum(r["perfect"] for r in sr),
                "over_cells": len(over), "over_sum": sum(r["diff"] for r in over),
                "under_cells": len(under), "under_sum": -sum(r["diff"] for r in under),
                "abs_diff_sum": sum(abs(r["diff"]) for r in rr),
                "readable_pct": round(100.0 * sum(r["readable"] for r in rr) / max(1, sum(r["frames"] for r in rr)), 2),
                "false_inc": sum(r["false_inc"] for r in rr), "missed_inc": sum(r["missed_inc"] for r in rr),
                "false_inc_iv": sum(r["false_inc_iv"] for r in rr), "missed_inc_iv": sum(r["missed_inc_iv"] for r in rr),
                "base_matches_official_read": sum(int(r["final"] == r["official_read"]) for r in rr) if cond == "基準" else "",
            })
    os.makedirs(a.out, exist_ok=True)
    for name, rows in [("V2_table.csv", table), ("V2_per_counter.csv", per_rows), ("V2_per_song.csv", song_rows)]:
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for r in table:
        print(json.dumps(r, ensure_ascii=False))
    # 曲×技法で値が変わったもの
    chg = collections.defaultdict(dict)
    for r in per_rows:
        chg[(r["video"], r["counter"])][r["cond"]] = r
    print("changed cells (video counter base->k3 dam):")
    for (s, c), d in sorted(chg.items()):
        if d["基準"]["final"] != d["k=3"]["final"]:
            print(f"  {s} {c} {d['基準']['final']}->{d['k=3']['final']} dam={d['基準']['dam']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
