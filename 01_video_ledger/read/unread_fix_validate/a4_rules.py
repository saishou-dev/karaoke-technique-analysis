#!/usr/bin/env python3
"""a4_rules.py ＝ 候補A-4＝「1が11」型の過検出を止める後処理の規則を、全フレーム走査の保存（<曲>.npz）に当てて測る（2026-10-30）

規則（k＝3 の読みに後処理として当てる・npz は読むだけ）
  (a) 2桁読みで十の位が「1」のとき、両方の桁の類似度が 0.80 以上でなければ読まない（コマ単位・帯消失ガードの前）
  (b) 前の確定値から +2 以上の跳びは確定しない（確定値は据え置き＝multi_increment 扱いで捨てる）
  (c) 1桁→2桁に上がる最初の確定は、2桁の値が 6 コマでなく 12 コマ続いた時だけ
      実装＝時刻順に見て、12 コマ以上続いた2桁の値が現れるまでは、12 コマ未満の2桁の値の連続を安定値の候補から外す
  参考 (b10) 1桁の確定値から +10 以上の跳びだけ確定しない（材料50 の候補A-4 の例）
条件＝基準・k＝3・k＝3＋(a)・k＝3＋(b)・k＝3＋(c)・k＝3＋(a)(b)・参考 k＝3＋(b10)
尺度は v2_summarize.py と同じ（曲末一致・完全一致の曲・過検出・不足・読めたコマの率・偽の増分と取り落とし（点・区間））
基準と k＝3 は v2_summarize の値（result/V2/V2_table.csv）と一致することを確かめる。
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
from v2_fullscan import FLAT, values_from
from v2_summarize import ROOT, TRAIN4, match_interval

COUNTERS = gc.COUNTERS
SIM_A = 0.80
STABLE_C = 12


def values_rule(z, use_ext, rule_a):
    """values_from と同じ値に、規則 (a) を足したもの。"""
    nb, bs, bl, es, el = z["nb"], z["bs"], z["bl"], z["es"], z["el"]
    out = values_from(nb, bs, bl, es, el, use_ext)
    if not rule_a:
        return out
    s = bs.copy()
    l = bl.copy()
    if use_ext:
        rep = (es > bs) & (bs != FLAT)
        s[rep] = es[rep]
        l[rep] = el[rep]
    out = out.copy()
    for ci in range(6):
        m = (nb[:, ci] == 2) & (out[:, ci] >= 0) & (l[:, ci, 0] == 1) & ((s[:, ci, 0] < SIM_A) | (s[:, ci, 1] < SIM_A))
        out[m, ci] = -1
    return out


def detect(seq, rule_b=False, rule_c=False, rule_b10=False, stable_min=gc.STABLE_MIN):
    """gc.detect_increments の写しに規則 (b)(c)(b10) を足したもの。全部 False なら同じ結果。"""
    runs = []
    run_v = None
    for fr, v in seq:
        if run_v is not None and v == run_v:
            runs[-1][2] += 1
        else:
            runs.append([fr, v, 1])
            run_v = v
    if rule_c:
        kept, seen_long2 = [], False
        for r in runs:
            if r[2] < stable_min:
                continue
            if r[1] >= 10 and not seen_long2:
                if r[2] < STABLE_C:
                    continue
                seen_long2 = True
            kept.append(r)
        runs = kept
    else:
        runs = [r for r in runs if r[2] >= stable_min]
    n = len(runs)
    accepted = []
    skipped = 0
    if n:
        dp = [0.0] * n
        prev = [-1] * n
        for i in range(n):
            dp[i] = runs[i][2]
            for j in range(i):
                if runs[j][1] <= runs[i][1] and dp[j] + runs[i][2] > dp[i]:
                    dp[i] = dp[j] + runs[i][2]
                    prev[i] = j
        best = int(np.argmax(dp))
        path = []
        while best != -1:
            path.append(best)
            best = prev[best]
        path.reverse()
        cur = None
        for idx in path:
            fr, v, _w = runs[idx]
            if cur is None:
                accepted.append((fr, v, 0))
                cur = v
            elif v > cur:
                if rule_b and v - cur >= 2:
                    skipped += 1
                    continue
                if rule_b10 and cur <= 9 and v - cur >= 10:
                    skipped += 1
                    continue
                accepted.append((fr, v, v - cur))
                cur = v
    return accepted, skipped


def accs_from(vals, rule_b, rule_c, rule_b10):
    nframes = vals.shape[0]
    values = {}
    for ci, name in enumerate(COUNTERS):
        idx = np.nonzero(vals[:, ci] >= 0)[0]
        values[name] = [(int(i), int(vals[i, ci])) for i in idx]
    values, _band = gc.band_trim(values, nframes, gc.BAND_W_FRAMES)
    accs, skips = {}, {}
    for name in COUNTERS:
        accs[name], skips[name] = detect(values[name], rule_b, rule_c, rule_b10)
    return accs, skips


CONDS = [
    ("基準", dict(ext=False, a=False, b=False, c=False, b10=False)),
    ("k=3", dict(ext=True, a=False, b=False, c=False, b10=False)),
    ("k=3+(a)", dict(ext=True, a=True, b=False, c=False, b10=False)),
    ("k=3+(b)", dict(ext=True, a=False, b=True, c=False, b10=False)),
    ("k=3+(c)", dict(ext=True, a=False, b=False, c=True, b10=False)),
    ("k=3+(a)(b)", dict(ext=True, a=True, b=True, c=False, b10=False)),
    ("参考 k=3+(b10)", dict(ext=True, a=False, b=False, c=False, b10=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v2", default="")
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
    songs = [s for s in ver if os.path.exists(os.path.join(a.scan, s + ".npz"))]
    per_rows = []
    for s in songs:
        z = np.load(os.path.join(a.scan, s + ".npz"))
        fps = float(z["fps"])
        n = z["nb"].shape[0]
        cache = {}
        for cond, o in CONDS:
            key = (o["ext"], o["a"])
            if key not in cache:
                cache[key] = values_rule(z, o["ext"], o["a"])
            vals = cache[key]
            accs, skips = accs_from(vals, o["b"], o["c"], o["b10"])
            for ci, c in enumerate(COUNTERS):
                acc = accs[c]
                fin = acc[-1][1] if acc else 0
                dam = int(ver[s]["csv_" + c])
                pf = sc.expand_events(acc, list(range(n)))
                fp, fn = sc.match_events(pf, truth[(s, c)], sc.MATCH_SEC * fps)
                fpi, fni = match_interval(pf, truth_iv[(s, c)], sc.MATCH_SEC * fps)
                per_rows.append({"video": s, "cond": cond, "counter": c, "final": fin, "dam": dam, "diff": fin - dam,
                                 "hit": int(fin == dam), "skipped_jumps": skips[c],
                                 "false_inc": fp, "missed_inc": fn, "false_inc_iv": fpi, "missed_inc_iv": fni,
                                 "readable": int((vals[:, ci] >= 0).sum()), "frames": n, "train4": int(s in TRAIN4)})
    table = []
    for grp, gs in [("全30曲", songs), ("見本作りに使っていない26曲", [s for s in songs if s not in TRAIN4])]:
        for cond, _o in CONDS:
            rr = [r for r in per_rows if r["cond"] == cond and r["video"] in gs]
            hits = collections.Counter()
            for r in rr:
                hits[r["video"]] += r["hit"]
            over = [r for r in rr if r["diff"] > 0]
            under = [r for r in rr if r["diff"] < 0]
            table.append({
                "group": grp, "cond": cond, "songs": len(gs),
                "final_hit": sum(r["hit"] for r in rr), "final_total": len(rr),
                "perfect_songs": sum(1 for v in hits.values() if v == 6),
                "over_cells": len(over), "over_sum": sum(r["diff"] for r in over),
                "under_cells": len(under), "under_sum": -sum(r["diff"] for r in under),
                "readable_pct": round(100.0 * sum(r["readable"] for r in rr) / max(1, sum(r["frames"] for r in rr)), 2),
                "false_inc": sum(r["false_inc"] for r in rr), "missed_inc": sum(r["missed_inc"] for r in rr),
                "false_inc_iv": sum(r["false_inc_iv"] for r in rr), "missed_inc_iv": sum(r["missed_inc_iv"] for r in rr),
                "skipped_jumps": sum(r["skipped_jumps"] for r in rr),
            })
    os.makedirs(a.out, exist_ok=True)
    for name, rows in [("A4_table.csv", table), ("A4_per_counter.csv", per_rows)]:
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for r in table:
        print(json.dumps(r, ensure_ascii=False))
    # 基準・k＝3 が v2_summarize の表と同じか
    if a.v2:
        ref = {}
        with open(os.path.join(a.v2, "V2_table.csv"), encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ref[(r["group"], r["cond"])] = r
        keys = ["final_hit", "perfect_songs", "over_cells", "over_sum", "under_cells", "under_sum", "readable_pct",
                "false_inc", "missed_inc", "false_inc_iv", "missed_inc_iv"]
        for r in table:
            if r["cond"] in ("基準", "k=3"):
                rf = ref[(r["group"], r["cond"])]
                bad = [k for k in keys if float(rf[k]) != float(r[k])]
                print("V2一致", r["group"], r["cond"], "OK" if not bad else f"NG {bad}")
    # k=3 から曲末の値が変わった曲×技法
    byk = {(r["video"], r["counter"], r["cond"]): r for r in per_rows}
    print("cells changed vs k=3 (video counter base k3 -> rule dam):")
    for cond, _o in CONDS[2:]:
        for s in songs:
            for c in COUNTERS:
                r = byk[(s, c, cond)]
                k3 = byk[(s, c, "k=3")]
                if r["final"] != k3["final"]:
                    print(f"  [{cond}] {s} {c} base={byk[(s, c, '基準')]['final']} k3={k3['final']} -> {r['final']} dam={r['dam']}")
    print("k=3 で改善した5件の残り具合:")
    for s in songs:
        for c in COUNTERS:
            b0, k3 = byk[(s, c, "基準")], byk[(s, c, "k=3")]
            if b0["final"] != k3["final"]:
                print("  ", s, c, "base", b0["final"], "k3", k3["final"], "dam", k3["dam"], " ".join(
                    f"{cond}={byk[(s, c, cond)]['final']}" for cond, _o in CONDS[2:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
