#!/usr/bin/env python3
"""supp_digit_only.py ＝ 補足測定：「数字でない」見本を照合から外した場合（2026-10-30）

内訳の (v)＝類似度はしきい値に届くのに、最も似た見本が「数字でない」見本（ラベル null）だったために
読まなかったコマ。この見本を照合の候補から外し、数字の見本だけで最良を選んだときの答えを
unread_sweep.py と同じ採点で測る（見本そのものは変えない＝過去の全収録の読み直しは不要な変更の候補）。

使い方: python supp_digit_only.py --work <work> --out <out>
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
import unread_sweep as us


def value_digit_only(f, t):
    nb, per = f
    if nb == 0:
        return None
    digits = []
    for _s, _l, sd, ld in per:
        if sd == us.FLAT or sd < t or ld is None:
            return None
        digits.append(ld)
    return digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/mnt/g/研究アーカイブ/成果物/動画解析")
    ap.add_argument("--results", default="results_wao_20261014")
    ap.add_argument("--session", default="20261014")
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()
    gt, cert = sc.load_gt(a.work)
    songs = sorted({k[0] for k in gt})
    frames = {s: np.load(os.path.join(a.work, s, "frames.npy")).tolist() for s in songs}
    fps = {}
    with open(os.path.join(a.work, "input_summary.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fps[r["video"]] = float(r["fps"])
    dam = {}
    with open(os.path.join(a.root, "dataset", f"verify_{a.session}_wao.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                dam[r["video"]] = {c: int(r["csv_" + c]) for c in gc.COUNTERS}
    truth_ev = collections.defaultdict(list)
    with open(os.path.join(a.root, "dataset", f"G1_events_{a.session}_wao.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                for _ in range(int(r["delta"])):
                    truth_ev[(r["video"], r["counter"])].append(int(r["frame"]))
    band = {s: us.band_present(a.root, a.results, s) for s in songs}
    in_band = {k for k in gt if band[k[0]] and band[k[0]][0] <= frames[k[0]][k[1]] <= band[k[0]][1]}
    tpl = gc.Templates(os.path.join(a.root, "templates", "wao_20260816", "cluster_reps_wao_ext.npy"))
    feat, tf = us.extract_features(a.work, songs, tpl)
    print(f"features {tf:.1f}s", flush=True)
    band_w = max(1, round(150.0 / a.stride))
    rows = []
    for t in us.NCC_GRID:
        preds = {k: value_digit_only(f, t) for k, f in feat.items()}
        for s_min in us.STABLE_GRID:
            r, _ = us.score_condition(preds, gt, cert, songs, frames, fps, dam, truth_ev, s_min, band_w, in_band)
            r = {"variant": "digit_only", "ncc_min": t, "stable_min": s_min, **r}
            rows.append(r)
            print(f"digit_only t={t:.2f} s={s_min} acc_all={r['acc_all']:6.2f} acc_ans={r['acc_answered']:6.2f} "
                  f"abstain={r['abstain']:6.2f} misread={r['misread']:5.2f} final={r['final_hit']}/24 "
                  f"FP={r['false_inc']} FN={r['missed_inc']}", flush=True)
    with open(os.path.join(a.out, "supp_digit_only.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
