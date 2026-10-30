#!/usr/bin/env python3
"""reproduce_check.py = 条件B（技法別の数字窓）の再実行が 2026-10-18 と一致するかを実測（2026-10-29）

2026-10-18 の比較実験の入力は「帯」ではなく**技法別の数字窓**だった（extract_rois.py が
results_wao_<S>/<曲>/roi.json のカウンターごとの枠で切っている）。
したがって今回の条件B が前回と同一条件であり、ここが食い違えば先へ進めない。

比べるもの = pred_B_<cand>.csv（全件・sub=1）と ocr_compare_20261018/pred/pred_<cand>.csv
1件ずつ突き合わせ、違う件数と最初の例を出す。
"""
import argparse
import csv
import os
import sys

PREV = "/mnt/g/研究アーカイブ/成果物/動画解析/ocr_compare_20261018/pred"


def load(path):
    d = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d[(r["video"], int(r["sample_idx"]), r["counter"])] = r["pred"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="条件Bの全件予測を置いた場所")
    ap.add_argument("--prev", default=PREV)
    ap.add_argument("--cands", default="ncc,nn_l2")
    a = ap.parse_args()
    bad = 0
    for cand in a.cands.split(","):
        pn = os.path.join(a.new, "pred_B_%s.csv" % cand)
        pp = os.path.join(a.prev, "pred_%s.csv" % cand)
        if not (os.path.exists(pn) and os.path.exists(pp)):
            print("SKIP %s (ファイルなし)" % cand)
            continue
        new, prev = load(pn), load(pp)
        ks = set(new) | set(prev)
        diff = [k for k in ks if new.get(k) != prev.get(k)]
        print("%-8s 件数 new=%d prev=%d 不一致=%d" % (cand, len(new), len(prev), len(diff)))
        for k in sorted(diff)[:5]:
            print("   %s new=%r prev=%r" % (k, new.get(k), prev.get(k)))
        if diff:
            bad += 1
    print("RESULT:", "IDENTICAL" if bad == 0 else "DIFFERENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
