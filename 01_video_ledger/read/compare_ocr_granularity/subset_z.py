#!/usr/bin/env python3
"""subset_z.py = 条件Z が部分集合（--every 32／64）で走った手法について、同じフレームの上で A〜C2 と並べる（2026-10-30）

score_gran.py の採点は「90%未満しか値の無い手法」の曲末一致・偽の増分を測定対象外（—）にする。
そこで Z の部分集合と全く同じキー（曲・標本番号・カウンター・正解確定）だけで、
値を出した時の正しさ・読まない率・「1が11」型の誤りの件数を条件ごとに数え直す。後処理は掛けない（1件ずつの読み）。
使い方: $PY subset_z.py --pred pred --out result
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_gran as S   # noqa: E402


def stats(keys, pr, gt):
    n = ans = cor = 0
    more = less = e111 = dup = 0
    for k in keys:
        g = gt[k]
        v = pr.get(k)
        n += 1
        if v is None:
            continue
        ans += 1
        if v == g:
            cor += 1
            continue
        sg, sv = str(g), str(v)
        e111 += (g == 1 and v == 11)
        dup += (len(sg) == 1 and sv == sg * 2)
        more += len(sv) > len(sg)
        less += len(sv) < len(sg)
    return {"cells": n, "answered": ans,
            "acc_answered": round(100.0 * cor / max(1, ans), 2),
            "abstain": round(100.0 * (n - ans) / max(1, n), 2),
            "acc_all": round(100.0 * cor / max(1, n), 2),
            "e_1to11": e111, "e_dup": dup, "e_more_digit": more, "e_less_digit": less}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/toshiaki/ocr_gran_20261029/work_b")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gt, cert = S.load_gt(a.work)
    rows = []
    for zc in ["easyocr_readtext", "paddle_full", "tess_psm11"]:
        zp, zm = S.load_pred_set(a.pred, "Z", zc)
        if zp is None:
            continue
        keys = [k for k in zp if k in gt and cert[k]]
        frames = len({(k[0], k[1]) for k in keys})
        for cond in ["Z", "A", "B", "C", "C2"]:
            for cand in [zc, "ncc", "paddle_rec_conf90"]:
                if cond == "Z" and cand != zc:
                    continue
                base = cand.replace("_conf90", "")
                cmin = 0.90 if cand.endswith("_conf90") else None
                pr, _m = S.load_pred_set(a.pred, cond, base, cmin)
                if pr is None or not all(k in pr for k in keys):
                    continue
                r = {"subset_of": "Z_" + zc, "subset_frames": frames, "cond": cond, "cand": cand}
                r.update(stats(keys, pr, gt))
                rows.append(r)
                print(r, flush=True)
    out = os.path.join(a.out, "subset_z.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
