#!/usr/bin/env python3
"""fix_b_diag.py ＝ 候補B（前半）＝区分 (ii)「数字の枠が1つも切り出せない」の原因を、切り出しの段階ごとに数える（2026-10-30）

対象＝材料47 の cells_v08e.csv で区分 (ii) の正解確定コマ（4,275件であることを先に確かめる）。
1コマについて7段階の明るさの区切りそれぞれで「どの段で落ちたか」を求め、
いちばん先まで進んだ区切りの段を、そのコマの「0個になった段」とする（fix_b_boxes.STAGES）。
「1 大きさ」で落ちたコマは、全区切りを通していちばん大きい白の塊が、どの大きさの条件を外れたかも数える。
既定値の glyph_boxes_p が gc.glyph_boxes と全コマで同じ箱を返すことも測る。
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_b_boxes as fb
import fix_common as fc
import glyph_core as gc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T0 = time.time()
    st = fc.load_setup()
    songs = st["songs"]
    rois = fc.load_rois(fc.WORK, songs)
    cat = {}
    with open(fc.CELLS47, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            cat[(r["video"], int(r["sample_idx"]), r["counter"])] = r["category"]
    ii = [k for k in st["gt"] if st["cert"][k] and cat[k].startswith("(ii)")]
    print("(ii) certain =", len(ii), flush=True)
    if len(ii) != 4275:
        print("STOP: (ii) の件数が材料47 と違う", flush=True)
        return 3

    # 同一性（全コマ）
    mism = 0
    for s in songs:
        for c in fc.COUNTERS:
            arr = rois[(s, c)]
            for i in range(arr.shape[0]):
                g = arr[i].max(axis=2)
                b0 = gc.glyph_boxes(g)
                b1 = fb.glyph_boxes_p(g)
                if (b0 is None) != (b1 is None) or (b0 is not None and [tuple(map(int, x)) for x in b0] != [tuple(map(int, x)) for x in b1]):
                    mism += 1
    print("identity glyph_boxes_p vs gc.glyph_boxes: mismatch =", mism, flush=True)
    if mism:
        print("STOP: 置き直した切り出しが一致しない", flush=True)
        return 4

    stage_cnt = collections.Counter()
    reason_cnt = collections.Counter()
    reason_single = collections.Counter()
    hist_h = collections.Counter()
    hist_w = collections.Counter()
    junk_min = collections.defaultdict(list)
    nbox_many = collections.Counter()
    rows = []
    for k in ii:
        s, i, c = k
        g = rois[(s, c)][i].max(axis=2)
        res, sl, comps = fb.glyph_boxes_p(g, diag=True)
        assert res is None
        cstage = max(x[0] for x in sl)
        grp = "しゃくり" if c == "shakuri" else "それ以外"
        stage_cnt[(grp, fb.STAGES[cstage])] += 1
        stage_cnt[(c, fb.STAGES[cstage])] += 1
        big = max(comps, key=lambda t: t[5]) if comps else None
        rs = fb.size_reasons(big) if big else []
        if cstage == 1:
            reason_cnt[(grp, "+".join(rs) if rs else "（条件内）")] += 1
            for r_ in rs:
                reason_single[(grp, r_)] += 1
            hist_h[(grp, int(big[4]) // 4 * 4)] += 1
            hist_w[(grp, int(big[3]) // 4 * 4)] += 1
        if cstage == 3:
            junk_min[grp].append(min(x[2] for x in sl if x[0] == 3))
        if cstage == 2:
            nbox_many[(grp, min(x[1] for x in sl if x[0] == 2))] += 1
        rows.append([s, i, c, st["gt"][k], fb.STAGES[cstage], ";".join(str(x[0]) for x in sl),
                     "" if big is None else f"T{big[0]} w{big[3]} h{big[4]} a{big[5]}", "+".join(rs)])

    grps = ["しゃくり", "それ以外"]
    with open(os.path.join(a.out, "B_stage_table.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["stage"] + grps + ["合計"] + fc.COUNTERS)
        for sname in fb.STAGES[:5]:
            w.writerow([sname] + [stage_cnt[(g_, sname)] for g_ in grps] +
                       [sum(stage_cnt[(g_, sname)] for g_ in grps)] + [stage_cnt[(c, sname)] for c in fc.COUNTERS])
        w.writerow(["合計"] + [sum(stage_cnt[(g_, sn)] for sn in fb.STAGES) for g_ in grps] + [len(ii)] +
                   [sum(stage_cnt[(c, sn)] for sn in fb.STAGES) for c in fc.COUNTERS])
    with open(os.path.join(a.out, "B_size_reasons.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["kind", "group", "reason_or_bin", "cells"])
        for (g_, r_), n in sorted(reason_cnt.items()):
            w.writerow(["組み合わせ", g_, r_, n])
        for (g_, r_), n in sorted(reason_single.items()):
            w.writerow(["単独（重複あり）", g_, r_, n])
        for (g_, b), n in sorted(hist_h.items()):
            w.writerow(["最大の塊の高さ（4px刻み）", g_, b, n])
        for (g_, b), n in sorted(hist_w.items()):
            w.writerow(["最大の塊の幅（4px刻み）", g_, b, n])
        for (g_, b), n in sorted(nbox_many.items()):
            w.writerow(["個数で落ちた時の最少の箱数", g_, b, n])
        for g_, v in junk_min.items():
            q = np.percentile(v, [0, 25, 50, 75, 100]).tolist()
            w.writerow(["ごみ上限で落ちた時の最小ごみ面積（0/25/50/75/100%点）", g_, ";".join(f"{x:.0f}" for x in q), len(v)])
    with open(os.path.join(a.out, "B_cells_ii.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["video", "sample_idx", "counter", "gt_value", "stage", "stage_per_threshold", "largest_component", "size_reasons"])
        w.writerows(rows)
    json.dump({"ii_certain": len(ii), "identity_mismatch": mism, "elapsed_sec": round(time.time() - T0, 1)},
              open(os.path.join(a.out, "B_diag_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for sname in fb.STAGES[:5]:
        print(sname, {g_: stage_cnt[(g_, sname)] for g_ in grps}, flush=True)
    print("reasons:", dict(reason_cnt), flush=True)
    print("DONE", round(time.time() - T0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
