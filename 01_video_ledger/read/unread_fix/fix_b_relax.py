#!/usr/bin/env python3
"""fix_b_relax.py ＝ 候補B（後半）＝切り出しの条件を1段ずつ緩めた時に、(ii) から戻る件数と他の区分で増える誤読を測る（2026-10-30）

使い方: python fix_b_relax.py --out <out> --tag <名前> --variants '<JSON 配列>'
  各要素＝{"name": "表示名", <fix_b_boxes.DEFAULT のキー>: 値, ...}（指定しないキーは既定値のまま）
見本・しきい値0.75・安定判定6・採点は現行のまま。箱の切り出しの条件だけを変える。
数える値
  (ii) の正解確定コマ（しゃくり／それ以外）が、値あり・正解／値あり・誤り／まだ読まない のどれになったか
  (ii) 以外の正解確定コマで、新たに誤りになった件数・正解でなくなった件数
  10/18 と同じ尺度の全体の行（正解率・値を出した時の正解率・読まない率・誤読率・曲末一致・偽の増分・取り落とし）
"""
import argparse
import collections
import csv
import functools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_b_boxes as fb
import fix_common as fc
import glyph_core as gc
import unread_sweep as us

METRICS = ["acc_all", "acc_answered", "abstain", "misread", "final_hit", "false_inc", "missed_inc",
           "answered", "correct", "misread_cells", "abstain_cells"]


def run_variant(rois, songs, tpl, st, P):
    cells, _g, _u, bs, bl = fc.extract(rois, songs, tpl, boxes_fn=functools.partial(fb.glyph_boxes_p, P=P))
    preds = {k: us.value_at(fc.feat_tuple(cells[k], bs, bl), gc.NCC_MIN) for k in st["gt"]}
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variants", required=True)
    a = ap.parse_args()
    T0 = time.time()
    variants = json.loads(a.variants)
    st = fc.load_setup()
    songs = st["songs"]
    tpl = gc.Templates(fc.TPL)
    rois = fc.load_rois(fc.WORK, songs)
    cat = {}
    with open(fc.CELLS47, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            cat[(r["video"], int(r["sample_idx"]), r["counter"])] = r["category"]

    base = run_variant(rois, songs, tpl, st, dict(fb.DEFAULT))
    rb, _ = fc.score(base, st, songs)
    print("base", {m: rb[m] for m in METRICS}, round(time.time() - T0, 1), flush=True)
    out_rows = [{"variant": "現行", **{m: rb[m] for m in METRICS}}]
    for v in variants:
        P = dict(fb.DEFAULT)
        for kk, vv in v.items():
            if kk != "name":
                P[kk] = tuple(vv) if kk == "thresholds" else vv
        p = run_variant(rois, songs, tpl, st, P)
        r, _ = fc.score(p, st, songs)
        c = collections.Counter()
        for k in st["gt"]:
            if not st["cert"][k]:
                continue
            g = st["gt"][k]
            nv, bv = p[k], base[k]
            if cat[k].startswith("(ii)"):
                grp = "しゃくり" if k[2] == "shakuri" else "それ以外"
                o = "読まない" if nv is None else ("正解" if nv == g else "誤り")
                c[f"(ii){grp}→{o}"] += 1
            else:
                was_bad = bv is not None and bv != g
                now_bad = nv is not None and nv != g
                if now_bad and not was_bad:
                    c["(ii)以外・新たに誤り"] += 1
                if bv == g and nv != g:
                    c["(ii)以外・正解でなくなった"] += 1
                if bv != g and nv == g:
                    c["(ii)以外・新たに正解"] += 1
        row = {"variant": v["name"], **{m: r[m] for m in METRICS}, **c}
        out_rows.append(row)
        print(v["name"], json.dumps(row, ensure_ascii=False), round(time.time() - T0, 1), flush=True)
    keys = []
    for r in out_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(a.out, f"B_relax_{a.tag}.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, 0 if k.startswith("(ii)") else "") for k in keys})
    print("DONE", round(time.time() - T0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
