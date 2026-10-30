#!/usr/bin/env python3
"""v1_make_templates.py ＝ 追加見本 k＝3 を「4曲全体」で作り直す（2026-10-30）

材料49 の fix_a_templates.py と同じ手順を、評価曲を除かず 10/14 の4曲すべてを学習に使って1回だけ行う。
  正解確定コマのうち区分 (iii)（箱の数は合うが類似度0.75未満）で、最も似た見本の値が正解と同じコマから
  類似度 0.60以上0.75未満の箱（字形）を集め、正解の数字ごとに k-means（k＝3・seed 0）で分け、
  各群の平均画像を代表にする（既存の見本100個は残し、ここで作るのは「足す分」だけ）。
出力
  --out_npy（既定 unread_fix_20261030/templates_k3_all4.npy）と同名 _labels.json
  正本 templates/wao_20260816/cluster_reps_wao_ext.npy は読むだけ（書き換えない）。
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_common as fc
import glyph_core as gc

ROOT = fc.ROOT
LO, HI = 0.60, 0.75
K = 3


def kmeans_reps(units, glyphs, k, seed=0):
    import cv2
    k = min(k, len(units))
    if k <= 1:
        return [glyphs.mean(axis=0)], [len(units)]
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _c, lab, _ = cv2.kmeans(units.astype(np.float32), k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    reps, ns = [], []
    for ci in range(k):
        m = lab == ci
        if m.sum() == 0:
            continue
        reps.append(glyphs[m].mean(axis=0))
        ns.append(int(m.sum()))
    return reps, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_npy", default=ROOT + "/unread_fix_20261030/templates_k3_all4.npy")
    ap.add_argument("--copy_dir", default=os.path.dirname(os.path.abspath(__file__)) + "/result")
    a = ap.parse_args()
    T0 = time.time()
    st = fc.load_setup()
    songs = st["songs"]
    tpl = gc.Templates(fc.TPL)
    rois = fc.load_rois(fc.WORK, songs)
    cells, glyphs, units, bs, bl = fc.extract(rois, songs, tpl)
    mism, cnt, cats = fc.check_identity(cells, bs, bl, st)
    print("identity vs 材料47 cells_v08e.csv: mismatch=", mism, flush=True)
    if mism:
        print("STOP: 区分が材料47 と一致しない", flush=True)
        return 3
    js_by_lab = collections.defaultdict(list)
    for k in st["gt"]:
        if not st["cert"][k] or not cats[k].startswith("(iii)"):
            continue
        labs = [bl[j] for j in cells[k]]
        if None in labs:
            continue
        vv = labs[0] if len(labs) == 1 else labs[0] * 10 + labs[1]
        if vv != st["gt"][k]:
            continue
        for j in cells[k]:
            if bs[j] != fc.FLAT and LO <= bs[j] < HI:
                js_by_lab[bl[j]].append(j)
    reps, labs_new, rows = [], [], []
    for lab in sorted(js_by_lab):
        js = np.array(js_by_lab[lab])
        rr, ns = kmeans_reps(units[js], glyphs[js], K)
        for rep, n in zip(rr, ns):
            reps.append(rep)
            labs_new.append(lab)
            rows.append({"digit": lab, "glyphs_for_digit": int(len(js)), "glyphs_in_cluster": n})
    reps = np.stack(reps).astype(np.float32)
    meta = {"k": K, "range": [LO, HI], "train_songs": songs, "glyphs_total": int(sum(len(v) for v in js_by_lab.values())),
            "added_templates": int(len(reps)), "per_cluster": rows, "identity_mismatch_vs_material47": mism,
            "base_templates": fc.TPL, "elapsed_sec": round(time.time() - T0, 1)}
    for out in [a.out_npy, os.path.join(a.copy_dir, "templates_k3_all4.npy")]:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, reps)
        json.dump({str(i): l for i, l in enumerate(labs_new)},
                  open(os.path.splitext(out)[0] + "_labels.json", "w"), ensure_ascii=False)
    json.dump(meta, open(os.path.join(a.copy_dir, "templates_k3_all4_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("added", len(reps), "glyphs", meta["glyphs_total"], rows, flush=True)
    print("DONE", round(time.time() - T0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
