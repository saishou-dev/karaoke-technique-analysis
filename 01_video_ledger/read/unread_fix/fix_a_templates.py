#!/usr/bin/env python3
"""fix_a_templates.py ＝ 候補A＝見本の追加を曲単位の交差検証で測る（2026-10-30）

やり方
  4曲のうち1曲を評価曲、残り3曲を学習曲にする（×4回）。
  学習曲の正解確定コマのうち、材料47 の区分 (iii)（箱の数は合うが類似度が0.75未満）で、
  最も似た見本の値が正解と同じコマから、類似度が 0.60以上0.75未満の箱（字形）を集める。
  正解の数字ごとに k-means（k＝1〜3・NCC と同じ平均0・長さ1のベクトルで分ける）で分け、
  各群の字形の平均画像を代表として見本に足す（既存の見本100個は残す）。
  評価曲を、しきい値0.75・安定判定6のまま 10/18 と同じ尺度で採点する。
  基準＝見本の追加なし（材料47 の現行の行）。
評価曲の箱は学習に1つも入らない（実行時に検査して記録する）。
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_common as fc
import glyph_core as gc
import unread_sweep as us

LO, HI = 0.60, 0.75
KS = [1, 2, 3]
METRICS = ["acc_all", "acc_answered", "abstain", "misread", "final_hit", "false_inc", "missed_inc",
           "cells", "answered", "correct", "misread_cells", "abstain_cells"]


def kmeans_reps(units, glyphs, k, seed=0):
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T0 = time.time()
    st = fc.load_setup()
    songs = st["songs"]
    tpl = gc.Templates(fc.TPL)
    rois = fc.load_rois(fc.WORK, songs)
    cells, glyphs, units, bs, bl = fc.extract(rois, songs, tpl)
    print(f"features: cells={len(cells)} boxes={len(bs)} {time.time()-T0:.1f}s", flush=True)

    mism, cnt, cats = fc.check_identity(cells, bs, bl, st)
    print("identity vs 材料47 cells_v08e.csv: mismatch=", mism, dict(cnt), flush=True)
    if mism:
        print("STOP: 区分が材料47 と一致しない", flush=True)
        return 3

    # 箱 → 曲
    box_song = np.empty(len(bs), dtype=object)
    for k, idx in cells.items():
        for j in idx:
            box_song[j] = k[0]

    def preds_with(extra_flat, extra_lab, keys):
        out = {}
        for k in keys:
            idx = cells[k]
            per = []
            for j in idx:
                s0, l0 = bs[j], bl[j]
                if s0 != fc.FLAT and extra_flat is not None:
                    sn = extra_flat @ units[j]
                    bi = int(np.argmax(sn))
                    if sn[bi] > s0:
                        s0, l0 = float(sn[bi]), extra_lab[bi]
                per.append((s0, l0, 0.0, None))
            out[k] = us.value_at((len(idx), per), gc.NCC_MIN)
        return out

    song_keys = {s: [k for k in st["gt"] if k[0] == s] for s in songs}
    base_preds = preds_with(None, None, list(st["gt"]))
    rows = []
    finals_rows = []
    # 基準（追加なし）
    r_all, _ = fc.score(base_preds, st, songs)
    rows.append({"cond": "基準（追加なし）", "k": 0, "eval_song": "4曲まとめ", **{m: r_all[m] for m in METRICS}})
    for s in songs:
        r, fin = fc.score(base_preds, st, [s])
        rows.append({"cond": "基準（追加なし）", "k": 0, "eval_song": s, **{m: r[m] for m in METRICS}})
        for x in fin:
            finals_rows.append(["基準", 0, *x])
    print("baseline all:", {m: r_all[m] for m in METRICS}, flush=True)

    added_rows = []
    trans = collections.Counter()
    leak = {}
    for kk in KS:
        pooled = {}
        for e in songs:
            train = [s for s in songs if s != e]
            js_by_lab = collections.defaultdict(list)
            for k in st["gt"]:
                if k[0] == e or not st["cert"][k] or not cats[k].startswith("(iii)"):
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
            all_js = [j for v in js_by_lab.values() for j in v]
            leak_n = int(sum(1 for j in all_js if box_song[j] == e))
            leak[f"k{kk}_{e}"] = {"train_songs": train, "boxes": len(all_js), "boxes_from_eval_song": leak_n}
            if leak_n:
                print("STOP: 評価曲の箱が学習に混ざった", e, flush=True)
                return 4
            reps, labs_new = [], []
            for lab in sorted(js_by_lab):
                js = np.array(js_by_lab[lab])
                rr, ns = kmeans_reps(units[js], glyphs[js], kk)
                for rep, n in zip(rr, ns):
                    reps.append(rep)
                    labs_new.append(lab)
                    added_rows.append([kk, e, lab, len(js), n])
            reps = np.stack(reps).astype(np.float32)
            np.save(os.path.join(a.out, f"A_added_k{kk}_eval_{e}.npy"), reps)
            json.dump({str(i): l for i, l in enumerate(labs_new)},
                      open(os.path.join(a.out, f"A_added_k{kk}_eval_{e}_labels.json"), "w"), ensure_ascii=False)
            flat = []
            for r_ in reps:
                v = r_.ravel() - r_.mean()
                flat.append(v / max(1e-6, np.linalg.norm(v)))
            flat = np.stack(flat).astype(np.float32)
            p = preds_with(flat, labs_new, song_keys[e])
            pooled.update(p)
            r, fin = fc.score(p, st, [e])
            rows.append({"cond": f"見本追加 k={kk}", "k": kk, "eval_song": e, "added_templates": len(reps),
                         "train_glyphs": len(all_js), **{m: r[m] for m in METRICS}})
            for x in fin:
                finals_rows.append([f"k={kk}", kk, *x])
            for k in song_keys[e]:
                if not st["cert"][k]:
                    continue
                v = p[k]
                out = "読まない" if v is None else ("正解" if v == st["gt"][k] else "誤り")
                trans[(kk, cats[k], out)] += 1
            print(f"k={kk} eval={e} added={len(reps)} glyphs={len(all_js)} abstain={r['abstain']} "
                  f"misread={r['misread']} acc_ans={r['acc_answered']} final={r['final_hit']}/6 "
                  f"FP={r['false_inc']} FN={r['missed_inc']}", flush=True)
        r_all, _ = fc.score(pooled, st, songs)
        rows.append({"cond": f"見本追加 k={kk}", "k": kk, "eval_song": "4曲まとめ", **{m: r_all[m] for m in METRICS}})
        print(f"k={kk} pooled:", {m: r_all[m] for m in METRICS}, flush=True)

    # 4回の平均（曲別の値の単純平均）
    mean_rows = []
    for kk in [0] + KS:
        sub = [r for r in rows if r["k"] == kk and r["eval_song"] != "4曲まとめ"]
        mean_rows.append({"cond": sub[0]["cond"], "k": kk, "eval_song": "4回の平均",
                          **{m: round(float(np.mean([r[m] for r in sub])), 2) for m in METRICS}})
    keys = ["cond", "k", "eval_song", "added_templates", "train_glyphs"] + METRICS
    with open(os.path.join(a.out, "A_table.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        for r in rows + mean_rows:
            w.writerow({k: r.get(k, "") for k in keys})
    with open(os.path.join(a.out, "A_added_templates.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["k", "eval_song", "digit", "glyphs_for_digit", "glyphs_in_cluster"])
        w.writerows(added_rows)
    with open(os.path.join(a.out, "A_transitions.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["k", "category_t075_base", "outcome_with_added", "certain_cells"])
        for (kk, c, o), n in sorted(trans.items()):
            w.writerow([kk, c, o, n])
    with open(os.path.join(a.out, "A_finals.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["cond", "k", "video", "counter", "final_pred", "final_dam", "hit"])
        w.writerows(finals_rows)
    json.dump({"identity_mismatch_vs_material47": mism, "category_counts_certain": dict(cnt),
               "leak_check": leak, "range": [LO, HI], "elapsed_sec": round(time.time() - T0, 1)},
              open(os.path.join(a.out, "A_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("DONE", round(time.time() - T0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
