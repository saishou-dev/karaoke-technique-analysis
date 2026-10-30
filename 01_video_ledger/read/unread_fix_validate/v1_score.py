#!/usr/bin/env python3
"""v1_score.py ＝ 検証1＝別の日の収録で、見本追加なし（基準）と k＝3（10/14 の4曲から作った追加見本30個）を採点（2026-10-30）

入力＝v1_extract.py の出力（<work>/gt.csv・input_summary.csv・<曲>/roi_<技法>.npy・frames.npy）。
採点＝材料47・49 と同じ関数（unread_sweep.score_condition・value_at・classify_cell）＝10/18 と同じ尺度。
  値＝v0.8e と同じ（しきい値0.75・安定判定6・帯消失ガード・減らない制約）。
  k＝3＝箱ごとに、追加見本30個での最良の類似度が現行見本での最良を真に上回る時だけ置き換える（材料49 と同じ）。
曲の組
  全曲＝verify_<S>.csv の全曲
  正解が揃った曲＝G1 の各技法の最終値が公式採点データ（verify の csv_*）と6技法すべて一致する曲（10/18 は完全一致の曲から選んだ）
帯の在区間（区分 (i) の判定だけに使う）＝見本追加なしの値に帯消失ガードを当てた区間（抽出列の上で求める・9/24 は v0.8e の全フレーム走査の記録が無いため）。
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
import fix_common as fc
import glyph_core as gc
import score_compare as sc
import unread_sweep as us

ROOT = fc.ROOT
COUNTERS = gc.COUNTERS
METRICS = ["acc_all", "acc_answered", "abstain", "misread", "final_hit", "final_total", "false_inc", "missed_inc",
           "cells", "answered", "correct", "misread_cells", "abstain_cells"]


def load_setup(work, verify, g1, stride=8):
    gt, cert = sc.load_gt(work)
    songs = sorted({k[0] for k in gt})
    frames = {s: np.load(os.path.join(work, s, "frames.npy")).tolist() for s in songs}
    fps = {}
    with open(os.path.join(work, "input_summary.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fps[r["video"]] = float(r["fps"])
    dam = {}
    with open(os.path.join(ROOT, "dataset", verify), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                dam[r["video"]] = {c: int(r["csv_" + c]) for c in COUNTERS}
    truth_ev = collections.defaultdict(list)
    g1max = collections.defaultdict(int)
    with open(os.path.join(ROOT, "dataset", g1), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                for _ in range(int(r["delta"])):
                    truth_ev[(r["video"], r["counter"])].append(int(r["frame"]))
                g1max[(r["video"], r["counter"])] = max(g1max[(r["video"], r["counter"])], int(r["count_value"]))
    g1_ok = [s for s in songs if all(g1max[(s, c)] == dam[s][c] for c in COUNTERS)]
    band_w = max(1, round(150.0 / stride))
    return dict(gt=gt, cert=cert, songs=songs, frames=frames, fps=fps, dam=dam, truth_ev=truth_ev,
                band_w=band_w, g1_ok=g1_ok, g1max=g1max, in_band=set())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--verify", required=True)
    ap.add_argument("--g1", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", default=ROOT + "/unread_fix_20261030/templates_k3_all4.npy")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T0 = time.time()
    st = load_setup(a.work, a.verify, a.g1)
    songs = st["songs"]
    tpl = gc.Templates(fc.TPL)
    ext = np.load(a.ext)
    ext_lab_d = json.load(open(os.path.splitext(a.ext)[0] + "_labels.json"))
    ext_lab = [ext_lab_d[str(i)] for i in range(len(ext))]
    flat = []
    for r_ in ext:
        v = r_.ravel() - r_.mean()
        flat.append(v / max(1e-6, np.linalg.norm(v)))
    ext_flat = np.stack(flat).astype(np.float32)
    # 箱ごとの（現行見本での最良の類似度・値）と（追加見本での最良の類似度・値）。曲ごとに読み込んで捨てる（メモリ節約）。
    cells, bs, bl, es, el = {}, [], [], [], []
    for s in songs:
        for c in COUNTERS:
            arr = np.load(os.path.join(a.work, s, f"roi_{c}.npy"))
            for i in range(arr.shape[0]):
                g = arr[i].max(axis=2)
                boxes = gc.glyph_boxes(g)
                idx = []
                if boxes is not None:
                    for b in boxes:
                        u = fc.unit(gc.norm_glyph(g, b))
                        if u is None:
                            bs.append(fc.FLAT); bl.append(None); es.append(fc.FLAT); el.append(None)
                        else:
                            s1 = tpl.reps_flat @ u
                            i1 = int(np.argmax(s1))
                            s2 = ext_flat @ u
                            i2 = int(np.argmax(s2))
                            bs.append(float(s1[i1])); bl.append(tpl.labels.get(i1))
                            es.append(float(s2[i2])); el.append(ext_lab[i2])
                        idx.append(len(bs) - 1)
                cells[(s, i, c)] = idx
            del arr
    print(f"[{a.tag}] songs={len(songs)} g1_ok={len(st['g1_ok'])} cells={len(cells)} boxes={len(bs)} "
          f"{time.time()-T0:.1f}s", flush=True)

    def preds_with(use_ext):
        out = {}
        for k, idx in cells.items():
            per = []
            for j in idx:
                s0, l0 = bs[j], bl[j]
                if s0 != fc.FLAT and use_ext and es[j] > s0:
                    s0, l0 = es[j], el[j]
                per.append((s0, l0, 0.0, None))
            out[k] = us.value_at((len(idx), per), gc.NCC_MIN)
        return out

    conds = {"基準（追加なし）": preds_with(False), "k=3（追加30個）": preds_with(True)}

    # 帯の在区間（区分 (i) 用）＝基準の値に帯消失ガードを当てた区間（抽出列の添字）
    base = conds["基準（追加なし）"]
    for s in songs:
        ns = len(st["frames"][s])
        vals = {c: [(i, base[(s, i, c)]) for i in range(ns) if base.get((s, i, c)) is not None] for c in COUNTERS}
        _v, (b0, b1) = gc.band_trim(vals, ns, st["band_w"])
        if b0 is not None:
            for i in range(b0, b1 + 1):
                for c in COUNTERS:
                    st["in_band"].add((s, i, c))

    rows, finals_rows = [], []
    for cname, p in conds.items():
        for grp, gsongs in [("全曲", songs), ("正解が揃った曲", st["g1_ok"])]:
            gt_sub = {k: v for k, v in st["gt"].items() if k[0] in gsongs}
            r, fin = us.score_condition(p, gt_sub, st["cert"], gsongs, st["frames"], st["fps"], st["dam"],
                                        st["truth_ev"], gc.STABLE_MIN, st["band_w"], st["in_band"])
            rows.append({"session": a.tag, "group": grp, "songs": len(gsongs), "cond": cname, **{m: r[m] for m in METRICS}})
            if grp == "全曲":
                for x in fin:
                    finals_rows.append([a.tag, cname, *x])
            print(f"[{a.tag}] {grp}({len(gsongs)}) {cname}: " + " ".join(f"{m}={r[m]}" for m in METRICS), flush=True)
        for s in songs:
            gt_sub = {k: v for k, v in st["gt"].items() if k[0] == s}
            r, _ = us.score_condition(p, gt_sub, st["cert"], [s], st["frames"], st["fps"], st["dam"],
                                      st["truth_ev"], gc.STABLE_MIN, st["band_w"], st["in_band"])
            rows.append({"session": a.tag, "group": "曲別:" + s, "songs": 1, "cond": cname,
                         "g1_ok": int(s in st["g1_ok"]), **{m: r[m] for m in METRICS}})

    # 区分ごとの行き先（正解確定・全曲）
    trans = collections.Counter()
    pk = conds["k=3（追加30個）"]
    for k in st["gt"]:
        if not st["cert"][k]:
            continue
        cat, _v = us.classify_cell(fc.feat_tuple(cells[k], bs, bl), st["gt"][k], gc.NCC_MIN, k in st["in_band"])
        v = pk[k]
        out = "読まない" if v is None else ("正解" if v == st["gt"][k] else "誤り")
        trans[(cat, out)] += 1

    keys = ["session", "group", "songs", "cond", "g1_ok"] + METRICS
    with open(os.path.join(a.out, f"V1_table_{a.tag}.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    with open(os.path.join(a.out, f"V1_finals_{a.tag}.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["session", "cond", "video", "counter", "final_pred", "final_dam", "hit"])
        w.writerows(finals_rows)
    with open(os.path.join(a.out, f"V1_transitions_{a.tag}.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["category_t075_base", "outcome_k3", "certain_cells"])
        for (c, o), n in sorted(trans.items()):
            w.writerow([c, o, n])
    json.dump({"tag": a.tag, "songs": songs, "g1_ok": st["g1_ok"],
               "g1_not_ok": {s: {c: [st["g1max"][(s, c)], st["dam"][s][c]] for c in COUNTERS
                                 if st["g1max"][(s, c)] != st["dam"][s][c]} for s in songs if s not in st["g1_ok"]},
               "ext": a.ext, "elapsed_sec": round(time.time() - T0, 1)},
              open(os.path.join(a.out, f"V1_meta_{a.tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{a.tag}] DONE {time.time()-T0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
