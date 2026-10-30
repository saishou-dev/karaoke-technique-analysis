#!/usr/bin/env python3
"""fix_common.py ＝ 読み落とし対策（候補A・候補B）の共通部分（2026-10-30）

10/18 と同じ入力・同じ採点を、unread_sweep.py（材料47）の関数をそのまま使って用意する。
  load_setup  ＝ 正解・抽出列・fps・DAM の曲末値・正解の加点時刻・帯の在区間
  extract     ＝ 1コマ×1技法ごとに v0.8e と同じ glyph_boxes で箱を切り、箱ごとの
                 正規化した字形（28×40）・NCC 用の単位ベクトル・現行見本での最良類似度と値 を返す
  check_identity ＝ 現行見本・しきい値0.75 の区分が材料47 の cells_v08e.csv と全件一致するかを測る
入力は読み取り専用（ocr_compare_20261018/work・templates・dataset・results_wao_20261014）。
"""
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyph_core as gc        # 10/18 参照実装の写し（v0.8e と同一・変更なし）
import score_compare as sc     # 10/18 採点の写し（変更なし）
import unread_sweep as us      # 材料47 の内訳・採点の写し（変更なし）

ROOT = "/mnt/g/研究アーカイブ/成果物/動画解析"
WORK = ROOT + "/ocr_compare_20261018/work"
TPL = ROOT + "/templates/wao_20260816/cluster_reps_wao_ext.npy"
CELLS47 = ROOT + "/unread_sweep_20261030/result/cells_v08e.csv"
COUNTERS = gc.COUNTERS
FLAT = us.FLAT


def load_setup(work=WORK, root=ROOT, results="results_wao_20261014", session="20261014", stride=8):
    gt, cert = sc.load_gt(work)
    songs = sorted({k[0] for k in gt})
    frames = {s: np.load(os.path.join(work, s, "frames.npy")).tolist() for s in songs}
    fps = {}
    with open(os.path.join(work, "input_summary.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fps[r["video"]] = float(r["fps"])
    dam = {}
    with open(os.path.join(root, "dataset", f"verify_{session}_wao.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                dam[r["video"]] = {c: int(r["csv_" + c]) for c in COUNTERS}
    truth_ev = collections.defaultdict(list)
    with open(os.path.join(root, "dataset", f"G1_events_{session}_wao.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                for _ in range(int(r["delta"])):
                    truth_ev[(r["video"], r["counter"])].append(int(r["frame"]))
    band = {s: us.band_present(root, results, s) for s in songs}
    in_band = set()
    for k in gt:
        s, i, _c = k
        bp = band[s]
        if bp and bp[0] <= frames[s][i] <= bp[1]:
            in_band.add(k)
    band_w = max(1, round(150.0 / stride))
    return dict(gt=gt, cert=cert, songs=songs, frames=frames, fps=fps, dam=dam,
                truth_ev=truth_ev, in_band=in_band, band_w=band_w)


def load_rois(work, songs):
    return {(s, c): np.load(os.path.join(work, s, f"roi_{c}.npy")) for s in songs for c in COUNTERS}


def unit(gl):
    v = gl.ravel().astype(np.float32) - gl.mean()
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def extract(rois, songs, tpl, boxes_fn=gc.glyph_boxes):
    """cells[key] = list of box index（箱なし＝空リスト）。箱ごとの配列を返す。"""
    cells = {}
    glyphs, units, best_s, best_l = [], [], [], []
    for s in songs:
        for c in COUNTERS:
            arr = rois[(s, c)]
            for i in range(arr.shape[0]):
                g = arr[i].max(axis=2)
                boxes = boxes_fn(g)
                idx = []
                if boxes is not None:
                    for b in boxes:
                        gl = gc.norm_glyph(g, b)
                        u = unit(gl)
                        glyphs.append(gl.astype(np.float32))
                        if u is None:
                            units.append(np.zeros(gl.size, np.float32))
                            best_s.append(FLAT)
                            best_l.append(None)
                        else:
                            sc_ = tpl.reps_flat @ u
                            bi = int(np.argmax(sc_))
                            units.append(u)
                            best_s.append(float(sc_[bi]))
                            best_l.append(tpl.labels.get(bi))
                        idx.append(len(glyphs) - 1)
                cells[(s, i, c)] = idx
    return cells, np.stack(glyphs), np.stack(units), np.array(best_s, np.float64), best_l


def feat_tuple(idx, bs, bl):
    """unread_sweep の feat 形式（箱の数, [(類似度, 値, -, -)…]）に直す。"""
    return (len(idx), [(bs[j], bl[j], 0.0, None) for j in idx])


def check_identity(cells, bs, bl, st):
    """現行条件の区分・値が材料47 の cells_v08e.csv と全件一致するか。"""
    ref = {}
    with open(CELLS47, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            ref[(r["video"], int(r["sample_idx"]), r["counter"])] = (r["category"], r["pred_t075"])
    mism = 0
    cnt = collections.Counter()
    cats = {}
    for k in st["gt"]:
        cat, v = us.classify_cell(feat_tuple(cells[k], bs, bl), st["gt"][k], gc.NCC_MIN, k in st["in_band"])
        cats[k] = cat
        rc, rv = ref[k]
        if cat != rc or ("" if v is None else str(v)) != rv:
            mism += 1
        if st["cert"][k]:
            cnt[cat] += 1
    return mism, cnt, cats


def score(preds, st, songs):
    gt_sub = {k: v for k, v in st["gt"].items() if k[0] in songs}
    r, fin = us.score_condition(preds, gt_sub, st["cert"], songs, st["frames"], st["fps"], st["dam"],
                                st["truth_ev"], gc.STABLE_MIN, st["band_w"], st["in_band"])
    return r, fin
