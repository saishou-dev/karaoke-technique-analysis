#!/usr/bin/env python3
"""unread_sweep.py ＝ 現行の見本照合（v0.8e）が「読まない」と答えたコマの内訳と、判定条件の掃引（2026-10-30）

目的
  10/18 比較で「読まない」率 52.34% だった現行手法（自作の見本照合＝NCC・v0.8e）について、
  (1) 「読まない」の原因を glyph_core の処理順に沿って分ける（内訳表）
  (2) 見本照合のしきい値（NCC_MIN）× 安定判定フレーム数（STABLE_MIN）を掃引し、
      「読まない」率と誤読率のつり合いを 10/18 と同じ採点で測る（掃引表）

入力（10/18 と同一・読み取り専用）
  --work  ocr_compare_20261018/work（gt.csv・input_summary.csv・<曲>/roi_<技法>.npy・frames.npy）
  見本    templates/wao_20260816/cluster_reps_wao_ext.npy（＋_labels.json・null＝数字でない見本）
  帯の在区間  results_wao_20261014/<曲>/read_log.txt の「band: present a..b」（全コマ走査の正式結果）

やり方
  1コマ×1技法ごとに、v0.8e と同じ glyph_boxes で箱を切り、箱ごとに全見本との NCC を1回だけ計算して
  「最も似た見本の番号・類似度」をキャッシュする。しきい値 t での答えは
     箱なし → 読まない ／ どれかの箱の最大類似度 < t → 読まない ／ 最も似た見本が数字でない → 読まない
     それ以外 → 1桁 or 2桁の値
  で決まり、glyph_core.read_value_ncc と完全に同じ（t=0.75 で 10/18 の pred_ncc.csv と全件一致を実測する）。
  後処理（帯消失ガード＋安定判定＋減らない制約）と採点は score_compare.py と同一の関数・同一の式を使う。

使い方: python unread_sweep.py --work <work> --out <out> [--ref_pred <pred_ncc.csv>]
"""
import argparse
import collections
import csv
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyph_core as gc          # 10/18 の参照実装の写し（v0.8e と同一）
import score_compare as sc       # 10/18 の採点の写し（load_gt・expand_events・match_events を流用）

COUNTERS = gc.COUNTERS
NCC_GRID = [round(0.75 + 0.02 * k, 2) for k in range(-5, 6)]   # 0.65〜0.85
STABLE_GRID = [4, 5, 6, 8]
CUR_T, CUR_S = gc.NCC_MIN, gc.STABLE_MIN

FLAT = -2.0   # 箱の中が一様で NCC が定義できない（v0.8e では None＝読まない）


def extract_features(work, songs, tpl):
    """各セルの箱の数と、箱ごとの（最大類似度・最も似た見本の値 or None）を返す。"""
    feat = {}
    t0 = time.perf_counter()
    for s in songs:
        for c in COUNTERS:
            arr = np.load(os.path.join(work, s, f"roi_{c}.npy"))
            for i in range(arr.shape[0]):
                g = arr[i].max(axis=2)
                boxes = gc.glyph_boxes(g)
                if boxes is None:
                    feat[(s, i, c)] = (0, [])
                    continue
                per = []
                for b in boxes:
                    gl = gc.norm_glyph(g, b)
                    v = gl.ravel() - gl.mean()
                    n = np.linalg.norm(v)
                    if n < 1e-6:
                        per.append((FLAT, None, FLAT, None))
                        continue
                    scores = tpl.reps_flat @ (v / n)
                    bi = int(np.argmax(scores))
                    # 参考＝数字の見本だけに限った最良（「数字でない見本」に負けた箱の診断用）
                    dig = [k for k in range(len(scores)) if tpl.labels.get(k) is not None]
                    bd = max(dig, key=lambda k: scores[k])
                    per.append((float(scores[bi]), tpl.labels.get(bi), float(scores[bd]), tpl.labels.get(bd)))
                feat[(s, i, c)] = (len(boxes), per)
    return feat, time.perf_counter() - t0


def value_at(f, t):
    nb, per = f
    if nb == 0:
        return None
    digits = []
    for sc_, lab, _sd, _ld in per:
        if sc_ == FLAT or sc_ < t:
            return None
        if lab is None:
            return None
        digits.append(lab)
    return digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]


def band_present(root, results, song):
    p = os.path.join(root, results, song, "read_log.txt")
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"band: present (\d+)\.\.(\d+)", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def classify_cell(f, gtv, t, in_band):
    """内訳の分類（glyph_core の処理順＝①箱を切る→②箱ごとに見本照合→③値にする）。"""
    v = value_at(f, t)
    if v is not None:
        return ("値あり・正解" if v == gtv else "値あり・誤り"), v
    if not in_band:
        return "(i) 表示なし（帯の在区間の外）", None
    nb, per = f
    if nb == 0:
        return "(ii) 箱が0個", None
    ndig = 1 if gtv < 10 else 2
    if nb != ndig:
        return ("(iv-a) 箱1個・正解2桁" if nb == 1 else "(iv-b) 箱2個・正解1桁"), None
    if any(p[0] == FLAT or p[0] < t for p in per):
        return "(iii) 箱の数は合う・最大類似度がしきい値未満", None
    return "(v) 箱の数は合う・類似度は届くが最も似た見本が「数字でない」見本", None


CAT_ORDER = ["値あり・正解", "値あり・誤り",
             "(i) 表示なし（帯の在区間の外）", "(ii) 箱が0個",
             "(iii) 箱の数は合う・最大類似度がしきい値未満",
             "(iv-a) 箱1個・正解2桁", "(iv-b) 箱2個・正解1桁",
             "(v) 箱の数は合う・類似度は届くが最も似た見本が「数字でない」見本"]


def score_condition(preds, gt, cert, songs, frames, fps, dam, truth_ev, stable_min, band_w, in_band_keys):
    """score_compare.main と同じ式（A コマ単位＋B 軌跡の確定）。"""
    certain_keys = [k for k in gt if cert[k]]
    n = ans = cor = 0
    n2 = a2 = cor2 = 0
    nb = ansb = 0
    for k in certain_keys:
        g = gt[k]
        v = preds.get(k)
        n += 1
        ans += v is not None
        ok = (v is not None and v == g)
        cor += ok
        if g >= 10:
            n2 += 1
            a2 += v is not None
            cor2 += ok
        if k in in_band_keys:
            nb += 1
            ansb += v is not None
    hit = fp_tot = fn_tot = 0
    finals = []
    for s in songs:
        nsamp = len(frames[s])
        vals = {cc: [(i, preds[(s, i, cc)]) for i in range(nsamp)
                     if preds.get((s, i, cc)) is not None] for cc in COUNTERS}
        vals, _ = gc.band_trim(vals, nsamp, band_w)
        tol = sc.MATCH_SEC * fps[s]
        for cc in COUNTERS:
            acc, _ = gc.detect_increments(vals[cc], stable_min=stable_min)
            fin = acc[-1][1] if acc else 0
            okf = fin == dam[s][cc]
            hit += okf
            finals.append((s, cc, fin, dam[s][cc], int(okf)))
            pf = sc.expand_events(acc, frames[s])
            fp, fn = sc.match_events(pf, truth_ev[(s, cc)], tol)
            fp_tot += fp
            fn_tot += fn
    return {
        "cells": n, "answered": ans, "correct": cor, "misread_cells": ans - cor, "abstain_cells": n - ans,
        "acc_all": round(100.0 * cor / max(1, n), 2),
        "acc_answered": round(100.0 * cor / max(1, ans), 2),
        "abstain": round(100.0 * (n - ans) / max(1, n), 2),
        "misread": round(100.0 * (ans - cor) / max(1, n), 2),
        "acc_2digit_answered": round(100.0 * cor2 / max(1, a2), 2),
        "abstain_in_band": round(100.0 * (nb - ansb) / max(1, nb), 2),
        "final_hit": hit, "final_total": len(songs) * len(COUNTERS),
        "false_inc": fp_tot, "missed_inc": fn_tot,
    }, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/mnt/g/研究アーカイブ/成果物/動画解析")
    ap.add_argument("--results", default="results_wao_20261014")
    ap.add_argument("--session", default="20261014")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--ref_pred", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T0 = time.time()

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
                dam[r["video"]] = {c: int(r["csv_" + c]) for c in COUNTERS}
    truth_ev = collections.defaultdict(list)
    with open(os.path.join(a.root, "dataset", f"G1_events_{a.session}_wao.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                for _ in range(int(r["delta"])):
                    truth_ev[(r["video"], r["counter"])].append(int(r["frame"]))
    band = {s: band_present(a.root, a.results, s) for s in songs}
    in_band_keys = set()
    for k in gt:
        s, i, _c = k
        bp = band[s]
        if bp and bp[0] <= frames[s][i] <= bp[1]:
            in_band_keys.add(k)

    tpl = gc.Templates(os.path.join(a.root, "templates", "wao_20260816", "cluster_reps_wao_ext.npy"))
    feat, tfeat = extract_features(a.work, songs, tpl)
    print(f"features: {len(feat)} cells in {tfeat:.1f}s", flush=True)

    # ---- 同一性：t=0.75 の答えが 10/18 の pred_ncc.csv と全件一致するか ----
    ident = {"checked": 0, "mismatch": 0}
    if a.ref_pred and os.path.exists(a.ref_pred):
        ref = sc.load_pred(a.ref_pred)
        for k, rv in ref.items():
            ident["checked"] += 1
            if value_at(feat[k], CUR_T) != rv:
                ident["mismatch"] += 1
        print(f"identity vs pred_ncc.csv: checked={ident['checked']} mismatch={ident['mismatch']}", flush=True)
        if ident["mismatch"]:
            print("STOP: cached reader differs from 10/18 pred_ncc.csv", flush=True)
            return 3

    # ---- 1) 内訳（現行条件 t=0.75） ----
    cnt_all = collections.Counter()
    cnt_cert = collections.Counter()
    by_counter = collections.Counter()
    iii_hist = collections.Counter()      # (iii) の最小の箱の最大類似度の分布
    iii_nearest_ok = 0
    iv_lowsim = collections.Counter()
    v_digit_score = []
    rows_cells = []
    for k in sorted(gt):
        f = feat[k]
        cat, v = classify_cell(f, gt[k], CUR_T, k in in_band_keys)
        cnt_all[cat] += 1
        if cert[k]:
            cnt_cert[cat] += 1
            by_counter[(k[2], cat)] += 1
            if cat.startswith("(iii)"):
                m = min(p[0] for p in f[1])
                lo = max(0.0, np.floor(m * 20) / 20)
                iii_hist[f"{lo:.2f}〜{lo + 0.05:.2f}"] += 1
                labs = [p[1] for p in f[1]]
                if None not in labs:
                    vv = labs[0] if len(labs) == 1 else labs[0] * 10 + labs[1]
                    iii_nearest_ok += (vv == gt[k])
            if cat.startswith("(iv"):
                iv_lowsim["しきい値未満の箱を含む" if any(p[0] == FLAT or p[0] < CUR_T for p in f[1]) else "全箱しきい値以上"] += 1
            if cat.startswith("(v)"):
                v_digit_score.append(min(p[2] for p in f[1]))
        nb, per = f
        rows_cells.append([k[0], k[1], k[2], gt[k], int(cert[k]), int(k in in_band_keys), nb,
                           ";".join(f"{p[0]:.4f}" for p in per), ";".join("" if p[1] is None else str(p[1]) for p in per),
                           "" if v is None else v, cat])
    with open(os.path.join(a.out, "cells_v08e.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["video", "sample_idx", "counter", "gt_value", "certain", "in_band", "nbox",
                    "box_best_ncc", "box_best_label", "pred_t075", "category"])
        w.writerows(rows_cells)
    ncert = sum(cnt_cert.values())
    nall = sum(cnt_all.values())
    with open(os.path.join(a.out, "breakdown_v08e.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["category", "certain_cells", "certain_pct", "all_cells", "all_pct"])
        for c in CAT_ORDER:
            w.writerow([c, cnt_cert[c], round(100.0 * cnt_cert[c] / ncert, 2), cnt_all[c], round(100.0 * cnt_all[c] / nall, 2)])
        w.writerow(["合計", ncert, 100.0, nall, 100.0])
    with open(os.path.join(a.out, "breakdown_by_counter.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["counter"] + CAT_ORDER)
        for cc in COUNTERS:
            w.writerow([cc] + [by_counter[(cc, c)] for c in CAT_ORDER])
    diag = {
        "iii_min_box_ncc_hist": dict(sorted(iii_hist.items())),
        "iii_nearest_label_equals_gt": iii_nearest_ok,
        "iv_sim_split": dict(iv_lowsim),
        "v_count": len(v_digit_score),
        "v_best_digit_ncc_median": float(np.median(v_digit_score)) if v_digit_score else None,
        "band_present": {s: band[s] for s in songs},
        "in_band_cells_certain": sum(1 for k in in_band_keys if cert[k]),
        "identity_vs_pred_ncc": ident,
        "feature_seconds": round(tfeat, 1),
    }
    json.dump(diag, open(os.path.join(a.out, "breakdown_diag.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for c in CAT_ORDER:
        print(f"  {c}: certain={cnt_cert[c]} ({100.0*cnt_cert[c]/ncert:.2f}%) all={cnt_all[c]}", flush=True)
    print(json.dumps(diag, ensure_ascii=False), flush=True)

    # ---- 2) 掃引（しきい値11段×安定判定4段＝44通り） ----
    band_w = max(1, round(150.0 / a.stride))
    rows = []
    all_finals = []
    for t in NCC_GRID:
        preds = {k: value_at(f, t) for k, f in feat.items()}
        for s_min in STABLE_GRID:
            r, fin = score_condition(preds, gt, cert, songs, frames, fps, dam, truth_ev, s_min, band_w, in_band_keys)
            r = {"ncc_min": t, "stable_min": s_min, "current": int(t == CUR_T and s_min == CUR_S), **r}
            rows.append(r)
            for x in fin:
                all_finals.append([t, s_min, *x])
            print(f"t={t:.2f} s={s_min} acc_all={r['acc_all']:6.2f} acc_ans={r['acc_answered']:6.2f} "
                  f"abstain={r['abstain']:6.2f} misread={r['misread']:5.2f} final={r['final_hit']}/24 "
                  f"FP={r['false_inc']} FN={r['missed_inc']}", flush=True)
    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "sweep_table.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "sweep_finals.csv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["ncc_min", "stable_min", "video", "counter", "final_pred", "final_dam", "hit"])
        w.writerows(all_finals)

    # ---- 3) 選定 ----
    cur = [r for r in rows if r["current"]][0]
    ok_mis = [r for r in rows if r["misread_cells"] <= cur["misread_cells"]]
    best_a = min(ok_mis, key=lambda r: (r["abstain_cells"], r["false_inc"] + r["missed_inc"]))
    ok_fin = [r for r in rows if r["final_hit"] == r["final_total"]]
    best_b = min(ok_fin, key=lambda r: (r["abstain_cells"], r["misread_cells"])) if ok_fin else None
    sel = {"current": cur, "rule_a": "誤読件数が現行以下（≦{}件＝{}%）の中で読まない件数が最小（同数なら偽の増分＋取り落としが小さい方）".format(cur["misread_cells"], cur["misread"]),
           "best_a": best_a,
           "rule_b": "曲末一致24/24の中で読まない件数が最小（同数なら誤読件数が小さい方）",
           "best_b": best_b, "n_final24": len(ok_fin),
           "final24_conditions": [(r["ncc_min"], r["stable_min"]) for r in ok_fin],
           "elapsed_sec": round(time.time() - T0, 1)}
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("SELECT", json.dumps(sel, ensure_ascii=False), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
