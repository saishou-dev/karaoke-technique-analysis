#!/usr/bin/env python3
"""score_compare.py ＝ 各手法の予測を同じ尺度で採点して比較表を作る（2026-10-18）

測る値（全候補で同じ・入力も同じ）
  A コマ単位（1コマ×1カウンター＝1件）
     正解率(全件)   ＝ 値が一致した件数 ÷ 正解が確定している全件（「読まない」は不正解に数える）
     正解率(回答分) ＝ 値が一致した件数 ÷ 値を出した件数
     「読まない」率 ＝ 値を出さなかった件数 ÷ 全件
     誤読率         ＝ 値を出したが違った件数 ÷ 全件
     1桁／2桁別の正解率（正解値が10未満／10以上で分ける）
  B 軌跡の確定（6コマ安定＋減らない制約＋帯消失ガード）を各候補の出力にかけた後
     曲末一致       ＝ 曲末の値が公式採点データ（DAM の採点CSV）と一致したカウンター数 ／ 対象24カウンター
     偽の増分       ＝ 正解の加点に ±1.0 秒で対応づかなかった候補側の加点の件数
     取り落とし     ＝ 候補側の加点に対応づかなかった正解の加点の件数
  C 小集合＝現行手法が値を出したコマだけでの正解率（「現行の強みは棄権だけではないか」の検証）
  D 処理時間（1,000コマあたり秒・同一マシン・各候補 専用4コア）

使い方: python score_compare.py --work <work> --pred <pred> --out <out>
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

COUNTERS = gc.COUNTERS
MATCH_SEC = 1.0

ORDER = ["ncc", "nn_l2", "tess_psm7", "tess_psm8", "tess_psm8_raw",
         "easyocr_rec", "easyocr_rec_prep", "easyocr_readtext", "easyocr_readtext_lr",
         "paddle_rec", "paddle_full"]
LABEL = {
    "ncc": "現行＝自作の見本照合（NCC・v0.8e）",
    "nn_l2": "対照＝生ピクセル最近傍（正規化なし・棄権なし）",
    "tess_psm7": "Tesseract 5.5.0（psm 7・前処理あり）",
    "tess_psm8": "Tesseract 5.5.0（psm 8・前処理あり）",
    "tess_psm8_raw": "Tesseract 5.5.0（psm 8・前処理なし）",
    "easyocr_rec": "EasyOCR 1.7.2（認識のみ・前処理なし）",
    "easyocr_rec_prep": "EasyOCR 1.7.2（認識のみ・前処理あり）",
    "easyocr_readtext": "EasyOCR 1.7.2（検出＋認識・最初の枠のみ）",
    "easyocr_readtext_lr": "EasyOCR 1.7.2（検出＋認識・複数枠は左から連結）",
    "paddle_rec": "PaddleOCR 3.7.0（認識のみ・PP-OCRv6_medium_rec）",
    "paddle_full": "PaddleOCR 3.7.0（検出＋認識の一式・部分集合）",
    "easyocr_rec_conf90": "EasyOCR（認識のみ・確信度0.90未満は読まない）",
    "paddle_rec_conf90": "PaddleOCR（認識のみ・確信度0.90未満は読まない）",
}


def load_gt(work):
    gt = {}
    cert = {}
    with open(os.path.join(work, "gt.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            k = (r["video"], int(r["sample_idx"]), r["counter"])
            gt[k] = int(r["gt_value"])
            cert[k] = r["certain"] == "1"
    return gt, cert


def load_pred(path, conf_min=None):
    pred = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            k = (r["video"], int(r["sample_idx"]), r["counter"])
            v = r["pred"]
            if v == "":
                pred[k] = None
            elif conf_min is not None and r.get("conf", "") not in ("", None) and float(r["conf"]) < conf_min:
                pred[k] = None
            else:
                pred[k] = int(v)
    return pred


def expand_events(acc, frames):
    """[(sample_idx, value, delta)] → 単位増分のコマ番号リスト（delta 分だけ並べる）"""
    out = []
    for i, v, d in acc:
        for _ in range(max(0, d)):
            out.append(frames[i] if i < len(frames) else frames[-1])
    return out


def match_events(pred_frames, true_frames, tol_frames):
    """近い順に1対1で対応づけ、未対応の件数を返す（偽の増分, 取り落とし）"""
    used = [False] * len(true_frames)
    fp = 0
    for pf in sorted(pred_frames):
        best = -1
        bestd = None
        for j, tf in enumerate(true_frames):
            if used[j]:
                continue
            d = abs(tf - pf)
            if d <= tol_frames and (bestd is None or d < bestd):
                bestd = d
                best = j
        if best < 0:
            fp += 1
        else:
            used[best] = True
    return fp, sum(1 for u in used if not u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/mnt/g/研究アーカイブ/成果物/動画解析")
    ap.add_argument("--session", default="20261014")
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gt, cert = load_gt(a.work)
    songs = sorted({k[0] for k in gt})
    frames = {s: np.load(os.path.join(a.work, s, "frames.npy")).tolist() for s in songs}
    fps = {}
    with open(os.path.join(a.work, "input_summary.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fps[r["video"]] = float(r["fps"])

    # 公式採点データ（DAM の採点CSV）の曲末値
    dam = {}
    with open(os.path.join(a.root, "dataset", f"verify_{a.session}_wao.csv"),
              encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                dam[r["video"]] = {c: int(r["csv_" + c]) for c in COUNTERS}

    # 技法イベント正解データ（真の加点時刻）
    truth_ev = collections.defaultdict(list)
    with open(os.path.join(a.root, "dataset", f"G1_events_{a.session}_wao.csv"),
              encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                for _ in range(int(r["delta"])):
                    truth_ev[(r["video"], r["counter"])].append(int(r["frame"]))

    # 候補の読み込み（存在するものだけ）＋確信度しきい値つきの派生候補
    cands = []
    preds = {}
    for c in ORDER:
        p = os.path.join(a.pred, f"pred_{c}.csv")
        if os.path.exists(p):
            cands.append(c)
            preds[c] = load_pred(p)
    for base, name in (("easyocr_rec", "easyocr_rec_conf90"), ("paddle_rec", "paddle_rec_conf90")):
        p = os.path.join(a.pred, f"pred_{base}.csv")
        if os.path.exists(p):
            cands.append(name)
            preds[name] = load_pred(p, conf_min=0.90)

    certain_keys = [k for k in gt if cert[k]]
    ncc_answered = set()
    if "ncc" in preds:
        ncc_answered = {k for k in certain_keys if preds["ncc"].get(k) is not None}

    band_w = max(1, round(150.0 / a.stride))
    rows = []
    percounter = []
    all_finals = []
    for c in cands:
        pr = preds[c]
        # 部分集合しか走らせていない候補（例＝PaddleOCR の一式は重いので間引いて実行）は、
        # その候補が実際に読んだコマだけで集計する。軌跡の確定は連続した列が要るので対象外にする。
        own = [k for k in certain_keys if k in pr]
        partial = len(own) < 0.9 * len(certain_keys)
        eval_keys = own if partial else certain_keys
        n = ans = cor = 0
        n1 = cor1 = n2 = cor2 = 0
        a1 = a2 = 0
        sub_n = sub_cor = 0
        gap_n = gap_ans = gap_cor = 0
        for k in eval_keys:
            g = gt[k]
            v = pr.get(k)
            n += 1
            if v is not None:
                ans += 1
            ok = (v is not None and v == g)
            if ok:
                cor += 1
            if g < 10:
                n1 += 1
                cor1 += ok
                a1 += v is not None
            else:
                n2 += 1
                cor2 += ok
                a2 += v is not None
            if k in ncc_answered:
                sub_n += 1
                sub_cor += ok
            else:
                gap_n += 1
                gap_ans += v is not None
                gap_cor += ok
        # 技法別
        bycnt = collections.Counter()
        for k in eval_keys:
            g = gt[k]
            v = pr.get(k)
            bycnt[(k[2], "n")] += 1
            if v is not None and v == g:
                bycnt[(k[2], "cor")] += 1
        for cc in COUNTERS:
            percounter.append([c, cc, bycnt[(cc, "n")], bycnt[(cc, "cor")],
                               round(100.0 * bycnt[(cc, "cor")] / max(1, bycnt[(cc, "n")]), 2)])

        # ---- B 軌跡の確定 ----
        hit = 0
        fp_tot = 0
        fn_tot = 0
        finals = {}
        for s in (() if partial else songs):
            nsamp = len(frames[s])
            vals = {cc: [(i, pr[(s, i, cc)]) for i in range(nsamp)
                         if pr.get((s, i, cc)) is not None] for cc in COUNTERS}
            vals, _ = gc.band_trim(vals, nsamp, band_w)
            tol = MATCH_SEC * fps[s]
            for cc in COUNTERS:
                acc, _ = gc.detect_increments(vals[cc])
                fin = acc[-1][1] if acc else 0
                finals[(s, cc)] = fin
                ok_fin = fin == dam[s][cc]
                if ok_fin:
                    hit += 1
                all_finals.append([c, s, cc, fin, dam[s][cc], 1 if ok_fin else 0])
                pf = expand_events(acc, frames[s])
                fp, fn = match_events(pf, truth_ev[(s, cc)], tol)
                fp_tot += fp
                fn_tot += fn
        tj = os.path.join(a.pred, f"time_{c.replace('_conf90', '')}.json")
        sec1000 = ""
        if os.path.exists(tj):
            sec1000 = json.load(open(tj))["sec_per_1000_cells"]
        rows.append({
            "cand": c, "label": LABEL.get(c, c),
            "cells": n, "answered": ans,
            "acc_all": round(100.0 * cor / max(1, n), 2),
            "acc_answered": round(100.0 * cor / max(1, ans), 2),
            "abstain": round(100.0 * (n - ans) / max(1, n), 2),
            "misread": round(100.0 * (ans - cor) / max(1, n), 2),
            "acc_1digit": round(100.0 * cor1 / max(1, n1), 2),
            "acc_2digit": round(100.0 * cor2 / max(1, n2), 2),
            "acc_1digit_answered": round(100.0 * cor1 / max(1, a1), 2),
            "acc_2digit_answered": round(100.0 * cor2 / max(1, a2), 2),
            "cells_2digit": n2, "answered_2digit": a2,
            "final_hit": "—" if partial else hit,
            "final_total": "—" if partial else len(songs) * len(COUNTERS),
            "false_inc": "—" if partial else fp_tot,
            "missed_inc": "—" if partial else fn_tot,
            "partial_subset": 1 if partial else 0,
            "acc_on_ncc_answered": round(100.0 * sub_cor / max(1, sub_n), 2),
            "on_ncc_abstained_cells": gap_n,
            "on_ncc_abstained_answer_rate": round(100.0 * gap_ans / max(1, gap_n), 2),
            "on_ncc_abstained_acc": round(100.0 * gap_cor / max(1, gap_n), 2),
            "on_ncc_abstained_acc_answered": round(100.0 * gap_cor / max(1, gap_ans), 2),
            "sec_per_1000_cells": sec1000,
        })
        print(f"{c:22s} n={n:6d}{'*' if partial else ' '} acc_all={rows[-1]['acc_all']:6.2f}% "
              f"ans={ans:6d} acc_ans={rows[-1]['acc_answered']:6.2f}% "
              f"1d={rows[-1]['acc_1digit_answered']:6.2f}% 2d={rows[-1]['acc_2digit_answered']:6.2f}% "
              f"final={rows[-1]['final_hit']}/{rows[-1]['final_total']} "
              f"FP={rows[-1]['false_inc']} FN={rows[-1]['missed_inc']} "
              f"sub={rows[-1]['acc_on_ncc_answered']:6.2f}% t={sec1000}", flush=True)

    with open(os.path.join(a.out, "finals.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cand", "video", "counter", "final_pred", "final_dam", "hit"])
        w.writerows(all_finals)
    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "compare_table.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "per_counter.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cand", "counter", "cells", "correct", "acc_pct"])
        w.writerows(percounter)
    json.dump({"certain_cells": len(certain_keys), "all_cells": len(gt),
               "songs": songs, "band_w_samples": band_w, "stride": a.stride,
               "dam_final": {s: dam[s] for s in songs}},
              open(os.path.join(a.out, "scoring_meta.json"), "w"), ensure_ascii=False, indent=1)
    print("wrote compare_table.csv / per_counter.csv / scoring_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
