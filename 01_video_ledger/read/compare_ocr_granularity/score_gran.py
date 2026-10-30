#!/usr/bin/env python3
"""score_gran.py = 切り出し粒度ごとの予測を同じ尺度で採点する（2026-10-29）

測る値（指示の5つ）
  (1) 値を出したコマのうち正しい割合   acc_answered
  (2) 被覆率（値を出せたコマの割合）   coverage
  (3) 曲末一致（n/24）                 final_hit
  (4) 1,000コマあたりの処理時間（秒）  sec_per_1000_cells
  (5) 過剰読み＝偽の増分の件数         false_inc
加えて
  取り落とし missed_inc／誤読率 misread／2桁だけの正解率／場所外の読み stray（条件A）
  「1が11に見える」型の誤りの件数（下記4種）

「1が11に見える」型の数え方
  e_1to11      … 正解1・予測11（先生の指摘そのもの）
  e_dup        … 正解が1桁 d・予測が dd（11,22,33,… 桁の重複）
  e_more_digit … 予測の桁数 > 正解の桁数（桁の水増し全体）
  e_less_digit … 予測の桁数 < 正解の桁数（桁の取り落とし全体）

後処理は 2026-10-18 と同一（帯消失ガード→6コマ安定→減らない制約）。
帯消失ガードの移動平均の半幅だけ、間引きに合わせて 150コマ → round(150/(8*sub)) 標本に換算する。
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glyph_core as gc   # noqa: E402

COUNTERS = gc.COUNTERS
MATCH_SEC = 1.0
CONDS = ["A", "B", "C", "C2"]
CAND_ORDER = ["ncc", "nn_l2", "tess_psm7", "tess_psm8", "easyocr_rec",
              "easyocr_readtext", "paddle_rec", "paddle_full",
              "easyocr_rec_conf90", "paddle_rec_conf90"]
LABEL = {
    "ncc": "現行＝自作の見本照合（NCC・v0.8e）",
    "nn_l2": "対照＝生ピクセル最近傍（正規化なし・棄権なし）",
    "tess_psm7": "Tesseract 5.5.0（psm 7・前処理あり）",
    "tess_psm8": "Tesseract 5.5.0（psm 8・前処理あり）",
    "easyocr_rec": "EasyOCR 1.7.2（認識のみ）",
    "easyocr_readtext": "EasyOCR 1.7.2（検出＋認識）",
    "paddle_rec": "PaddleOCR 3.7.0（認識のみ）",
    "paddle_full": "PaddleOCR 3.7.0（検出＋認識の一式）",
    "easyocr_rec_conf90": "EasyOCR（認識のみ・確信度0.90で足切り）",
    "paddle_rec_conf90": "PaddleOCR（認識のみ・確信度0.90で足切り）",
}


def load_gt(work):
    gt, cert = {}, {}
    with open(os.path.join(work, "gt.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            k = (r["video"], int(r["sample_idx"]), r["counter"])
            gt[k] = int(r["gt_value"])
            cert[k] = r["certain"] == "1"
    return gt, cert


def load_pred_set(pdir, cond, cand, conf_min=None):
    """分割実行した予測をまとめて読む。処理時間も足し合わせる。"""
    pats = sorted(glob.glob(os.path.join(pdir, "pred_%s_%s.csv" % (cond, cand))) +
                  glob.glob(os.path.join(pdir, "pred_%s_%s.shard*of*.csv" % (cond, cand))))
    if not pats:
        return None, None
    pred = {}
    for p in pats:
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                k = (r["video"], int(r["sample_idx"]), r["counter"])
                v = r["pred"]
                if v == "":
                    pred[k] = None
                elif (conf_min is not None and r.get("conf", "") not in ("", None)
                      and float(r["conf"]) < conf_min):
                    pred[k] = None
                else:
                    pred[k] = int(v)
    sec = 0.0
    cells = 0
    stray = 0
    imgs = 0
    crash = 0
    for p in pats:
        tj = p.replace("pred_", "time_").replace(".csv", ".json")
        if os.path.exists(tj):
            m = json.load(open(tj))
            sec += float(m.get("seconds", 0))
            cells += int(m.get("cells", 0))
            stray += int(m.get("stray", 0) or 0)
            imgs += int(m.get("images", 0) or 0)
            crash += int(m.get("crash_cells", 0) or 0)
    meta = {"seconds": sec, "timed_cells": cells, "stray": stray,
            "images": imgs, "crash_cells": crash,
            "sec_per_1000_cells": round(1000.0 * sec / cells, 3) if cells else ""}
    return pred, meta


def expand_events(acc, frames):
    out = []
    for i, v, d in acc:
        for _ in range(max(0, d)):
            out.append(frames[i] if i < len(frames) else frames[-1])
    return out


def match_events(pred_frames, true_frames, tol):
    used = [False] * len(true_frames)
    fp = 0
    for pf in sorted(pred_frames):
        best, bestd = -1, None
        for j, tf in enumerate(true_frames):
            if used[j]:
                continue
            d = abs(tf - pf)
            if d <= tol and (bestd is None or d < bestd):
                bestd, best = d, j
        if best < 0:
            fp += 1
        else:
            used[best] = True
    return fp, sum(1 for u in used if not u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/home/toshiaki/ocr_gran_20261029/work_b")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/mnt/g/研究アーカイブ/成果物/動画解析")
    ap.add_argument("--session", default="20261014")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--tag", default="sub4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gt, cert = load_gt(a.work)
    songs = sorted({k[0] for k in gt})
    frames = {s: np.load(os.path.join(a.work, s, "frames.npy")).tolist() for s in songs}
    fps = {}
    with open(os.path.join(a.work, "input_summary.csv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            fps[r["video"]] = float(r["fps"])

    dam = {}
    with open(os.path.join(a.root, "dataset", "verify_%s_wao.csv" % a.session),
              encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                dam[r["video"]] = {c: int(r["csv_" + c]) for c in COUNTERS}

    truth_ev = collections.defaultdict(list)
    with open(os.path.join(a.root, "dataset", "G1_events_%s_wao.csv" % a.session),
              encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in songs:
                for _ in range(int(r["delta"])):
                    truth_ev[(r["video"], r["counter"])].append(int(r["frame"]))

    # 採点に使うコマ列（間引き後）
    sel = {s: [i for i in range(len(frames[s])) if i % a.sub == 0] for s in songs}
    pos = {s: {i: j for j, i in enumerate(sel[s])} for s in songs}
    certain_keys = [k for k in gt if cert[k] and k[1] % a.sub == 0]
    band_w = max(1, round(150.0 / (a.stride * a.sub)))

    rows = []
    errrows = []
    percounter = []
    finals_rows = []
    for cond in CONDS:
        for cand in CAND_ORDER:
            base = cand.replace("_conf90", "")
            cmin = 0.90 if cand.endswith("_conf90") else None
            pr, meta = load_pred_set(a.pred, cond, base, cmin)
            if pr is None:
                continue
            own = [k for k in certain_keys if k in pr]
            if not own:
                continue
            partial = len(own) < 0.9 * len(certain_keys)
            keys = own
            n = ans = cor = 0
            n2 = cor2 = a2 = 0
            e = collections.Counter()
            bycnt = collections.Counter()
            for k in keys:
                g = gt[k]
                v = pr.get(k)
                n += 1
                bycnt[(k[2], "n")] += 1
                if v is None:
                    continue
                ans += 1
                ok = (v == g)
                if ok:
                    cor += 1
                    bycnt[(k[2], "cor")] += 1
                if g >= 10:
                    n2 += 1
                    a2 += 1
                    cor2 += ok
                if ok:
                    continue
                sg, sv = str(g), str(v)
                if g == 1 and v == 11:
                    e["e_1to11"] += 1
                if len(sg) == 1 and sv == sg * 2:
                    e["e_dup"] += 1
                if len(sv) > len(sg):
                    e["e_more_digit"] += 1
                elif len(sv) < len(sg):
                    e["e_less_digit"] += 1
            for k in keys:
                if gt[k] >= 10 and pr.get(k) is None:
                    n2 += 1
            for cc in COUNTERS:
                percounter.append([cond, cand, cc, bycnt[(cc, "n")], bycnt[(cc, "cor")],
                                   round(100.0 * bycnt[(cc, "cor")] /
                                         max(1, bycnt[(cc, "n")]), 2)])

            hit = fp_tot = fn_tot = 0
            if not partial:
                for s in songs:
                    nsamp = len(sel[s])
                    vals = {cc: [(pos[s][i], pr[(s, i, cc)]) for i in sel[s]
                                 if pr.get((s, i, cc)) is not None] for cc in COUNTERS}
                    vals, _ = gc.band_trim(vals, nsamp, band_w)
                    tol = MATCH_SEC * fps[s]
                    fr = [frames[s][i] for i in sel[s]]
                    for cc in COUNTERS:
                        acc, _ = gc.detect_increments(vals[cc])
                        fin = acc[-1][1] if acc else 0
                        ok_fin = fin == dam[s][cc]
                        hit += ok_fin
                        finals_rows.append([cond, cand, s, cc, fin, dam[s][cc],
                                            1 if ok_fin else 0])
                        pf = expand_events(acc, fr)
                        f_, m_ = match_events(pf, truth_ev[(s, cc)], tol)
                        fp_tot += f_
                        fn_tot += m_

            rows.append({
                "cond": cond, "cand": cand, "label": LABEL.get(cand, cand),
                "cells": n, "answered": ans,
                "acc_answered": round(100.0 * cor / max(1, ans), 2),
                "coverage": round(100.0 * ans / max(1, n), 2),
                "acc_all": round(100.0 * cor / max(1, n), 2),
                "misread": round(100.0 * (ans - cor) / max(1, n), 2),
                "acc_2digit_answered": round(100.0 * cor2 / max(1, a2), 2),
                "cells_2digit": n2, "answered_2digit": a2,
                "final_hit": "—" if partial else hit,
                "final_total": "—" if partial else len(songs) * len(COUNTERS),
                "false_inc": "—" if partial else fp_tot,
                "missed_inc": "—" if partial else fn_tot,
                "sec_per_1000_cells": meta["sec_per_1000_cells"],
                "seconds": round(meta["seconds"], 1),
                "images": meta["images"], "stray": meta["stray"],
                "crash_cells": meta["crash_cells"],
                "e_1to11": e["e_1to11"], "e_dup": e["e_dup"],
                "e_more_digit": e["e_more_digit"], "e_less_digit": e["e_less_digit"],
            })
            errrows.append([cond, cand, n, ans, e["e_1to11"], e["e_dup"],
                            e["e_more_digit"], e["e_less_digit"]])
            print("%-3s %-20s n=%6d cov=%6.2f%% acc=%6.2f%% final=%s FP=%s "
                  "1->11=%d dup=%d more=%d less=%d t=%s"
                  % (cond, cand, n, rows[-1]["coverage"], rows[-1]["acc_answered"],
                     rows[-1]["final_hit"], rows[-1]["false_inc"],
                     e["e_1to11"], e["e_dup"], e["e_more_digit"], e["e_less_digit"],
                     meta["sec_per_1000_cells"]), flush=True)

    if not rows:
        print("no predictions found")
        return 1
    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "compare_gran_%s.csv" % a.tag), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "errtypes_%s.csv" % a.tag), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cond", "cand", "cells", "answered", "e_1to11", "e_dup",
                    "e_more_digit", "e_less_digit"])
        w.writerows(errrows)
    with open(os.path.join(a.out, "per_counter_%s.csv" % a.tag), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cond", "cand", "counter", "cells", "correct", "acc_pct"])
        w.writerows(percounter)
    with open(os.path.join(a.out, "finals_%s.csv" % a.tag), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cond", "cand", "video", "counter", "final_pred", "final_dam", "hit"])
        w.writerows(finals_rows)
    json.dump({"certain_cells_scored": len(certain_keys), "sub": a.sub,
               "stride": a.stride, "band_w_samples": band_w, "songs": songs,
               "dam_final": dam},
              open(os.path.join(a.out, "scoring_meta_%s.json" % a.tag), "w"),
              ensure_ascii=False, indent=1)
    print("wrote compare_gran_%s.csv / errtypes_%s.csv / per_counter_%s.csv"
          % (a.tag, a.tag, a.tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
