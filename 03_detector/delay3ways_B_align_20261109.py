# -*- coding: utf-8 -*-
r"""案B＝順番の対応付け（2026-11-09 Opus・指示書 2026-11-09_1429 delay-3ways・学習しない）。
正解側の列＝カウンターの時刻 tc（引く前＝index の時刻＋1.6722秒・全技法共通の値なので1対1で戻せる）と技法。
候補側の列＝2通り。
  rule ＝規則（rule_detect_c4_20261105 の4技法・11/5 に dev で選ばれた数値）＝ハンマリングは候補なし。
  model＝今のモデル種0（c3_5tech_8ep_s0）の確率の山（pick と同じ山・境目＝11/5 手順8 の 8a の値）。確率の列＝probs/probs_base_all.npz（infer_all_delay3ways_20261109.py が書く）。
対応付け＝曲ごとに、順番を崩さない（正解の並びと候補の並びの両方で単調）・技法が一致・差 d＝tc−t候補 が 0.5〜3.5秒の窓、の組だけ許す。
  目的（辞書式）＝①対応した数が最大 ②その中で、隣り合う対応どうしの遅れの変化 Σ|d_i−d_前| が最小（遅れは曲の中でなめらか＝固定の値は使わない）。
  解き方＝動的計画（状態＝許される組・前の組は「正解も候補も前」の物）。同点は候補の早い方。
出す物＝対応できた割合（技法別）・件ごとの遅れ d の中央値と四分位（技法別）・1.6722 との差。
正解の作り直し（rule の時だけ）＝対応した件は t候補、未対応は index の時刻（1.6722秒を引いた時刻）のまま。208曲すべて（学習・調整・試験）。
使い方: python delay3ways_B_align_20261109.py --cand rule|model
出力＝out/delay_3ways_20261109/B/align_<cand>_20261109.csv（1件ずつ）・summary_<cand>_20261109.json・（rule）labels/index_B_rule_20261109.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, collections, csv, json, sys, time
import numpy as np
from common_c4_20261105 import ROOT, TYPES, JP, DEV, TEST, FR, peaks
import rule_detect_c4_20261105 as RD

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
SHIFT = 1.6722
IX = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
OUT = os.path.join(ROOT, "out", "delay_3ways_20261109")
RULE_P = {"shakuri": (2.0, 30, 0), "fall": (2.0, 15), "kobushi": (0.4,), "accent": (2.5, 5)}   # 11/5 手順6a の選ばれた数値
THR8A_BASE = {"shakuri": 0.8, "kobushi": 0.68, "fall": 0.15, "accent": 0.52, "hammering": 0.67}  # 11/5 手順8 の基準種0 の 8a
LO, HI = 0.5, 3.5
BIG = 1e6


def split_of(ses):
    return "test" if ses == TEST else ("dev" if ses == DEV else "train")


def cand_rule(video):
    S = RD.prep(video)
    out = []
    for c, p in RULE_P.items():
        f = {"shakuri": RD.det_shakuri, "fall": RD.det_fall, "kobushi": RD.det_kobushi, "accent": RD.det_accent}[c]
        out += [(t, c) for t in f(S, *p)]
    return sorted(out)


def align(truth, cand):
    """truth＝[(tc, 技法, 元の番号)]（tc の順）・cand＝[(t, 技法)]（t の順）。返す物＝{元の番号: 候補の番号}。"""
    ct = np.array([t for t, _ in cand]); cc = [c for _, c in cand]
    St = []  # (i, j, d)
    for i, (tc, c, _) in enumerate(truth):
        lo = np.searchsorted(ct, tc - HI - 1e-9, "left"); hi = np.searchsorted(ct, tc - LO + 1e-9, "right")
        for j in range(lo, hi):
            if cc[j] == c:
                St.append((i, j, tc - ct[j]))
    if not St:
        return {}
    I = np.array([s[0] for s in St]); J = np.array([s[1] for s in St]); D = np.array([s[2] for s in St])
    n = len(St); DP = np.zeros(n); PR = np.full(n, -1)
    for s in range(n):
        m = (I[:s] < I[s]) & (J[:s] < J[s])
        best, arg = 0.0, -1
        if m.any():
            idx = np.nonzero(m)[0]; val = DP[idx] - np.abs(D[idx] - D[s])
            k = int(np.argmax(val)); best, arg = float(val[k]), int(idx[k])
        DP[s] = BIG + best; PR[s] = arg
    s = int(np.argmax(DP)); res = {}
    while s >= 0:
        res[truth[I[s]][2]] = int(J[s]); s = int(PR[s])
    return res


def q(xs):
    if not xs:
        return dict(n=0, median=None, q1=None, q3=None)
    a = np.array(xs)
    return dict(n=len(xs), median=round(float(np.median(a)), 4), q1=round(float(np.percentile(a, 25)), 4), q3=round(float(np.percentile(a, 75)), 4))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cand", required=True, choices=["rule", "model"])
    ap.add_argument("--probs", default="out/delay_3ways_20261109/probs/probs_base_all.npz")
    a = ap.parse_args()
    t0 = time.time()
    ix = json.load(open(os.path.join(ROOT, IX), encoding="utf-8"))
    songs = [d for d in ix if os.path.exists(os.path.join(ROOT, "feats", d["video"] + ".npz"))]
    if a.cand == "model":
        Z = np.load(os.path.join(ROOT, a.probs))
    rows, newix = [], []
    ncand = collections.Counter()
    for n_, d in enumerate(songs):
        v = d["video"]
        truth = sorted([(t + SHIFT, c, k) for k, (c, t) in enumerate(d["events"]) if c in TYPES], key=lambda x: (x[0], x[2]))
        if a.cand == "rule":
            cand = cand_rule(v)
        else:
            P = Z[v]; cand = sorted((t, c) for k, c in enumerate(TYPES) for t, p in peaks(P[k]) if p >= THR8A_BASE[c])
        for _, c in cand:
            ncand[c] += 1
        m = align(truth, cand)
        ev2 = []
        for k, (c, t) in enumerate(d["events"]):
            tc = t + SHIFT
            if k in m:
                tcand = cand[m[k]][0]; rows.append([v, d["session"], split_of(d["session"]), c, f"{t:.4f}", f"{tc:.4f}", 1, f"{tcand:.2f}", f"{tc - tcand:.4f}"])
                ev2.append([c, round(float(tcand), 4)])
            else:
                rows.append([v, d["session"], split_of(d["session"]), c, f"{t:.4f}", f"{tc:.4f}", 0, "", ""])
                ev2.append([c, t])
        if a.cand == "rule":
            d2 = dict(d); d2["events"] = ev2
            d2["b_note"] = "delay3ways_20261109 案B＝規則の候補と順番の対応付け。対応した件＝候補の時刻・未対応＝1.6722秒を引いた時刻のまま"
            newix.append(d2)
        if (n_ + 1) % 20 == 0:
            print(f"{n_ + 1}/{len(songs)} 曲 {time.time() - t0:.0f}秒", flush=True)
    os.makedirs(os.path.join(OUT, "B"), exist_ok=True)
    with open(os.path.join(OUT, "B", f"align_{a.cand}_20261109.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["video", "session", "split", "type", "t_glo", "tc", "matched", "t_cand", "delay"]); w.writerows(rows)
    summ = {}
    for sp in ("all", "train", "dev", "test"):
        summ[sp] = {}
        for c in TYPES + ["ALL"]:
            R = [r for r in rows if (sp == "all" or r[2] == sp) and (c == "ALL" or r[3] == c)]
            dl = [float(r[8]) for r in R if r[6] == 1]
            e = q(dl); e.update(total=len(R), matched=len(dl), rate=round(len(dl) / len(R), 4) if R else None,
                               diff_from_1_6722=(round(e["median"] - SHIFT, 4) if e["median"] is not None else None))
            summ[sp][c] = e
    summ["n_cand"] = dict(ncand); summ["elapsed_sec"] = round(time.time() - t0, 1); summ["cand"] = a.cand
    json.dump(summ, open(os.path.join(OUT, "B", f"summary_{a.cand}_20261109.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.cand == "rule":
        json.dump(newix, open(os.path.join(OUT, "labels", "index_B_rule_20261109.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for c in TYPES + ["ALL"]:
        e = summ["all"][c]
        print(f"{c}: 対応 {e['matched']}/{e['total']} ({e['rate']}) 遅れ 中央 {e['median']} 四分位 {e['q1']}〜{e['q3']}")
    print("候補の数", dict(ncand), f"所要 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()
