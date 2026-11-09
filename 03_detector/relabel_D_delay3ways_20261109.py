# -*- coding: utf-8 -*-
r"""案D＝反復の置き直し（EM の考え）の1回分（2026-11-09 Opus・delay-3ways・学習しない）。
入力＝前の回のモデル（1回目＝基準種0 c3_5tech_8ep_s0）の確率の列（学習曲・dev）と、前の回の正解の一覧（1回目＝全体1つの遅れの正解）。
手順
  1. 前の回のモデルの 8a の境目を dev（20260911・30曲・全体1つの遅れの正解）で決める（11/5 手順8 と同じ決め方）。
  2. 学習曲（150曲）の各加点（技法 c・カウンターの時刻 tc＝全体1つの遅れの正解の時刻＋1.6722）について、
     [tc−3.0, tc] の中の同じ技法の確率の山（pick と同じ山・前後 0.3秒の中で最大）のうち、確率が 8a の境目以上で最も高い山の時刻へ置き直す。
     届く山が無い件＝前の回の時刻のまま。同じ曲・同じ技法で、別の件が既に取った山は取らない（時刻の順に処理・取れない件は前の回の時刻のまま）。
  3. dev・試験の曲の正解は変えない（全体1つの遅れの正解のまま＝数える物差しは今と同じ）。
出す物＝置き直した件の割合（技法別）・置き直した件の遅れ tc−t新 の分布・全件の tc−t の分布（中央値・四分位）。
使い方: python relabel_D_delay3ways_20261109.py --round 1 --mdir out/c3_5tech_global_20261105/c3_5tech_8ep_s0 --probs_train out/delay_3ways_20261109/probs/probs_base_all.npz --probs_dev out/c4_compare_20261105/probs/probs_base_dev.npz --prev_index out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json
出力＝out/delay_3ways_20261109/labels/index_D_r<回>_20261109.json・D/relabel_r<回>_20261109.json・.csv
"""
import argparse, csv, json, os, sys, time
import numpy as np
from common_delay3ways_20261109 import ROOT, TYPES, DEV, TEST, FR, SHIFT, IXG, OUT, truth_of, load_index, peaks, pk_of, choose_ii, qstat

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True); ap.add_argument("--mdir", required=True)
    ap.add_argument("--probs_train", required=True); ap.add_argument("--probs_dev", required=True); ap.add_argument("--prev_index", required=True)
    ap.add_argument("--win", type=float, default=3.0)
    a = ap.parse_args(); t0 = time.time()
    glo = load_index(IXG); prev = load_index(a.prev_index)
    assert [d["video"] for d in glo] == [d["video"] for d in prev]
    zd = np.load(os.path.join(ROOT, a.probs_dev)); thr = choose_ii(pk_of(zd), truth_of(glo, DEV))
    zt = np.load(os.path.join(ROOT, a.probs_train))
    rows, new = [], []
    for dg, dp in zip(glo, prev):
        d2 = dict(dp)
        if dg["session"] in (DEV, TEST) or not os.path.exists(os.path.join(ROOT, "feats", dg["video"] + ".npz")):
            new.append(d2); continue
        P = zt[dg["video"]]
        pks = {c: [(t, p) for t, p in peaks(P[k]) if p >= thr[c]] for k, c in enumerate(TYPES)}
        taken = {c: set() for c in TYPES}
        ev = [list(e) for e in dp["events"]]
        order = sorted(range(len(dg["events"])), key=lambda i: dg["events"][i][1])
        for i in order:
            c, tg = dg["events"][i]
            if c not in TYPES:
                continue
            assert dp["events"][i][0] == c
            tc = tg + SHIFT; tprev = dp["events"][i][1]
            cand = [(p, t) for t, p in pks[c] if tc - a.win - 1e-9 <= t <= tc + 1e-9 and round(t * FR) not in taken[c]]
            if cand:
                p, t = max(cand); taken[c].add(round(t * FR)); ev[i][1] = round(float(t), 4)
                rows.append([dg["video"], c, f"{tg:.4f}", f"{tc:.4f}", f"{tprev:.4f}", f"{t:.2f}", 1, f"{p:.3f}", f"{tc - t:.4f}"])
            else:
                rows.append([dg["video"], c, f"{tg:.4f}", f"{tc:.4f}", f"{tprev:.4f}", f"{tprev:.4f}", 0, "", f"{tc - tprev:.4f}"])
        d2["events"] = ev
        d2["d_note"] = f"delay3ways_20261109 案D {a.round}回目＝学習曲だけ、前の回のモデル（{a.mdir}）の確率の山（8a の境目以上・[tc−{a.win}, tc]）へ置き直し"
        new.append(d2)
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True); os.makedirs(os.path.join(OUT, "D"), exist_ok=True)
    json.dump(new, open(os.path.join(OUT, "labels", f"index_D_r{a.round}_20261109.json"), "w", encoding="utf-8"), ensure_ascii=False)
    with open(os.path.join(OUT, "D", f"relabel_r{a.round}_20261109.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["video", "type", "t_glo", "tc", "t_prev", "t_new", "moved_to_peak", "peak_prob", "delay_tc_minus_t"]); w.writerows(rows)
    S = dict(round=a.round, mdir=a.mdir, thr8a=thr, per={})
    for c in TYPES + ["ALL"]:
        R = [r for r in rows if c == "ALL" or r[1] == c]
        mv = [float(r[8]) for r in R if r[6] == 1]; al = [float(r[8]) for r in R]
        chg = sum(1 for r in R if abs(float(r[5]) - float(r[4])) > 0.005)
        S["per"][c] = dict(total=len(R), moved=len(mv), rate=round(len(mv) / len(R), 4) if R else None, changed_from_prev=chg,
                           delay_moved=qstat(mv), delay_all=qstat(al))
    S["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(S, open(os.path.join(OUT, "D", f"relabel_r{a.round}_20261109.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"D {a.round}回目 8a の境目 {thr}")
    for c in TYPES + ["ALL"]:
        e = S["per"][c]
        print(f"{c}: 山へ置いた {e['moved']}/{e['total']} ({e['rate']}) 前の回から動いた {e['changed_from_prev']} 遅れ（置いた件）{e['delay_moved']} 全件 {e['delay_all']}")
    print(f"所要 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()
