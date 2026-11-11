# -*- coding: utf-8 -*-
r"""手順8＝全部の本を、境目をそろえた形で数え直す（2026-11-05 Opus・c4-compare 追補3・学習しない）。
確率の列＝out/c4_compare_20261105/probs/probs_<name>_{dev,test}.npz（infer_probs_c4 と infer_probs2_c4 が書いた物・pred_*.csv と食い違い 0 を確認済みの物だけ使う）。
8a＝境目を手順1の (ii) で決める＝dev（20260911・30曲）の合計で「出した数」が「正解の数」に最も近い境目（技法ごと・0.01〜0.99 を 0.01 刻み・同じ差なら dev の F ±0.5秒が高い方・なお同じなら低い境目）。
    dev で使う正解＝そのモデルが学習した正解。試験（20260924・28曲）で数える（許容±0.5秒・参考に±1.0秒）。
8b＝境目に依らない数＝技法ごとに、試験の28曲の山（span の内）を確率の高い順に並べ、正解の数と同じ件数だけ出した時の当てた数。
    同じ確率の並びは（曲名・時刻）の順。試験の曲の正解の数を使う物差し＝境目を選ばない比べ用・報告書の主の表には使わない。
数える正解＝そのモデルが学習した正解（own）。学習した正解が全体1つの遅れと違う本は、全体1つの遅れの正解（glo）でも数える（8a の境目は own の dev で決めた物のまま・8b の件数は数える正解の数）。
自己確認＝確率の列から今の境目（metrics.json の thr）で数えた ±0.5秒の 正解・出した・当てた が metrics.json と一致すること（own）。合わなければ止める。
確率の列が無い本は飛ばす。
使い方: python rescore_step8_c4_20261105.py
出力＝out/c4_compare_20261105/手順8_境目をそろえた数え直し_c4_2026-11-05.md ・ 手順8a_c4_2026-11-05.csv ・ 手順8b_c4_2026-11-05.csv ・ .json
"""
import csv, json, os, sys
import numpy as np
from common_c4_20261105 import ROOT, TYPES, JP, DEV, TEST, truth_of, load_index, peaks, match, finish, row, resolve_mdir

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
OUT = os.path.join(ROOT, "out", "c4_compare_20261105")
PR = os.path.join(OUT, "probs")
IXG = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
L4 = "out/c4_compare_20261105/labels/"
IX0, IXP, IXE = L4 + "index_c4_shift0_5tech_20261105.json", L4 + "index_c4_pertech_5tech_20261105.json", L4 + "index_c4_endpoint_5tech_20261105.json"
LN = {IXG: "全体1つの遅れ", IX0: "遅れ 0", IXP: "技法ごとの遅れ", IXE: "終わりの時刻"}
C4 = os.path.join(OUT)
RUNS = [  # key, 名, 確率の名（技法ごとの本は辞書）, 学習した正解, モデルの所在
    ("base", "基準（全体1つの遅れ・seed 0）", "base", IXG, os.path.join(ROOT, "out", "c3_5tech_global_20261105", "c3_5tech_8ep_s0")),
    ("shift0", "遅れ 0 の正解で学習", "shift0", IX0, os.path.join(C4, "c4_shift0_8ep_s0")),
    ("pertech", "技法ごとの遅れの正解で学習", "pertech", IXP, os.path.join(C4, "c4_pertech_8ep_s0")),
    ("endpoint", "終わりの時刻の正解で学習", "endpoint", IXE, os.path.join(C4, "c4_endpoint_8ep_s0")),
    ("es1", "終わりの時刻の正解・seed 1", "es1", IXE, os.path.join(C4, "c4_endpoint_8ep_s1")),
    ("es2", "終わりの時刻の正解・seed 2", "es2", IXE, os.path.join(C4, "c4_endpoint_8ep_s2")),
    ("s1", "全体1つの遅れ・seed 1", "s1", IXG, os.path.join(C4, "c4_global_8ep_s1")),
    ("s2", "全体1つの遅れ・seed 2", "s2", IXG, os.path.join(C4, "c4_global_8ep_s2")),
    ("lc1000", "学習イベント 1,009件（28曲）", "lc1000", IXG, os.path.join(C4, "c4_lc1000_8ep_s0")),
    ("lc2000", "学習イベント 1,994件（52曲）", "lc2000", IXG, os.path.join(C4, "c4_lc2000_8ep_s0")),
    ("lc4000", "学習イベント 3,986件（104曲）", "lc4000", IXG, os.path.join(C4, "c4_lc4000_8ep_s0")),
    ("lc5000", "学習イベント 5,001件（132曲）", "lc5000", IXG, os.path.join(C4, "c4_lc5000_8ep_s0")),
    ("conv", "6b 畳み込みだけ", "conv", IXG, os.path.join(C4, "c4_conv_8ep_s0")),
    ("gru", "6c GRU だけ", "gru", IXG, os.path.join(C4, "c4_gru_8ep_s0")),
    ("half", "6d 幅を半分（64）", "half", IXG, os.path.join(C4, "c4_half_8ep_s0")),
    ("wide", "6e 幅を2倍（256）", "wide", IXG, os.path.join(C4, "c4_wide_8ep_s0")),
    ("only", "6f 技法ごとに1本ずつ（5本）", {c: f"only_{c}" for c in TYPES}, IXG, {c: os.path.join(C4, f"c4_only_{c}_8ep_s0") for c in TYPES}),
]
FINE = [round(x, 2) for x in np.arange(0.01, 0.995, 0.01)]


def ok_probs(name):
    chk = os.path.join(PR, f"probs_{name}_check.txt")
    if not all(os.path.exists(os.path.join(PR, f"probs_{name}_{sp}.npz")) for sp in ("dev", "test")) or not os.path.exists(chk):
        return False
    t = open(chk, encoding="utf-8").read()
    return t.count("食い違い 0件") == 2


def load_pk(names, sp):
    """{video: [技法ごとの山の一覧]}（5技法の順）。names＝確率の名（文字列）か {技法: 名}。"""
    if isinstance(names, str):
        z = np.load(os.path.join(PR, f"probs_{names}_{sp}.npz"))
        return {v: [peaks(z[v][k]) for k in range(len(TYPES))] for v in z.files}
    zs = {c: np.load(os.path.join(PR, f"probs_{n}_{sp}.npz")) for c, n in names.items()}
    vids = zs[TYPES[0]].files
    return {v: [peaks(zs[c][v][0]) for c in TYPES] for v in vids}


def count_thr(pk, truth, thr, tol, only=None):
    agg = {}
    for c in (only or TYPES):
        k = TYPES.index(c); h = npd = nt = 0
        for v, (span, tr) in truth.items():
            lo, hi = span
            pr = [t for t, p in pk[v][k] if p >= thr[c] and lo <= t <= hi]
            h += match(pr, tr[c], tol); npd += len(pr); nt += len(tr[c])
        agg[c] = [h, npd, nt]
    return finish(agg) if only is None else {c: row(*agg[c]) for c in agg}


def choose_ii(pk, truth):
    thr = {}
    for c in TYPES:
        best = None
        for th in FINE:
            e = count_thr(pk, truth, {c: th}, 0.5, only=[c])[c]
            key = (abs(e["pred"] - e["true"]), -e["f1"], th)
            if best is None or key < best[0]:
                best = (key, th)
        thr[c] = best[1]
    return thr


def topn(pk, truth, tol):
    agg = {}
    for c in TYPES:
        k = TYPES.index(c)
        cand = []
        for v, (span, tr) in truth.items():
            lo, hi = span
            cand += [(-p, v, t) for t, p in pk[v][k] if lo <= t <= hi]
        nt = sum(len(tr[c]) for _, tr in truth.values())
        cand.sort()
        sel = {}
        for _, v, t in cand[:nt]:
            sel.setdefault(v, []).append(t)
        h = sum(match(sel.get(v, []), tr[c], tol) for v, (span, tr) in truth.items())
        agg[c] = [h, min(nt, len(cand)), nt]
    return finish(agg)


def main():
    ixs = {ix: load_index(ix) for ix in (IXG, IX0, IXP, IXE)}
    tdev = {ix: truth_of(ixs[ix], DEV) for ix in ixs}; ttest = {ix: truth_of(ixs[ix], TEST) for ix in ixs}
    R, notes, skipped = {}, [], []
    for key, name, pn, ix, mdir in RUNS:
        names = [pn] if isinstance(pn, str) else list(pn.values())
        if not all(ok_probs(n) for n in names):
            skipped.append(key); continue
        mds0 = [mdir] if isinstance(mdir, str) else [mdir[c] for c in TYPES]
        rs = [resolve_mdir(m) for m in mds0]   # 追補6＝打ち切りの本は partial_c4 へ振り替える
        if any(d is None for d, _ in rs):
            skipped.append(key); continue
        pnote = "・".join(nt for _, nt in rs if nt)
        if pnote:
            name = f"{name}［{pnote}］"
        pkd, pkt = load_pk(pn, "dev"), load_pk(pn, "test")
        # 自己確認＝今の境目で metrics.json と照らす
        mds = [d for d, _ in rs]
        n = 0
        for i, md in enumerate(mds):
            mj = json.load(open(os.path.join(md, "metrics.json"), encoding="utf-8"))
            cs = TYPES if isinstance(mdir, str) else [TYPES[i]]
            thr0 = mj["thr"]
            e = count_thr(pkt, ttest[ix], {c: thr0[c] for c in cs}, 0.5, only=cs)
            for c in cs:
                mm = mj["metrics"]["test_tol0.5"][c]
                if (e[c]["true"], e[c]["pred"], e[c]["hit"]) != (mm["true"], mm["pred"], mm["hit"]):
                    raise SystemExit(f"止める＝自己確認の不一致 {key} {c} {e[c]} ／ {mm}")
                n += 1
        notes.append(f"{key}＝今の境目で metrics.json と一致（{n}技法）")
        thr = choose_ii(pkd, tdev[ix])
        devc = count_thr(pkd, tdev[ix], thr, 0.5)
        res = {}
        for which, tix in (("own", ix), ("glo", IXG)):
            if which == "glo" and ix == IXG:
                continue
            res[which] = {}
            for tol in (0.5, 1.0):
                res[which][tol] = dict(a=count_thr(pkt, ttest[tix], thr, tol), b=topn(pkt, ttest[tix], tol))
        R[key] = dict(name=name, ix=ix, thr=thr, dev=devc, res=res)
        print(f"{key}: 8a ±0.5 own " + " ".join(f"{c[:4]} {res['own'][0.5]['a'][c]['pred']}/{res['own'][0.5]['a'][c]['hit']}" for c in TYPES + ["ALL"])
              + f" ｜8b own 当てた {res['own'][0.5]['b']['ALL']['hit']}", flush=True)

    # csv
    with open(os.path.join(OUT, "手順8a_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["モデル", "学習した正解", "数えた正解", "許容秒", "技法", "境目", "正解の数", "モデルが出した数", "当てた数", "余計に出した数", "見落とした数", "dev_正解の数", "dev_出した数"])
        for k, x in R.items():
            for which in x["res"]:
                for tol in (0.5, 1.0):
                    for c in TYPES + ["ALL"]:
                        e = x["res"][which][tol]["a"][c]; d = x["dev"][c]
                        w.writerow([x["name"], LN[x["ix"]], LN[x["ix"]] if which == "own" else LN[IXG], tol, JP[c], "" if c == "ALL" else x["thr"][c],
                                    e["true"], e["pred"], e["hit"], e["extra"], e["miss"], d["true"], d["pred"]])
    with open(os.path.join(OUT, "手順8b_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["モデル", "学習した正解", "数えた正解", "許容秒", "技法", "正解の数（＝出した数）", "当てた数", "注記"])
        for k, x in R.items():
            for which in x["res"]:
                for tol in (0.5, 1.0):
                    for c in TYPES + ["ALL"]:
                        e = x["res"][which][tol]["b"][c]
                        w.writerow([x["name"], LN[x["ix"]], LN[x["ix"]] if which == "own" else LN[IXG], tol, JP[c], e["true"], e["hit"], "境目を選ばない比べ用・報告書の主の表には使わない"])

    L = ["# 手順8＝境目をそろえた数え直し（学習なし）2026-11-05", "", "**Opus 参考値・本人未検証**", "",
         "試験＝20260924 の28曲。照合＝span の内側・技法ごとの貪欲一致。比べの主の列＝当てた数と 8a の表。今の境目（F 最大）の表は c4_比較の表 と 手順6 に残す。", ""]
    if skipped:
        L += ["まだ確率の列が無い本（表に無い）＝" + "・".join(skipped), ""]
    for tol in (0.5, 1.0):
        L += [f"## 8a＝境目を (ii)（調整の30曲で出す数を正解の数に合わせる）で決めた数（許容±{tol}秒）" + ("（主）" if tol == 0.5 else "（参考）"), "",
              "| モデル | 数えた正解 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|"]
        for k, x in R.items():
            for which in x["res"]:
                e = x["res"][which][tol]["a"]["ALL"]
                L.append(f"| {x['name']} | {LN[x['ix']] if which == 'own' else LN[IXG]} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
        L += ["", "技法ごと（当てた数／モデルが出した数・学習した正解で数えた）", "", "| モデル | " + " | ".join(JP[c] for c in TYPES) + " |", "|---|" + "---|" * 5]
        for k, x in R.items():
            e = x["res"]["own"][tol]["a"]
            L.append(f"| {x['name']} | " + " | ".join(f"{e[c]['hit']}／{e[c]['pred']}" for c in TYPES) + " |")
        L.append("")
    L += ["8a の境目（技法ごと）と dev の数", ""]
    for k, x in R.items():
        L.append(f"- {x['name']}＝" + "・".join(f"{JP[c]} {x['thr'][c]:.2f}" for c in TYPES) + f"（dev の正解 {x['dev']['ALL']['true']}・出した {x['dev']['ALL']['pred']}）")
    for tol in (0.5, 1.0):
        L += ["", f"## 8b＝境目に依らない数（確率の高い順に正解の数と同じ件数だけ出した時の当てた数・許容±{tol}秒）" + ("" if tol == 0.5 else "（参考）"), "",
              "注記＝試験の曲の正解の数を使う物差し。境目を選ばない比べ用・報告書の主の表には使わない。", "",
              "| モデル | 数えた正解 | 出した数（＝正解の数） | 当てた数 | " + " | ".join(f"{JP[c]}（当てた／{'出した'}）" for c in TYPES) + " |", "|---|---|---|---|" + "---|" * 5]
        for k, x in R.items():
            for which in x["res"]:
                e = x["res"][which][tol]["b"]
                L.append(f"| {x['name']} | {LN[x['ix']] if which == 'own' else LN[IXG]} | {e['ALL']['pred']} | {e['ALL']['hit']} | " + " | ".join(f"{e[c]['hit']}／{e[c]['pred']}" for c in TYPES) + " |")
    L += ["", "注記",
          "- 8a の境目は、そのモデルが学習した正解で dev を数えて決めた。全体1つの遅れの正解で数えた行も同じ境目。",
          "- 8b の出した数＝数える正解の数（山の数が足りない技法はその数まで）。同じ確率の並びは（曲名・時刻）の順。",
          "- 6a（決め事）と 6g（過去の候補）は確率の列が無い＝8a・8b の対象外。",
          "- 6f は技法ごとの5本の確率の列を、その技法の分だけ使った。", "",
          "自己確認＝" + "／".join(notes)]
    open(os.path.join(OUT, "手順8_境目をそろえた数え直し_c4_2026-11-05.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({k: dict(name=x["name"], thr=x["thr"], dev=x["dev"], res={w_: {str(t): v for t, v in r.items()} for w_, r in x["res"].items()}) for k, x in R.items()},
              open(os.path.join(OUT, "手順8_境目をそろえた数え直し_c4_2026-11-05.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for nt in notes:
        print(nt)
    print("飛ばした＝", skipped)


if __name__ == "__main__":
    main()
