# -*- coding: utf-8 -*-
r"""手順1＝学習なしの再集計（2026-11-05 Opus・c4-compare・指示書 2026-11-05_1806）。
基準のモデル＝out/c3_5tech_global_20261105/c3_5tech_8ep_s0（全体1つの遅れ 1.6722秒・5技法・seed 0・8エポック）。
確率の列＝infer_probs_c4_20261105.py が書いた out/c4_compare_20261105/probs/probs_base_{dev,test}.npz（pred_*.csv と食い違い 0 を確認済み）。
正解＝out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json（全体1つの遅れ）。
1a 許容の幅＝±0.25・±0.5・±1.0・±2.0秒（境目は今の物＝metrics.json の thr）。
1b 境目の決め方（境目は dev＝20260911 の30曲だけで決める・試験＝20260924 の28曲で数える）
  (i)  今の決め方＝学習台本が決めた物（dev・技法ごとに F 最大・許容±0.5秒・0.05〜0.95 を 0.025 刻み）。
  (ii) 数を合わせる＝dev の30曲で、技法ごとに「出した数」が「正解の数」に最も近い境目（0.01〜0.99 を 0.01 刻み。同じ差なら dev の F（±0.5秒）が高い方、なお同じなら低い境目）。
  (iii) F 最大＝dev の30曲で、技法ごとに F が最大の境目（0.01〜0.99 を 0.01 刻み・数える許容と同じ許容で選ぶ。同じ F なら低い境目）。
自己確認＝(i) の ±0.5・±1.0・±2.0 を metrics.json の test_tol* と照らす。(i) の境目を元の刻みで選び直して metrics.json の thr と照らす。
使い方: python rescore_step1_c4_20261105.py
"""
import csv, json, os, sys
import numpy as np
from common_c4_20261105 import ROOT, TYPES, JP, DEV, TEST, truth_of, peaks_of, count, load_index, md_table

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
OUT = os.path.join(ROOT, "out", "c4_compare_20261105")
MDIR = os.path.join(ROOT, "out", "c3_5tech_global_20261105", "c3_5tech_8ep_s0")
IX = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
TOLS = (0.25, 0.5, 1.0, 2.0)
FINE = [round(x, 2) for x in np.arange(0.01, 0.995, 0.01)]
ORIG = list(np.arange(0.05, 0.96, 0.025))


def per_tech(pk, tr, c, th, tol):
    thr = {cc: (th if cc == c else 2.0) for cc in TYPES}
    return count(pk, tr, thr, tol)[c]


def main():
    mj = json.load(open(os.path.join(MDIR, "metrics.json"), encoding="utf-8"))
    thr_i = mj["thr"]
    ix = load_index(IX)
    tr_dev, tr_test = truth_of(ix, DEV), truth_of(ix, TEST)
    pk_dev = peaks_of(os.path.join(OUT, "probs", "probs_base_dev.npz"))
    pk_test = peaks_of(os.path.join(OUT, "probs", "probs_base_test.npz"))
    print(f"dev {len(tr_dev)}曲・試験 {len(tr_test)}曲")
    notes = []

    # 自己確認 1＝(i) の境目を元の刻み・元の決め方（他の技法は 0.99）で選び直す
    re_i = {}
    for c in TYPES:
        best = None
        for th in ORIG:
            f = round(per_tech(pk_dev, tr_dev, c, th, 0.5)["f1"], 3)
            if best is None or f > best[0]:
                best = (f, float(th))
        re_i[c] = best[1]
    same = all(abs(re_i[c] - thr_i[c]) < 1e-9 for c in TYPES)
    notes.append(f"(i) の境目を元の刻みで選び直し＝metrics.json の thr と{'一致' if same else '不一致'}（{ {c: round(re_i[c], 3) for c in TYPES} }）")

    # 1a
    A = {tol: count(pk_test, tr_test, thr_i, tol) for tol in TOLS}
    nchk = 0
    for tol in (0.5, 1.0, 2.0):
        mm = mj["metrics"][f"test_tol{tol}"]
        for c in TYPES + ["ALL"]:
            e = A[tol][c]
            if (e["true"], e["pred"], e["hit"]) != (mm[c]["true"], mm[c]["pred"], mm[c]["hit"]):
                raise SystemExit(f"止める＝1a の自己確認の不一致 ±{tol} {c} {e} ／ {mm[c]}")
            nchk += 1
    notes.append(f"1a の ±0.5・±1.0・±2.0 秒＝metrics.json の test と一致（3許容×6行×正解・出した・当てた＝{nchk}行）")

    # 1b
    thr_ii, dev_ii = {}, {}
    for c in TYPES:
        best = None
        for th in FINE:
            e = per_tech(pk_dev, tr_dev, c, th, 0.5)
            key = (abs(e["pred"] - e["true"]), -e["f1"], th)
            if best is None or key < best[0]:
                best = (key, th, e)
        thr_ii[c] = best[1]; dev_ii[c] = best[2]
    thr_iii = {}
    for tol in TOLS:
        thr_iii[tol] = {}
        for c in TYPES:
            best = None
            for th in FINE:
                f = per_tech(pk_dev, tr_dev, c, th, tol)["f1"]
                if best is None or f > best[0] + 1e-12:
                    best = (f, th)
            thr_iii[tol][c] = best[1]
    B = {}
    for tol in TOLS:
        B[("i", tol)] = A[tol]
        B[("ii", tol)] = count(pk_test, tr_test, thr_ii, tol)
        B[("iii", tol)] = count(pk_test, tr_test, thr_iii[tol], tol)
    D = {("i", tol): count(pk_dev, tr_dev, thr_i, tol) for tol in TOLS}
    D.update({("ii", tol): count(pk_dev, tr_dev, thr_ii, tol) for tol in TOLS})
    D.update({("iii", tol): count(pk_dev, tr_dev, thr_iii[tol], tol) for tol in TOLS})
    MN = {"i": "(i) 今の決め方", "ii": "(ii) 出す数を正解の数に合わせる", "iii": "(iii) F 最大（細かい刻み・数える許容で選ぶ）"}

    # csv
    with open(os.path.join(OUT, "手順1a_許容の幅_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["許容秒", "技法", "正解の数", "モデルが出した数", "当てた数", "余計に出した数", "見落とした数"])
        for tol in TOLS:
            for c in TYPES + ["ALL"]:
                e = A[tol][c]; w.writerow([tol, JP[c], e["true"], e["pred"], e["hit"], e["extra"], e["miss"]])
    with open(os.path.join(OUT, "手順1b_境目の決め方_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["決め方", "許容秒", "技法", "境目", "正解の数", "モデルが出した数", "当てた数", "余計に出した数", "見落とした数", "dev_正解の数", "dev_出した数", "dev_当てた数"])
        for m in ("i", "ii", "iii"):
            for tol in TOLS:
                th = thr_i if m == "i" else (thr_ii if m == "ii" else thr_iii[tol])
                for c in TYPES + ["ALL"]:
                    e = B[(m, tol)][c]; d = D[(m, tol)][c]
                    w.writerow([MN[m], tol, JP[c], "" if c == "ALL" else round(th[c], 3), e["true"], e["pred"], e["hit"], e["extra"], e["miss"], d["true"], d["pred"], d["hit"]])
    with open(os.path.join(OUT, "手順1_参考F_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["決め方", "許容秒", "技法", "適合率", "再現率", "F値"])
        for m in ("i", "ii", "iii"):
            for tol in TOLS:
                for c in TYPES + ["ALL"]:
                    e = B[(m, tol)][c]; w.writerow([MN[m], tol, JP[c], round(e["precision"], 3), round(e["recall"], 3), round(e["f1"], 3)])

    # md
    L = ["# 手順1＝学習なしの再集計（基準のモデル c3_5tech_8ep_s0）2026-11-05", "", "**Opus 参考値・本人未検証**", "",
         "基準のモデル＝5技法・全体1つの遅れ 1.6722秒・seed 0・8エポック・学習150曲・調整（dev）30曲・試験28曲。",
         "試験＝20260924 の28曲。正解＝全体1つの遅れの正解。照合＝span の内側・技法ごとの貪欲一致。",
         "余計に出した数＝出した数−当てた数。見落とした数＝正解の数−当てた数。", "",
         "## 表1＝許容の幅（境目は今の物）", ""]
    L += ["| 許容 | 技法 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|"]
    for tol in TOLS:
        for c in TYPES + ["ALL"]:
            e = A[tol][c]
            L.append(f"| ±{tol}秒 | {JP[c]} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
    L += ["", "合計だけ（許容ごと）", "", "| 許容 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|"]
    for tol in TOLS:
        e = A[tol]["ALL"]; L.append(f"| ±{tol}秒 | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
    for tol in (0.5, 1.0):
        L += ["", f"## 表2＝境目の決め方（境目は dev の30曲で決めた・試験の28曲で数えた・許容±{tol}秒）", ""]
        L += ["| 決め方 | 技法 | 境目 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|---|"]
        for m in ("i", "ii", "iii"):
            th = thr_i if m == "i" else (thr_ii if m == "ii" else thr_iii[tol])
            for c in TYPES + ["ALL"]:
                e = B[(m, tol)][c]
                L.append(f"| {MN[m]} | {JP[c]} | {'' if c == 'ALL' else f'{th[c]:.3f}'} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
    L += ["", "合計だけ（4許容×3通り）", "", "| 決め方 | 許容 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|"]
    for m in ("i", "ii", "iii"):
        for tol in TOLS:
            e = B[(m, tol)]["ALL"]; L.append(f"| {MN[m]} | ±{tol}秒 | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
    L += ["", "## 参考＝F値・適合率・再現率（試験）", "", "| 決め方 | 許容 | " + " | ".join(JP[c] for c in TYPES + ["ALL"]) + " |", "|---|---|" + "---|" * 6]
    for m in ("i", "ii", "iii"):
        for tol in TOLS:
            e = B[(m, tol)]
            L.append(f"| {MN[m]} | ±{tol}秒 | " + " | ".join(f"F {e[c]['f1']:.3f}（適合 {e[c]['precision']:.3f}／再現 {e[c]['recall']:.3f}）" for c in TYPES + ["ALL"]) + " |")
    L += ["", "## dev の30曲での数（境目を決めた側）", "", "| 決め方 | 許容 | 正解の数 | モデルが出した数 | 当てた数 |", "|---|---|---|---|---|"]
    for m in ("i", "ii", "iii"):
        for tol in (0.5, 1.0):
            d = D[(m, tol)]["ALL"]; L.append(f"| {MN[m]} | ±{tol}秒 | {d['true']} | {d['pred']} | {d['hit']} |")
    L += ["", "## 決め方の中身", "",
          "- (i)＝学習台本の決め方。dev・技法ごとに F 最大（許容±0.5秒）・0.05〜0.95 を 0.025 刻み・選ぶ技法の外は 0.99。採用エポックの時の値。",
          "- (ii)＝dev の30曲の合計で「出した数」と「正解の数」の差が最小の境目。0.01〜0.99 を 0.01 刻み。同じ差なら dev の F（±0.5秒）が高い方。",
          "  学習台本の --thr_mode count（曲ごとの差の絶対値の和）とは別物＝合計の差。",
          "- (iii)＝dev の30曲で F 最大の境目。0.01〜0.99 を 0.01 刻み。数える許容と同じ許容で選ぶ（±1.0秒の行は ±1.0秒の F で選んだ境目）。",
          "  (i) との違い＝刻みの細かさと、選ぶ時の許容だけ。", "",
          "境目（技法ごと）",
          "- (i)＝" + "・".join(f"{JP[c]} {thr_i[c]:.3f}" for c in TYPES),
          "- (ii)＝" + "・".join(f"{JP[c]} {thr_ii[c]:.2f}" for c in TYPES)]
    for tol in TOLS:
        L.append(f"- (iii) ±{tol}秒＝" + "・".join(f"{JP[c]} {thr_iii[tol][c]:.2f}" for c in TYPES))
    L += ["", "自己確認"] + [f"- {n}" for n in notes]
    open(os.path.join(OUT, "手順1_再集計_c4_2026-11-05.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(thr_i=thr_i, thr_ii=thr_ii, thr_iii={str(k): v for k, v in thr_iii.items()},
                   A={str(t): A[t] for t in TOLS}, B={f"{m}_{t}": B[(m, t)] for m in ("i", "ii", "iii") for t in TOLS},
                   D={f"{m}_{t}": D[(m, t)] for m in ("i", "ii", "iii") for t in TOLS}, notes=notes),
              open(os.path.join(OUT, "手順1_再集計_c4_2026-11-05.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for n in notes:
        print(n)
    for tol in TOLS:
        print(f"1a ±{tol}: " + " ".join(f"{c[:4]} {A[tol][c]['true']}/{A[tol][c]['pred']}/{A[tol][c]['hit']}" for c in TYPES + ["ALL"]))
    for m in ("i", "ii", "iii"):
        for tol in TOLS:
            e = B[(m, tol)]
            print(f"1b {m} ±{tol}: " + " ".join(f"{c[:4]} {e[c]['true']}/{e[c]['pred']}/{e[c]['hit']} F{e[c]['f1']:.3f}" for c in TYPES + ["ALL"]))
    print("thr_ii", thr_ii); print("thr_iii", thr_iii)


if __name__ == "__main__":
    main()
