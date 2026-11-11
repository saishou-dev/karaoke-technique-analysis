# -*- coding: utf-8 -*-
r"""手順9＝終わりの時刻の正解 seed 0・1・2 と、全体1つの遅れ seed 0・1・2 を並べる（2026-11-06 Opus・c4-compare 追補4・学習しない）。
数は eval_c4_runs_20261105.py と rescore_step8_c4_20261105.py が書いた json から取る（両方を先に回す）。
3通りの数え方＝今の境目（F 最大）・8a（数を合わせる境目）・8b（境目に依らない）。数える正解＝終わりの時刻と全体1つの遅れの両方。
全体1つの遅れの3本は全体1つの遅れの正解だけで数える（学習した正解＝全体1つ）。
「終わりの時刻の方が上」＝3本どうしの当てた数の範囲が重ならない（終わりの最小 > 全体1つの最大）かで判定。
使い方: python eval_step9_c4_20261105.py
出力＝out/c4_compare_20261105/手順9_終わりの時刻のseed_c4_2026-11-06.md ・ .csv
"""
import csv, json, os, sys
from common_c4_20261105 import ROOT, TYPES, JP

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
OUT = os.path.join(ROOT, "out", "c4_compare_20261105")
A = json.load(open(os.path.join(OUT, "c4_比較の表_2026-11-05.json"), encoding="utf-8"))
B = json.load(open(os.path.join(OUT, "手順8_境目をそろえた数え直し_c4_2026-11-05.json"), encoding="utf-8"))
END = [("endpoint", "0"), ("es1", "1"), ("es2", "2")]
GLO = [("base", "0"), ("s1", "1"), ("s2", "2")]
WAYS = [("now", "今の境目（F 最大）"), ("a", "8a（数を合わせる境目）"), ("b", "8b（境目に依らない）")]


def get(k, way, which, c, y):
    if way == "now":
        src = A.get(k)
        if src is None:
            return None
        r = src["own" if which == "own" else "glo"]["0.5"]
    else:
        src = B.get(k)
        if src is None:
            return None
        r = src["res"].get(which, src["res"]["own"])["0.5"][way]
    return r[c][y]


def main():
    have_e = [k for k, _ in END if k in A and k in B]
    have_g = [k for k, _ in GLO if k in A and k in B]
    L = ["# 手順9＝終わりの時刻の正解 seed 0・1・2 と全体1つの遅れ seed 0・1・2（許容±0.5秒）2026-11-06", "", "**Opus 参考値・本人未検証**", "",
         "試験＝20260924 の28曲・正解の数 982。網・曲の分け方・8エポックは6本とも同じ。違うのは学習の正解と seed。", ""]
    rows = []
    lines1 = []
    for way, wn in WAYS:
        L += [f"## {wn}", "", "| モデル | 数えた正解 | seed | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|"]
        groups = {}
        for grp, lst, which, lab in (("end_own", END, "own", "終わりの時刻"), ("end_glo", END, "glo", "全体1つの遅れ"), ("glo", GLO, "own", "全体1つの遅れ")):
            vals = []
            for k, sd in lst:
                if k not in A or k not in B:
                    continue
                pr, h, ex, mi = (get(k, way, which, "ALL", y) for y in ("pred", "hit", "extra", "miss"))
                name = "終わりの時刻の正解で学習" if lst is END else "全体1つの遅れの正解で学習"
                L.append(f"| {name} | {lab} | {sd} | {pr} | {h} | {ex} | {mi} |")
                rows.append([wn, name, lab, sd, "ALL", pr, h, ex, mi])
                for c in TYPES:
                    rows.append([wn, name, lab, sd, JP[c]] + [get(k, way, which, c, y) for y in ("pred", "hit", "extra", "miss")])
                vals.append(h)
            groups[grp] = vals
        L += ["", "当てた数の最小・最大・幅", ""]
        for grp, nm in (("end_own", "終わりの時刻で学習（終わりの時刻で数えた）"), ("end_glo", "終わりの時刻で学習（全体1つで数えた）"), ("glo", "全体1つの遅れで学習")):
            v = groups[grp]
            if v:
                L.append(f"- {nm}＝{len(v)}本・最小 {min(v)}・最大 {max(v)}・幅 {max(v) - min(v)}（{'・'.join(str(x) for x in v)}）")
        if len(groups["end_own"]) == 3 and len(groups["glo"]) == 3:
            e, g, eg = groups["end_own"], groups["glo"], groups["end_glo"]
            verdict = "言える（範囲が重ならない）" if min(e) > max(g) else ("言えない（範囲が重なる）" if min(e) <= max(g) and max(e) >= min(g) else "逆（終わりの時刻の方が下）")
            verdict2 = "言える" if min(eg) > max(g) else "言えない"
            line = f"{wn}＝終わりの時刻の方が上と{verdict}＝終わり {min(e)}〜{max(e)}・全体1つ {min(g)}〜{max(g)}（全体1つの正解で数えた終わり {min(eg)}〜{max(eg)}＝{verdict2}）。"
            L += ["", "判定（1行）＝" + line]
            lines1.append(line)
        L.append("")
        # 技法ごと（当てた数）
        L += ["技法ごとの当てた数（seed 0・1・2）", "", "| モデル | 数えた正解 | " + " | ".join(JP[c] for c in TYPES) + " |", "|---|---|" + "---|" * 5]
        for lst, which, lab, name in ((END, "own", "終わりの時刻", "終わりの時刻で学習"), (END, "glo", "全体1つの遅れ", "終わりの時刻で学習"), (GLO, "own", "全体1つの遅れ", "全体1つの遅れで学習")):
            ks = [k for k, _ in lst if k in A and k in B]
            if ks:
                L.append(f"| {name} | {lab} | " + " | ".join("・".join(str(get(k, way, which, c, "hit")) for k in ks) for c in TYPES) + " |")
        L.append("")
    L += ["注記",
          "- 8b＝試験の曲の正解の数を使う物差し。境目を選ばない比べ用・報告書の主の表には使わない。",
          "- この台帳は今の遅れの値を使って終わりを探した物（遅れの定めから独立ではない）。3技法のうち 524件は終わりが取れず、カウンターの時刻から中央の値を引いた時刻。",
          "- 全体1つの遅れの seed 0 は基準（c3_5tech_8ep_s0）・seed 1・2 は手順3 の2本。"]
    open(os.path.join(OUT, "手順9_終わりの時刻のseed_c4_2026-11-06.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "手順9_終わりの時刻のseed_c4_2026-11-06.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["数え方", "モデル", "数えた正解", "seed", "技法", "モデルが出した数", "当てた数", "余計に出した数", "見落とした数"]); w.writerows(rows)
    print("終わりの時刻の本＝", have_e, "全体1つの本＝", have_g)
    for ln in lines1:
        print(ln)


if __name__ == "__main__":
    main()
