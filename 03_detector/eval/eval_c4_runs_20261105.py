# -*- coding: utf-8 -*-
r"""c4-compare 手順2〜4 の採点の表（2026-11-05 Opus・eval_c3_5tech_20261105.py の作法を写した新規1本）。学習も推論もしない＝保存済みの pred_test.csv を照合し直すだけ。
採点＝試験（20260924 の28曲）・span の内側・技法ごとの貪欲一致。許容 ±0.5秒（主）・±1.0秒（参考）・csv には ±0.25・±2.0秒も。
数える正解＝「そのモデルが学習した正解」と「全体1つの遅れの正解（index_c3_5tech_20261105）」の両方（列に書く）。
自己確認＝学習した正解で数えた ±0.5・±1.0・±2.0秒を metrics.json（学習台本が書いた test_tol*）と照らす。合わなければ止める。
終わっていない本（metrics.json が無い）は飛ばす＝途中でも、終わった本までの表を書く。
使い方:  python eval_c4_runs_20261105.py
出力＝out/c4_compare_20261105/c4_比較の表_2026-11-05.md ・ c4_比較の表_2026-11-05.csv ・ c4_比較の表_参考F_2026-11-05.csv
"""
import csv, json, os, re, sys
from common_c4_20261105 import ROOT, TYPES, JP, TEST, truth_of, load_index, load_pred_csv, count_pred, resolve_mdir

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
OUT = os.path.join(ROOT, "out", "c4_compare_20261105")
BASE = os.path.join(ROOT, "out", "c3_5tech_global_20261105")
IXG = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
IX0 = "out/c4_compare_20261105/labels/index_c4_shift0_5tech_20261105.json"
IXP = "out/c4_compare_20261105/labels/index_c4_pertech_5tech_20261105.json"
IXE = "out/c4_compare_20261105/labels/index_c4_endpoint_5tech_20261105.json"
TOLS = (0.25, 0.5, 1.0, 2.0)
LABEL_NAME = {IXG: "全体1つの遅れ（1.6722秒）", IX0: "遅れ 0（付け直す前）", IXP: "技法ごとの遅れ", IXE: "終わりの時刻（3技法）"}
RUNS = [
    dict(key="base", name="基準（全体1つの遅れ・seed 0）", mdir=os.path.join(BASE, "c3_5tech_8ep_s0"), log=os.path.join(BASE, "c3_5tech_8ep_s0.log"), ix=IXG),
    dict(key="shift0", name="遅れ 0 の正解で学習", mdir=os.path.join(OUT, "c4_shift0_8ep_s0"), log=os.path.join(OUT, "c4_shift0_8ep_s0.log"), ix=IX0),
    dict(key="pertech", name="技法ごとの遅れの正解で学習", mdir=os.path.join(OUT, "c4_pertech_8ep_s0"), log=os.path.join(OUT, "c4_pertech_8ep_s0.log"), ix=IXP),
    dict(key="endpoint", name="終わりの時刻の正解で学習", mdir=os.path.join(OUT, "c4_endpoint_8ep_s0"), log=os.path.join(OUT, "c4_endpoint_8ep_s0.log"), ix=IXE),
    dict(key="es1", name="終わりの時刻の正解・seed 1", mdir=os.path.join(OUT, "c4_endpoint_8ep_s1"), log=os.path.join(OUT, "c4_endpoint_8ep_s1.log"), ix=IXE),
    dict(key="es2", name="終わりの時刻の正解・seed 2", mdir=os.path.join(OUT, "c4_endpoint_8ep_s2"), log=os.path.join(OUT, "c4_endpoint_8ep_s2.log"), ix=IXE),
    dict(key="s1", name="全体1つの遅れ・seed 1", mdir=os.path.join(OUT, "c4_global_8ep_s1"), log=os.path.join(OUT, "c4_global_8ep_s1.log"), ix=IXG),
    dict(key="s2", name="全体1つの遅れ・seed 2", mdir=os.path.join(OUT, "c4_global_8ep_s2"), log=os.path.join(OUT, "c4_global_8ep_s2.log"), ix=IXG),
]
LC = [(1000, "c4_lc1000_8ep_s0"), (2000, "c4_lc2000_8ep_s0"), (4000, "c4_lc4000_8ep_s0"), (5000, "c4_lc5000_8ep_s0")]
for tg, tag in LC:
    RUNS.append(dict(key=f"lc{tg}", name=f"学習イベント 約{tg:,}件", mdir=os.path.join(OUT, tag), log=os.path.join(OUT, tag + ".log"), ix=IXG))
FIELDS = ("true", "pred", "hit", "extra", "miss")
FJP = dict(true="正解の数", pred="モデルが出した数", hit="当てた数", extra="余計に出した数", miss="見落とした数")


def dev_curve(log):
    if not os.path.exists(log):
        return []
    return [float(m.group(2)) for m in (re.match(r"ep(\d+) loss=\S+ dev F1\(±0.5s\)=([\d.]+) ", ln.strip()) for ln in open(log, encoding="utf-8")) if m]


def lc_amounts():
    path = os.path.join(OUT, "labels", "学習量の内訳_c4_2026-11-05.csv")
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        out[r["目標"]] = (int(r["学習曲数"]), int(r["学習イベント件数"]))
    return out


def tbl(rows):
    """rows＝[(行の名, 数えた正解, res)]・技法ごとの表。"""
    L = ["| モデル | 数えた正解 | 技法 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|---|"]
    for nm, lab, res in rows:
        for c in TYPES + ["ALL"]:
            e = res[c]; L.append(f"| {nm} | {lab} | {JP[c]} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
    return L


def tbl_sum(rows):
    L = ["| モデル | 数えた正解 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|"]
    for nm, lab, res in rows:
        e = res["ALL"]; L.append(f"| {nm} | {lab} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
    return L


def main():
    truths = {ix: truth_of(load_index(ix), TEST) for ix in (IXG, IX0, IXP, IXE)}
    R, notes, skipped = {}, [], []
    for run in RUNS:
        md_, pnote = resolve_mdir(run["mdir"])   # 追補6＝打ち切りの本は partial_c4 へ振り替える
        if md_ is not None and pnote:
            run = dict(run, mdir=md_, name=f"{run['name']}［{pnote}］")
        mj_path = os.path.join(run["mdir"], "metrics.json")
        if not os.path.exists(mj_path):
            skipped.append(run["key"]); continue
        mj = json.load(open(mj_path, encoding="utf-8")); pred = load_pred_csv(os.path.join(run["mdir"], "pred_test.csv"))
        own = {tol: count_pred(pred, truths[run["ix"]], tol) for tol in TOLS}
        glo = own if run["ix"] == IXG else {tol: count_pred(pred, truths[IXG], tol) for tol in TOLS}
        n = 0
        for tol in (0.5, 1.0, 2.0):
            mm = mj["metrics"][f"test_tol{tol}"]
            for c in TYPES + ["ALL"]:
                e = own[tol][c]
                if (e["true"], e["pred"], e["hit"]) != (mm[c]["true"], mm[c]["pred"], mm[c]["hit"]) or abs(round(e["f1"], 3) - mm[c]["f1"]) > 0.0005:
                    raise SystemExit(f"止める＝自己確認の不一致 {run['key']} ±{tol} {c} {e} ／ {mm[c]}")
                n += 1
        notes.append(f"{run['key']}＝metrics.json と一致（3許容×6行・{n}行）")
        R[run["key"]] = dict(run=run, own=own, glo=glo, best_ep=mj["best"][1], thr=mj["thr"], sec=mj.get("elapsed_sec"), n_train=mj.get("n_train"), curve=dev_curve(run["log"]))
    lab_own = lambda k: LABEL_NAME[R[k]["run"]["ix"]]

    # csv（全部の本・両方の正解・4許容）
    with open(os.path.join(OUT, "c4_比較の表_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["モデル", "学習した正解", "数えた正解", "許容秒", "技法"] + [FJP[x] for x in FIELDS] + ["学習曲数", "採用エポック"])
        for k, x in R.items():
            for which, res in (("own", x["own"]), ("glo", x["glo"])):
                if which == "glo" and x["run"]["ix"] == IXG:
                    continue
                for tol in TOLS:
                    for c in TYPES + ["ALL"]:
                        e = res[tol][c]
                        w.writerow([x["run"]["name"], lab_own(k), lab_own(k) if which == "own" else LABEL_NAME[IXG], tol, JP[c]] + [e[y] for y in FIELDS] + [x["n_train"], x["best_ep"]])
    with open(os.path.join(OUT, "c4_比較の表_参考F_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["モデル", "数えた正解", "許容秒", "技法", "適合率", "再現率", "F値"])
        for k, x in R.items():
            for which, res in (("own", x["own"]), ("glo", x["glo"])):
                if which == "glo" and x["run"]["ix"] == IXG:
                    continue
                for tol in TOLS:
                    for c in TYPES + ["ALL"]:
                        e = res[tol][c]
                        w.writerow([x["run"]["name"], lab_own(k) if which == "own" else LABEL_NAME[IXG], tol, JP[c], round(e["precision"], 3), round(e["recall"], 3), round(e["f1"], 3)])

    L = ["# c4-compare 手順2〜4 の表（5技法・試験 20260924 の28曲）2026-11-05", "", "**Opus 参考値・本人未検証**", "",
         "照合＝span の内側・技法ごとの貪欲一致（c3-5tech と同じ）。境目＝各モデルの学習台本が調整（dev 20260911・30曲）で決めた物。",
         "余計に出した数＝出した数−当てた数。見落とした数＝正解の数−当てた数。", ""]
    if skipped:
        L += ["まだ終わっていない本（表に無い）＝" + "・".join(skipped), ""]

    # 手順2
    for tol in (0.5, 1.0):
        rows = []
        for k in ("base", "shift0", "pertech", "endpoint"):
            if k not in R:
                continue
            rows.append((R[k]["run"]["name"], lab_own(k), R[k]["own"][tol]))
            if R[k]["run"]["ix"] != IXG:
                rows.append((R[k]["run"]["name"], LABEL_NAME[IXG], R[k]["glo"][tol]))
        L += [f"## 手順2＝正解の時刻の比較（許容±{tol}秒）" + ("（主）" if tol == 0.5 else "（参考）"), "", "合計", ""] + tbl_sum(rows) + ["", "技法ごと", ""] + tbl(rows) + [""]
        if "endpoint" in R:
            e3 = lambda res: [sum(res[c][y] for c in ("shakuri", "fall", "hammering")) for y in ("true", "pred", "hit")]
            b, eo, eg = e3(R["base"]["own"][tol]), e3(R["endpoint"]["own"][tol]), e3(R["endpoint"]["glo"][tol])
            L += [f"- 3技法（しゃくり・フォール・ハンマリング）の和＝基準 正解 {b[0]}・出した {b[1]}・当てた {b[2]}／終わりの時刻で学習（学習した正解で数えた）正解 {eo[0]}・出した {eo[1]}・当てた {eo[2]}／同（全体1つの遅れの正解で数えた）当てた {eg[2]}", ""]
    L += ["手順2の注記",
          "- 遅れ 0＝付け直す前（カウンターが増えた時刻そのまま＝index.json の時刻）。全体1つ＝1.6722秒を前へ。技法ごと＝しゃくり 1.641・こぶし 1.942・フォール 1.899・アクセント 1.618・ハンマリング 1.623秒を前へ。",
          "- 終わりの時刻＝別の部下（endpoint-all・11/5 18:38）の台帳。しゃくり・フォール・ハンマリングだけ終わりの時刻、ほかは全体1つの遅れのまま。",
          "  この台帳は今の遅れの値を使って終わりを探した物（遅れの定めから独立ではない）。3技法のうち 524件は終わりが取れず、カウンターの時刻から中央の値を引いた時刻。",
          "- 4本とも網・曲の分け方・seed 0・8エポックは同じ。違うのは学習の正解だけ。", ""]

    # 手順3
    seeds = [k for k in ("base", "s1", "s2") if k in R]
    if len(seeds) >= 2:
        for tol in (0.5, 1.0):
            L += [f"## 手順3＝seed のばらつき（全体1つの遅れ・許容±{tol}秒・seed {'・'.join({'base': '0', 's1': '1', 's2': '2'}[k] for k in seeds)}）", ""]
            L += ["| 技法 | 数 | " + " | ".join(f"seed {({'base': '0', 's1': '1', 's2': '2'}[k])}" for k in seeds) + " | 最小 | 最大 | 幅 |", "|---|---|" + "---|" * (len(seeds) + 3)]
            for c in TYPES + ["ALL"]:
                for y in ("pred", "hit", "extra", "miss"):
                    v = [R[k]["own"][tol][c][y] for k in seeds]
                    L.append(f"| {JP[c]} | {FJP[y]} | " + " | ".join(str(z) for z in v) + f" | {min(v)} | {max(v)} | {max(v) - min(v)} |")
            L += ["", "正解の数は3本とも同じ（同じ正解・同じ試験の曲）＝" + "・".join(f"{JP[c]} {R['base']['own'][tol][c]['true']}" for c in TYPES + ["ALL"]), ""]
            # 参考 F
            L += ["参考＝F値（合計と技法ごと）", "", "| 技法 | " + " | ".join(f"seed {({'base': '0', 's1': '1', 's2': '2'}[k])}" for k in seeds) + " | 最小 | 最大 | 幅 |", "|---|" + "---|" * (len(seeds) + 3)]
            for c in TYPES + ["ALL"]:
                v = [R[k]["own"][tol][c]["f1"] for k in seeds]
                L.append(f"| {JP[c]} | " + " | ".join(f"{z:.3f}" for z in v) + f" | {min(v):.3f} | {max(v):.3f} | {max(v) - min(v):.3f} |")
            L.append("")
            # 手順2の差と幅の比べ
            for k in ("shift0", "pertech", "endpoint"):
                if k not in R:
                    continue
                for which, lab in (("own", lab_own(k)), ("glo", LABEL_NAME[IXG])):
                    d_hit = R[k][which][tol]["ALL"]["hit"] - R["base"]["own"][tol]["ALL"]["hit"]
                    d_ex = R[k][which][tol]["ALL"]["extra"] - R["base"]["own"][tol]["ALL"]["extra"]
                    vh = [R[s]["own"][tol]["ALL"]["hit"] for s in seeds]; ve = [R[s]["own"][tol]["ALL"]["extra"] for s in seeds]
                    L.append(f"- {R[k]['run']['name']}（{lab}で数えた）−基準＝当てた {d_hit:+d}（seed の幅 {max(vh) - min(vh)}）・余計 {d_ex:+d}（seed の幅 {max(ve) - min(ve)}）＝"
                             + ("当てた数の差は seed の幅より大きい" if abs(d_hit) > max(vh) - min(vh) else "当てた数の差は seed の幅の内"))
            L.append("")

    # 手順4
    amt = lc_amounts()
    pts = [(tg, f"lc{tg}") for tg, _ in LC if f"lc{tg}" in R]
    if pts:
        for tol in (0.5, 1.0):
            L += [f"## 手順4＝学習に使う正解の数ごと（全体1つの遅れ・seed 0・許容±{tol}秒）", "",
                  "| 学習イベント（100件単位） | 実際の件数 | 学習の曲数 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|---|"]
            for tg, k in pts + [(None, "base")]:
                n_s, n_e = amt["全部（基準）"] if k == "base" else amt[str(tg)]
                e = R[k]["own"][tol]["ALL"]
                L.append(f"| {round(n_e / 100):d}00 | {n_e:,} | {n_s} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |" + ("" if k != "base" else ""))
            L += ["", "技法ごと（当てた数／モデルが出した数）", "", "| 学習イベント | " + " | ".join(JP[c] for c in TYPES) + " |", "|---|" + "---|" * 5]
            for tg, k in pts + [(None, "base")]:
                n_s, n_e = amt["全部（基準）"] if k == "base" else amt[str(tg)]
                L.append(f"| {round(n_e / 100):d}00（{n_e:,}件・{n_s}曲） | " + " | ".join(f"{R[k]['own'][tol][c]['hit']}／{R[k]['own'][tol][c]['pred']}" for c in TYPES) + " |")
            L += ["", "参考＝F値（合計）＝" + "・".join(f"{round((amt['全部（基準）'] if k == 'base' else amt[str(tg)])[1] / 100):d}00件 {R[k]['own'][tol]['ALL']['f1']:.3f}" for tg, k in pts + [(None, 'base')]), ""]
        L += ["注記",
              "- 6,000件は学習の曲の全件（150曲・5,717件）を超える＝全部の曲＝基準と同じ。代わりに 約5,000件（132曲・5,001件）を回した。",
              "- 学習の曲を video 名で並べ random.Random(0) で1回混ぜ、先頭から取った入れ子（小さい集合は大きい集合の部分集合）。調整と試験の曲は全部同じ。", ""]

    # 参考 F の一覧と学習の記録
    L += ["## 参考＝F値・適合率・再現率（合計・許容±0.5秒）", "", "| モデル | 数えた正解 | F値 | 適合率 | 再現率 |", "|---|---|---|---|---|"]
    for k, x in R.items():
        for which in ("own", "glo"):
            if which == "glo" and x["run"]["ix"] == IXG:
                continue
            e = x[which][0.5]["ALL"]
            L.append(f"| {x['run']['name']} | {lab_own(k) if which == 'own' else LABEL_NAME[IXG]} | {e['f1']:.3f} | {e['precision']:.3f} | {e['recall']:.3f} |")
    L += ["", "## 学習の記録", ""]
    for k, x in R.items():
        L.append(f"- {x['run']['name']}＝学習 {x['n_train']}曲・採用エポック {x['best_ep']}・所要 {x['sec']}秒・dev F（±0.5秒）＝" + "・".join(f"{f:.3f}" for f in x["curve"]))
        L.append("  しきい値（dev）＝" + "・".join(f"{JP[c]} {x['thr'][c]:.3f}" for c in TYPES))
    L += ["", "自己確認＝" + "／".join(notes)]
    open(os.path.join(OUT, "c4_比較の表_2026-11-05.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({k: dict(own={str(t): v["own"][t] for t in TOLS}, glo={str(t): v["glo"][t] for t in TOLS}, thr=v["thr"], best_ep=v["best_ep"], sec=v["sec"], n_train=v["n_train"], curve=v["curve"]) for k, v in R.items()},
              open(os.path.join(OUT, "c4_比較の表_2026-11-05.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for nt in notes:
        print(nt)
    for k, x in R.items():
        for which in ("own", "glo"):
            e = x[which][0.5]
            print(f"{k} {which} ±0.5: " + " ".join(f"{c[:4]} {e[c]['true']}/{e[c]['pred']}/{e[c]['hit']}" for c in TYPES + ["ALL"]) + f" F{e['ALL']['f1']:.3f}")
    print("飛ばした（未完）＝", skipped)


if __name__ == "__main__":
    main()
