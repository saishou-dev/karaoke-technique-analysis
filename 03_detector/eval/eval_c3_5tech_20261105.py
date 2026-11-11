# -*- coding: utf-8 -*-
r"""c3-5tech の採点の表を作る（2026-11-05 Opus・eval_c2_shift2_20261105.py の作法を写した新規1本）。学習も推論もしない＝保存済みの pred_test.csv を照合し直すだけ。

採点＝試験（20260924 の28曲＝学習に使っていない曲）・span の内側・技法ごとの貪欲一致（今朝と同じ match）。許容 ±0.5秒（主）・±1.0秒（参考）。
行
  本命＝今回のモデル（c3_5tech_8ep_s0・5技法）× 今回の正解（index_c3_5tech_20261105＝遅れ 1.6722秒・全技法共通・vibrato なし）
  並べる物＝今朝の c2-shift2（c2_shift2_8ep_s0・6技法で学習）× 今朝の正解（index_shift2_20261105＝技法ごとの遅れ）の同じ5技法
数＝正解の数（span 内の正解イベント）・出した数（span 内の予測）・当てた数（貪欲一致の数）・余計に出した数＝出した数−当てた数・見落とした数＝正解の数−当てた数。
自己確認＝両方とも metrics.json（学習台本が書いた test_tol0.5・test_tol1.0）と、5技法と合計の 正解・出した・当てた・F を照らす。合わなければ止める。
使い方:  python eval_c3_5tech_20261105.py
出力＝out/c3_5tech_global_20261105/技法別の表_c3_5tech_2026-11-05.md ／ .csv（主の表）／ 技法別の表_参考F_c3_5tech_2026-11-05.csv
"""
import csv, json, os, re, sys, collections

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "c3_5tech_global_20261105")
OUT2 = os.path.join(ROOT, "out", "c2_shift2_20261105")
TYPES = ["shakuri", "kobushi", "fall", "accent", "hammering"]
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング", "ALL": "5技法の合計"}
TOLS = (0.5, 1.0)
TEST = "20260924"
RUNS = [
    dict(key="c3", name="今回（全体1つの遅れ 1.6722秒・5技法で学習）", mdir=os.path.join(OUT, "c3_5tech_8ep_s0"),
         truth=os.path.join(OUT, "labels", "index_c3_5tech_20261105.json"), log=os.path.join(OUT, "c3_5tech_8ep_s0.log")),
    dict(key="c2", name="今朝の c2-shift2（技法ごとの遅れ・6技法で学習）", mdir=os.path.join(OUT2, "c2_shift2_8ep_s0"),
         truth=os.path.join(OUT2, "labels", "index_shift2_20261105.json"), log=os.path.join(OUT2, "c2_shift2_8ep_s0.log")),
]


def match(pred, truth, tol):
    used, hit = set(), 0
    for p in sorted(pred):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - p) > tol:
                continue
            if best is None or abs(t - p) < abs(truth[best] - p):
                best = k
        if best is not None:
            used.add(best); hit += 1
    return hit


def prf(h, npd, nt):
    p = h / npd if npd else 0.0; r = h / nt if nt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def load_pred(mdir):
    pred = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(mdir, "pred_test.csv"), encoding="utf-8-sig")):
        pred[(r["video"], r["type"])].append(float(r["wav_time_sec"]))
    return pred


def evaluate(pred, index):
    vids = [d for d in index if d["session"] == TEST]
    res = {}
    for tol in TOLS:
        agg = {c: [0, 0, 0] for c in TYPES}
        for d in vids:
            v = d["video"]; lo, hi = d["span"]
            for c in TYPES:
                tr = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]
                pr = [t for t in pred.get((v, c), []) if lo <= t <= hi]
                agg[c][0] += match(pr, tr, tol); agg[c][1] += len(pr); agg[c][2] += len(tr)
        out = {}
        for c in TYPES:
            h, npd, nt = agg[c]; p, r, f = prf(h, npd, nt)
            out[c] = dict(true=nt, pred=npd, hit=h, extra=npd - h, miss=nt - h, precision=p, recall=r, f1=f)
        H, NP, NT = (sum(x[i] for x in agg.values()) for i in range(3))
        p, r, f = prf(H, NP, NT)
        out["ALL"] = dict(true=NT, pred=NP, hit=H, extra=NP - H, miss=NT - H, precision=p, recall=r, f1=f)
        res[tol] = out
    return res, len(vids)


def dev_curve(log):
    return [(int(m.group(1)), float(m.group(2)), int(m.group(3))) for m in
            (re.match(r"ep(\d+) loss=\S+ dev F1\(±0.5s\)=([\d.]+) .* (\d+)s$", ln.strip()) for ln in open(log, encoding="utf-8")) if m]


def main():
    R, notes = {}, []
    for run in RUNS:
        mj = json.load(open(os.path.join(run["mdir"], "metrics.json"), encoding="utf-8"))
        ix = json.load(open(run["truth"], encoding="utf-8"))
        ev, nsong = evaluate(load_pred(run["mdir"]), ix)
        # 自己確認（metrics.json）＝5技法は技法ごと、合計は5技法の和で照らす（c2 の metrics の ALL は6技法なので和で作る）
        nchk = 0
        for tol in TOLS:
            mm = mj["metrics"][f"test_tol{tol}"]
            for c in TYPES:
                e = ev[tol][c]
                if (e["true"], e["pred"], e["hit"]) != (mm[c]["true"], mm[c]["pred"], mm[c]["hit"]) or abs(round(e["f1"], 3) - mm[c]["f1"]) > 0.0005:
                    raise SystemExit(f"止める＝自己確認の不一致 {run['key']} ±{tol} {c} 照合し直し {e} ／ metrics.json {mm[c]}")
                nchk += 1
            s = [sum(mm[c][k] for c in TYPES) for k in ("true", "pred", "hit")]
            e = ev[tol]["ALL"]
            if s != [e["true"], e["pred"], e["hit"]]:
                raise SystemExit(f"止める＝合計の不一致 {run['key']} ±{tol} {s} ／ {e}")
            nchk += 1
            if run["key"] == "c3":
                if (e["true"], e["pred"], e["hit"]) != (mm["ALL"]["true"], mm["ALL"]["pred"], mm["ALL"]["hit"]) or abs(round(e["f1"], 3) - mm["ALL"]["f1"]) > 0.0005:
                    raise SystemExit(f"止める＝ALL の不一致 c3 ±{tol} {e} ／ {mm['ALL']}")
        notes.append(f"自己確認 {run['key']}＝metrics.json と一致（2許容×6行×正解・出した・当てた・F・照らした数 {nchk}）")
        R[run["key"]] = dict(run=run, ev=ev, nsong=nsong, best_ep=mj["best"][1], thr=mj["thr"], sec=mj.get("elapsed_sec"), curve=dev_curve(run["log"]))
    for n in notes:
        print(n)

    c3, c2 = R["c3"], R["c2"]
    stem = os.path.join(OUT, "技法別の表_c3_5tech_2026-11-05")
    with open(stem + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["モデル", "許容秒", "技法", "正解の数", "出した数", "当てた数", "余計に出した数", "見落とした数"])
        for k in ("c3", "c2"):
            for tol in TOLS:
                for c in TYPES + ["ALL"]:
                    e = R[k]["ev"][tol][c]
                    w.writerow([R[k]["run"]["name"], tol, JP[c], e["true"], e["pred"], e["hit"], e["extra"], e["miss"]])
    with open(os.path.join(OUT, "技法別の表_参考F_c3_5tech_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["モデル", "許容秒", "技法", "適合率", "再現率", "F値"])
        for k in ("c3", "c2"):
            for tol in TOLS:
                for c in TYPES + ["ALL"]:
                    e = R[k]["ev"][tol][c]
                    w.writerow([R[k]["run"]["name"], tol, JP[c], round(e["precision"], 3), round(e["recall"], 3), round(e["f1"], 3)])

    L = ["# 技法ごとの表（5技法・全体1つの遅れ 1.6722秒で付け直した正解）2026-11-05", "",
         "**Opus 参考値・seed 1本（seed 0）・本人未検証**", "",
         f"試験＝{TEST} の {c3['nsong']}曲（学習に使っていない曲）。照合＝span の内側・技法ごとの貪欲一致（今朝の c2-shift2 と同じ）。",
         "学習＝8エポック・seed 0・dev 20260911（30曲）でしきい値と採用エポックを決定・学習 150曲（今朝と同じ分け方）。",
         "余計に出した数＝出した数−当てた数。見落とした数＝正解の数−当てた数。", ""]
    for tol in TOLS:
        L += [f"## 今回のモデル・許容±{tol}秒" + ("（主）" if tol == 0.5 else "（参考）"), "",
              "| 技法 | 正解の数 | 出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|"]
        for c in TYPES + ["ALL"]:
            e = c3["ev"][tol][c]
            L.append(f"| {JP[c]} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
        L.append("")
    L += ["## 並べて見せる物＝今朝の c2-shift2 との比べ（許容±0.5秒）", "",
          "| 技法 | 正解の数（今回／今朝） | 出した数（今回／今朝） | 当てた数（今回／今朝） | 余計に出した数（今回／今朝） | 見落とした数（今回／今朝） |", "|---|---|---|---|---|---|"]
    for c in TYPES + ["ALL"]:
        a, b = c3["ev"][0.5][c], c2["ev"][0.5][c]
        L.append(f"| {JP[c]} | {a['true']}／{b['true']} | {a['pred']}／{b['pred']} | {a['hit']}／{b['hit']} | {a['extra']}／{b['extra']} | {a['miss']}／{b['miss']} |")
    L += ["", "注記",
          "- 今回＝ビブラートを除く5技法で学習（出力5本）・正解は全技法共通の遅れ 1.6722秒で付け直した物。",
          "- 今朝＝6技法（ビブラート込み）で学習・正解は技法ごとの遅れ（しゃくり 1.641・こぶし 1.942・フォール 1.899・アクセント 1.618・ハンマリング 1.623秒）。今朝の行は同じ5技法だけを抜き出した数（合計も5技法の和）。",
          "- 技法の数が違う学習（5技法と6技法）。遅れの付け方だけの差ではない。",
          ("- 正解の数＝今回と今朝で5技法とも同じ（遅れの量の違いで span の外へ出た件なし）。" if all(c3["ev"][0.5][c]["true"] == c2["ev"][0.5][c]["true"] for c in TYPES)
           else "- 正解の数の差＝遅れの量が技法で違う分、span の内側に入る件が変わった差。"), ""]
    L += ["## 参考＝適合率・再現率・F値", "", "| モデル | 許容 | " + " | ".join(JP[c] for c in TYPES + ["ALL"]) + " |", "|---|---|" + "---|" * 6]
    for k in ("c3", "c2"):
        for tol in TOLS:
            e = R[k]["ev"][tol]
            L.append(f"| {R[k]['run']['name']} | ±{tol}秒 | " + " | ".join(f"F {e[c]['f1']:.3f}（適合 {e[c]['precision']:.3f}／再現 {e[c]['recall']:.3f}）" for c in TYPES + ["ALL"]) + " |")
    L += ["", "## 学習の記録", ""]
    for k in ("c3", "c2"):
        x = R[k]
        L.append(f"- {x['run']['name']}＝採用エポック {x['best_ep']}・学習の所要 {x['sec']}秒・dev F（±0.5秒・各エポック）＝" + "・".join(f"{f:.3f}" for _, f, _ in x["curve"]))
        L.append("  しきい値（dev）＝" + "・".join(f"{JP[c]} {x['thr'][c]:.3f}" for c in TYPES))
    L += ["", "自己確認＝" + "／".join(notes)]
    open(stem + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    for k in ("c3", "c2"):
        for tol in TOLS:
            e = R[k]["ev"][tol]
            print(f"{k} ±{tol}: " + " ".join(f"{c[:4]} {e[c]['true']}/{e[c]['pred']}/{e[c]['hit']}/{e[c]['extra']}/{e[c]['miss']} F{e[c]['f1']:.3f}" for c in TYPES + ["ALL"]))
    print("書いた＝", stem + ".md")


if __name__ == "__main__":
    main()
