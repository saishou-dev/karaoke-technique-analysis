# -*- coding: utf-8 -*-
r"""C2 補正比較の表を作る（2026-11-04・研究 D2 の一歩目）。学習も推論もしない＝保存済みの予測（pred_test.csv）を正解と照合し直すだけ。

評価の3通り（＋参考）。
  (i)   現行モデル × 元の正解        ＝基準（out/c2a_base_8ep＝10/23・Windows で学習・10/28 学習曲線の 100%）
  (ii)  補正モデル × 補正した正解    ＝out/c2_shift_20261104/c2_shift_8ep_s0（WSL で学習）
  (iii) 現行モデル × 補正した正解    ＝ずらすだけで何が起きるか（同じ予測を補正した正解で数え直す）
  参考 ＝WSL で再現した現行（c2_base_wsl_8ep_s0）× 元／補正・補正モデル × 元の正解

照合の決まり＝学習台本 train_crnn_refdiff_20261023.py の evaluate と同じ。
  試験＝20260924 の28曲・曲ごとの span（index の採点区間）の内側だけ・技法ごとに近い順の貪欲一致・許容 ±0.5／±1.0／±2.0 秒。
  件数の一致度＝1 − Σ|予測件数−正解件数| ÷ Σ正解件数（曲×技法ごと・span の内側・10/28 学習曲線の表と同じ）。
  補助＝観測できた区間（song_valid_range の exclude_band）で切った ±0.5秒 の F値と件数の一致度（lag 0）。
  自己確認＝各モデル×自分の学習した正解は、metrics.json の値と一致しなければ止める。
seed の揺れ＝10/24 の現行 種0/1/2 の範囲（±0.5秒 0.003・±1.0秒 0.018・±2.0秒 0.036）。指示どおり 0.003 との比較を1列で示す。

使い方:  python eval_c2_shift_20261104.py
出力＝out/c2_shift_20261104/C2補正比較_2026-11-04.csv ／ .md
"""
import csv
import json
import os
import sys
import collections

if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "scripts"))
from song_valid_range_20261015 import clip_span  # noqa: E402

OUT = os.path.join(ROOT, "out", "c2_shift_20261104")
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング"}
TOLS = (0.5, 1.0, 2.0)
SEED_RANGE = {0.5: 0.003, 1.0: 0.018, 2.0: 0.036}
TEST = "20260924"
MODELS = {
    "現行（10/23 Windows）": os.path.join(ROOT, "out", "c2a_base_8ep"),
    "現行の再現（WSL）": os.path.join(OUT, "c2_base_wsl_8ep_s0"),
    "補正モデル（WSL）": os.path.join(OUT, "c2_shift_8ep_s0"),
}
TRUTHS = {"元の正解": os.path.join(ROOT, "index.json"),
          "補正した正解": os.path.join(OUT, "labels", "index_shift_20261104.json")}
# (行の名前, モデル, 正解, metrics.json と一致すべきか)
ROWS = [("(i) 現行×元の正解（基準）", "現行（10/23 Windows）", "元の正解", True),
        ("(ii) 補正モデル×補正した正解", "補正モデル（WSL）", "補正した正解", True),
        ("(iii) 現行×補正した正解", "現行（10/23 Windows）", "補正した正解", False),
        ("参考 再現（WSL）×元の正解", "現行の再現（WSL）", "元の正解", True),
        ("参考 再現（WSL）×補正した正解", "現行の再現（WSL）", "補正した正解", False),
        ("参考 補正モデル×元の正解", "補正モデル（WSL）", "元の正解", False)]


def match(pred, truth, tol):
    used = set()
    hit = 0
    for p in sorted(pred):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - p) > tol:
                continue
            if best is None or abs(t - p) < abs(truth[best] - p):
                best = k
        if best is not None:
            used.add(best)
            hit += 1
    return hit


def prf(h, npred, ntrue):
    p = h / npred if npred else 0.0
    r = h / ntrue if ntrue else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def load_pred(mdir):
    pred = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(mdir, "pred_test.csv"), encoding="utf-8-sig")):
        pred[(r["video"], r["type"])].append(float(r["wav_time_sec"]))
    return pred


def evaluate(pred, index, mode="span"):
    vids = [d for d in index if d["session"] == TEST]
    res = {}
    for tol in TOLS:
        agg = {c: [0, 0, 0, 0] for c in TYPES}
        for d in vids:
            v = d["video"]
            if mode == "span":
                lo, hi = d["span"]
            else:
                lo, hi = clip_span(v, d["span"], float(d.get("offset", 0.0)), "exclude_band")[0]
            for c in TYPES:
                tr = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]
                pr = [t for t in pred.get((v, c), []) if lo <= t <= hi]
                agg[c][0] += match(pr, tr, tol)
                agg[c][1] += len(pr)
                agg[c][2] += len(tr)
                agg[c][3] += abs(len(pr) - len(tr))
        out = {}
        for c in TYPES:
            h, npd, nt, ce = agg[c]
            p, r, f = prf(h, npd, nt)
            out[c] = dict(hit=h, pred=npd, true=nt, precision=p, recall=r, f1=f, agree=(1 - ce / nt) if nt else float("nan"))
        H = sum(x[0] for x in agg.values())
        NP = sum(x[1] for x in agg.values())
        NT = sum(x[2] for x in agg.values())
        CE = sum(x[3] for x in agg.values())
        p, r, f = prf(H, NP, NT)
        out["ALL"] = dict(hit=H, pred=NP, true=NT, precision=p, recall=r, f1=f, agree=1 - CE / NT)
        res[tol] = out
    return res


def inout(d, rng):
    return "内" if abs(round(d, 3)) <= rng else "外"


def main():
    truths = {k: json.load(open(p, encoding="utf-8")) for k, p in TRUTHS.items()}
    preds, metrics = {}, {}
    for k, mdir in MODELS.items():
        if os.path.exists(os.path.join(mdir, "pred_test.csv")) and os.path.exists(os.path.join(mdir, "metrics.json")):
            preds[k] = load_pred(mdir)
            metrics[k] = json.load(open(os.path.join(mdir, "metrics.json"), encoding="utf-8"))
        else:
            print(f"未完了のため飛ばす＝{k}（{mdir}）")
    R = []
    nchk = 0
    for name, mk, tk, must in ROWS:
        if mk not in preds:
            continue
        ev = evaluate(preds[mk], truths[tk])
        evb = evaluate(preds[mk], truths[tk], "exclude_band")
        if must:
            nchk += 1
            for tol in TOLS:
                mm = metrics[mk]["metrics"][f"test_tol{tol}"]
                for c in TYPES + ["ALL"]:
                    a, b = round(ev[tol][c]["f1"], 3), mm[c]["f1"]
                    if abs(a - b) > 0.0005 or ev[tol][c]["true"] != mm[c]["true"] or ev[tol][c]["hit"] != mm[c]["hit"]:
                        raise SystemExit(f"止める＝自己確認の不一致 {name} ±{tol} {c} 照合し直し {a} ／ metrics.json {b}")
        R.append(dict(name=name, model=mk, truth=tk, ev=ev, evb=evb, best_ep=metrics[mk]["best"][1],
                      sec=metrics[mk].get("elapsed_sec"), thr=metrics[mk]["thr"]))
    print(f"自己確認＝metrics.json と一致（{nchk} 行・3許容×7列）")
    base = R[0]["ev"]
    wsl = next((r["ev"] for r in R if r["name"].startswith("参考 再現（WSL）×元")), None)
    for r in R:
        r["dw"] = (r["ev"][0.5]["ALL"]["f1"] - wsl[0.5]["ALL"]["f1"]) if wsl else None
        r["d"] = {tol: r["ev"][tol]["ALL"]["f1"] - base[tol]["ALL"]["f1"] for tol in TOLS}
        r["inout"] = inout(r["d"][0.5], 0.003)
        r["inout_tol"] = "・".join(f"±{tol} {inout(r['d'][tol], SEED_RANGE[tol])}" for tol in TOLS)

    cpath = os.path.join(OUT, "C2補正比較_2026-11-04.csv")
    with open(cpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["評価", "モデル", "正解", "F値±0.5秒", "F値±1.0秒", "F値±2.0秒", "基準との差±0.5秒", "基準との差±1.0秒", "基準との差±2.0秒",
                "seed揺れ0.003の内外（±0.5秒）", "許容別のseed揺れ（0.003/0.018/0.036）の内外", "正解件数", "予測件数", "当たり±0.5秒",
                "再現率±0.5秒", "適合率±0.5秒", "件数の一致度", "WSL再現×元との差±0.5秒", "F値±0.5秒_exclude_band", "件数の一致度_exclude_band", "採用エポック", "学習の所要秒"]
        head += [f"{JP[c]}のF値±{tol}秒" for tol in TOLS for c in TYPES] + [f"{JP[c]}の件数の一致度" for c in TYPES]
        w.writerow(head)
        for r in R:
            e = r["ev"]
            w.writerow([r["name"], r["model"], r["truth"]] + [round(e[t]["ALL"]["f1"], 3) for t in TOLS] + [round(r["d"][t], 3) for t in TOLS]
                       + [r["inout"], r["inout_tol"], e[0.5]["ALL"]["true"], e[0.5]["ALL"]["pred"], e[0.5]["ALL"]["hit"],
                          round(e[0.5]["ALL"]["recall"], 3), round(e[0.5]["ALL"]["precision"], 3), round(e[0.5]["ALL"]["agree"], 3),
                          "" if r["dw"] is None else round(r["dw"], 3), round(r["evb"][0.5]["ALL"]["f1"], 3), round(r["evb"][0.5]["ALL"]["agree"], 3), r["best_ep"], r["sec"]]
                       + [round(e[t][c]["f1"], 3) for t in TOLS for c in TYPES] + [round(e[0.5][c]["agree"], 3) for c in TYPES])

    shift_rows = list(csv.reader(open(os.path.join(OUT, "labels", "shift_table_20261104.csv"), encoding="utf-8-sig")))
    L = ["# C2 補正比較（検出器・正解時刻を技法マーク側へ補正して学習し直す）2026-11-04", "",
         "**Opus 参考値・本人未検証**", "",
         "条件＝8エポック・seed 0・現行の入力（お手本の音程差なし）・しきい値は dev（20260911・30曲）で決定・試験＝20260924 の28曲（未見）。",
         "補正＝正解（G1）の各行の時刻を技法別の中央値だけ前へ（材料48 2-2・9/24 と 10/14 の対応件を合わせた中央値）。span（採点区間）は元のまま。",
         "照合＝学習台本の evaluate と同じ（span の内側・貪欲一致）。件数の一致度＝1 − Σ|予測−正解| ÷ Σ正解（曲×技法）。",
         "基準＝(i)。補正モデルは WSL で学習したので、同じ WSL で回した現行の再現（参考 再現（WSL）×元）との差も1列で示す（環境だけで ±0.5秒 −0.004 動いた）。",
         "seed 揺れ＝10/24 の現行 種0/1/2 の範囲（±0.5秒 0.003・±1.0秒 0.018・±2.0秒 0.036）。", "",
         "## 全体", "",
         "| 評価 | F値±0.5秒 | F値±1.0秒 | F値±2.0秒 | 基準との差（±0.5／±1.0／±2.0） | seed揺れ0.003の内外（±0.5秒） | 許容別のseed揺れの内外 | 当たり／予測／正解（±0.5秒） | 再現率 | 適合率 | 件数の一致度 | WSL再現×元との差（±0.5秒） | ±0.5秒 exclude_band | 採用エポック | 学習の所要秒 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in R:
        e = r["ev"]
        L.append(f"| {r['name']} | {e[0.5]['ALL']['f1']:.3f} | {e[1.0]['ALL']['f1']:.3f} | {e[2.0]['ALL']['f1']:.3f} | "
                 f"{r['d'][0.5]:+.3f}／{r['d'][1.0]:+.3f}／{r['d'][2.0]:+.3f} | {r['inout']} | {r['inout_tol']} | "
                 f"{e[0.5]['ALL']['hit']}／{e[0.5]['ALL']['pred']}／{e[0.5]['ALL']['true']} | {e[0.5]['ALL']['recall']:.3f} | {e[0.5]['ALL']['precision']:.3f} | "
                 f"{e[0.5]['ALL']['agree']:.3f} | {'—' if r['dw'] is None else format(r['dw'], '+.3f')} | {r['evb'][0.5]['ALL']['f1']:.3f} | {r['best_ep']} | {r['sec']} |")
    for tol in TOLS:
        L += ["", f"## 技法ごとの F値（許容±{tol}秒）", "", "| 評価 | " + " | ".join(JP[c] for c in TYPES) + " | 全体 |",
              "|---|" + "---|" * (len(TYPES) + 1)]
        for r in R:
            L.append(f"| {r['name']} | " + " | ".join(f"{r['ev'][tol][c]['f1']:.3f}" for c in TYPES) + f" | {r['ev'][tol]['ALL']['f1']:.3f} |")
    L += ["", "## 技法ごとの件数の一致度（span の内側）", "", "| 評価 | " + " | ".join(JP[c] for c in TYPES) + " | 全体 |",
          "|---|" + "---|" * (len(TYPES) + 1)]
    for r in R:
        L.append(f"| {r['name']} | " + " | ".join(f"{r['ev'][0.5][c]['agree']:.3f}" for c in TYPES) + f" | {r['ev'][0.5]['ALL']['agree']:.3f} |")
    L += ["", "## 補正量（秒・前へ）", "", "| " + " | ".join(shift_rows[0]) + " |", "|" + "---|" * len(shift_rows[0])]
    for s in shift_rows[1:]:
        L.append("| " + " | ".join(s) + " |")
    L += ["", "## しきい値（dev で決定）", ""]
    seen = set()
    for r in R:
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        L.append(f"- {r['model']}＝" + "・".join(f"{JP[c]} {r['thr'][c]:.3f}" for c in TYPES))
    open(os.path.join(OUT, "C2補正比較_2026-11-04.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    for r in R:
        e = r["ev"]
        print(f"{r['name']}: F {e[0.5]['ALL']['f1']:.3f}/{e[1.0]['ALL']['f1']:.3f}/{e[2.0]['ALL']['f1']:.3f} "
              f"d05 {r['d'][0.5]:+.3f} {r['inout']} agree {e[0.5]['ALL']['agree']:.3f} exb {r['evb'][0.5]['ALL']['f1']:.3f}")
    print("書いた＝", cpath)


if __name__ == "__main__":
    main()
