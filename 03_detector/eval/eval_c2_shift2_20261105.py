# -*- coding: utf-8 -*-
r"""C2 補正比較2 の表を作る（2026-11-05・c2-shift2・eval_c2_shift_20261104.py と同型）。学習も推論もしない＝保存済みの pred_test.csv を照合し直すだけ。

行（すべて試験＝20260924 の28曲・span の内側・技法ごとの貪欲一致・±0.5／±1.0／±2.0秒）
  (a) 今回のモデル（c2_shift2_8ep_s0）× 今回の正解（index_shift2_20261105＝カウンター−M）  ＝本命・metrics.json と一致しなければ止める
  (b) 今回のモデル × C の正解（index_shift_20261104＝11/4 の補正版）
  (c) 今回のモデル × 元の正解（index.json）
  (d) 11/4 の補正モデル（c2_shift_20261104/c2_shift_8ep_s0）× 今回の正解（ずらすだけ）
  引用（再計算しない）＝11/4 の表 C2補正比較_2026-11-04.csv の (i)・(ii)・WSL 再現
ビブラートの追加＝予測が「複合の区間」（#57 labels_composite_20260924.csv の vib_start〜vib_end）の中か、前後の許容内に入れば当たり
  （1010 報告の根拠C と同じ作法＝dc_composite_20261105.py の basis_c を写した。index.json の 9/24 イベントを labels と対応・
  対応しない件は「埋め（印なし）」の技法別中央値でずらす）。11/4 の補正モデルでも計算し、1010 報告の値（±0.5 0.214・±1.0 0.238）と一致を確かめる。
使い方:  python eval_c2_shift2_20261105.py
出力＝out/c2_shift2_20261105/C2補正比較2_2026-11-05.csv ／ .md
"""
import csv, json, os, sys, collections
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "c2_shift2_20261105")
OUT4 = os.path.join(ROOT, "out", "c2_shift_20261104")
COMP = r"G:\本人確認待ち\#57_正解時刻の複合_DとC_2026-11-05\labels_composite_20260924.csv"
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング", "ALL": "全体"}
TOLS = (0.5, 1.0, 2.0)
TEST = "20260924"
MODELS = {"今回のモデル": os.path.join(OUT, "c2_shift2_8ep_s0"), "11/4 の補正モデル": os.path.join(OUT4, "c2_shift_8ep_s0")}
TRUTHS = {"今回の正解": os.path.join(OUT, "labels", "index_shift2_20261105.json"),
          "C の正解": os.path.join(OUT4, "labels", "index_shift_20261104.json"),
          "元の正解": os.path.join(ROOT, "index.json")}
ROWS = [("(a) 今回のモデル×今回の正解（本命）", "今回のモデル", "今回の正解", True),
        ("(b) 今回のモデル×C の正解", "今回のモデル", "C の正解", False),
        ("(c) 今回のモデル×元の正解", "今回のモデル", "元の正解", False),
        ("(d) 11/4 の補正モデル×今回の正解", "11/4 の補正モデル", "今回の正解", False)]
QUOTE = [("(i) 現行×元の正解（基準）", "引用 (i) 現行×元の正解"), ("(ii) 補正モデル×補正した正解", "引用 (ii) 11/4 補正モデル×C の正解"),
         ("参考 再現（WSL）×元の正解", "引用 WSL 再現×元の正解")]


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


def match_iv(pred, ivs, tol):
    used, hit = set(), 0
    dist = lambda p, a, b: 0.0 if a <= p <= b else min(abs(p - a), abs(p - b))
    for p in sorted(pred):
        best = bd = None
        for k, (a, b) in enumerate(ivs):
            if k in used:
                continue
            dd = dist(p, a, b)
            if dd > tol:
                continue
            if best is None or dd < bd:
                best, bd = k, dd
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
        agg = {c: [0, 0, 0, 0] for c in TYPES}
        for d in vids:
            v = d["video"]; lo, hi = d["span"]
            for c in TYPES:
                tr = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]
                pr = [t for t in pred.get((v, c), []) if lo <= t <= hi]
                agg[c][0] += match(pr, tr, tol); agg[c][1] += len(pr); agg[c][2] += len(tr); agg[c][3] += abs(len(pr) - len(tr))
        out = {}
        for c in TYPES:
            h, npd, nt, ce = agg[c]; p, r, f = prf(h, npd, nt)
            out[c] = dict(hit=h, pred=npd, true=nt, precision=p, recall=r, f1=f, agree=(1 - ce / nt) if nt else float("nan"))
        H, NP, NT, CE = (sum(x[i] for x in agg.values()) for i in range(4))
        p, r, f = prf(H, NP, NT)
        out["ALL"] = dict(hit=H, pred=NP, true=NT, precision=p, recall=r, f1=f, agree=1 - CE / NT)
        res[tol] = out
    return res


def composite_vib_intervals(index):
    """1010 報告 根拠C と同じ作り（dc_composite_20261105.py basis_c のビブラート部分）。{video: [(start, end), ...]}"""
    L = pd.read_csv(COMP, encoding="utf-8-sig")
    Ls = L[L.session == int(TEST)]
    x = Ls[(Ls.counter == "vibrato") & (Ls.src == "埋め（印なし）")]
    fb_s = float((x.video_time_sec - x.vib_start).median()); fb_e = float((x.video_time_sec - x.C).median())
    comp, nmatch, nfb = {}, 0, 0
    for d in index:
        if d["session"] != TEST:
            continue
        v = d["video"]; off = float(d["offset"]); g = Ls[(Ls.video == v) & (Ls.counter == "vibrato")]
        iv = []
        for tch, t in d["events"]:
            if tch != "vibrato":
                continue
            k = (g.video_time_sec + off - t).abs()
            if len(k) and k.min() < 0.02:
                e = g.loc[k.idxmin()]; nmatch += 1
                iv.append((t - (e.video_time_sec - e.vib_start), t - (e.video_time_sec - e.vib_end)))
            else:
                nfb += 1; iv.append((t - fb_s, t - fb_e))
        comp[v] = iv
    return comp, nmatch, nfb


def eval_vib_comp(pred, index, comp):
    res = {}
    for tol in TOLS:
        h = npd = nt = 0
        for d in index:
            if d["session"] != TEST:
                continue
            v = d["video"]; lo, hi = d["span"]
            pr = [t for t in pred.get((v, "vibrato"), []) if lo <= t <= hi]
            iv = [(a, b) for a, b in comp[v] if b >= lo and a <= hi]
            h += match_iv(pr, iv, tol); npd += len(pr); nt += len(iv)
        p, r, f = prf(h, npd, nt)
        res[tol] = dict(hit=h, pred=npd, true=nt, precision=p, recall=r, f1=f)
    return res


def main():
    truths = {k: json.load(open(p, encoding="utf-8")) for k, p in TRUTHS.items()}
    preds, metrics = {}, {}
    for k, mdir in MODELS.items():
        if os.path.exists(os.path.join(mdir, "pred_test.csv")) and os.path.exists(os.path.join(mdir, "metrics.json")):
            preds[k] = load_pred(mdir); metrics[k] = json.load(open(os.path.join(mdir, "metrics.json"), encoding="utf-8"))
        else:
            print(f"未完了のため飛ばす＝{k}（{mdir}）")
    R, nchk = [], 0
    for name, mk, tk, must in ROWS:
        if mk not in preds:
            continue
        ev = evaluate(preds[mk], truths[tk])
        if must:
            nchk += 1
            for tol in TOLS:
                mm = metrics[mk]["metrics"][f"test_tol{tol}"]
                for c in TYPES + ["ALL"]:
                    a, b = round(ev[tol][c]["f1"], 3), mm[c]["f1"]
                    if abs(a - b) > 0.0005 or ev[tol][c]["true"] != mm[c]["true"] or ev[tol][c]["hit"] != mm[c]["hit"]:
                        raise SystemExit(f"止める＝自己確認の不一致 {name} ±{tol} {c} 照合し直し {a} ／ metrics.json {b}")
        R.append(dict(name=name, model=mk, truth=tk, ev=ev, best_ep=metrics[mk]["best"][1], sec=metrics[mk].get("elapsed_sec"), thr=metrics[mk]["thr"]))
    print(f"自己確認＝metrics.json と一致（{nchk} 行・3許容×7列）")
    # 引用（11/4 の表・再計算しない）
    q = {r["評価"]: r for r in csv.DictReader(open(os.path.join(OUT4, "C2補正比較_2026-11-04.csv"), encoding="utf-8-sig"))}
    Q = []
    for key, lab in QUOTE:
        r = q[key]
        Q.append(dict(name=lab, f={t: float(r[f"F値±{t}秒"]) for t in TOLS}, tech={t: {c: float(r[f"{JP[c]}のF値±{t}秒"]) for c in TYPES} for t in TOLS},
                      hit=r["当たり±0.5秒"], pred=r["予測件数"], true=r["正解件数"], prec=float(r["適合率±0.5秒"]), agree=float(r["件数の一致度"]), ep=r["採用エポック"]))
    base_ii = Q[1]["f"]
    # ビブラート 複合の区間
    comp, nmatch, nfb = composite_vib_intervals(truths["元の正解"])
    VC = {k: eval_vib_comp(preds[k], truths["元の正解"], comp) for k in preds}
    if "11/4 の補正モデル" in VC:
        v4 = VC["11/4 の補正モデル"]
        ok = abs(round(v4[0.5]["f1"], 3) - 0.214) < 0.0006 and abs(round(v4[1.0]["f1"], 3) - 0.238) < 0.0006
        print(f"複合の区間の再現（11/4 の補正モデル）＝±0.5 {v4[0.5]['f1']:.3f}・±1.0 {v4[1.0]['f1']:.3f}（1010 報告 0.214・0.238）{'一致' if ok else '不一致'}・対応 {nmatch}・対応なし {nfb}")
        if not ok:
            raise SystemExit("止める＝複合の区間の作法が 1010 報告と一致しない")

    cpath = os.path.join(OUT, "C2補正比較2_2026-11-05.csv")
    with open(cpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["評価", "モデル", "正解", "F値±0.5秒", "F値±1.0秒", "F値±2.0秒", "(ii)との差±0.5秒", "(ii)との差±1.0秒", "(ii)との差±2.0秒",
                "正解件数", "予測件数", "当たり±0.5秒", "再現率±0.5秒", "適合率±0.5秒", "適合率±1.0秒", "適合率±2.0秒", "件数の一致度", "採用エポック", "学習の所要秒"]
        head += [f"{JP[c]}のF値±{t}秒" for t in TOLS for c in TYPES] + [f"{JP[c]}の適合率±{t}秒" for t in TOLS for c in TYPES] + [f"{JP[c]}の予測件数" for c in TYPES] + [f"{JP[c]}の正解件数" for c in TYPES]
        w.writerow(head)
        for r in R:
            e = r["ev"]
            w.writerow([r["name"], r["model"], r["truth"]] + [round(e[t]["ALL"]["f1"], 3) for t in TOLS] + [round(e[t]["ALL"]["f1"] - base_ii[t], 3) for t in TOLS]
                       + [e[0.5]["ALL"]["true"], e[0.5]["ALL"]["pred"], e[0.5]["ALL"]["hit"], round(e[0.5]["ALL"]["recall"], 3)]
                       + [round(e[t]["ALL"]["precision"], 3) for t in TOLS] + [round(e[0.5]["ALL"]["agree"], 3), r["best_ep"], r["sec"]]
                       + [round(e[t][c]["f1"], 3) for t in TOLS for c in TYPES] + [round(e[t][c]["precision"], 3) for t in TOLS for c in TYPES]
                       + [e[0.5][c]["pred"] for c in TYPES] + [e[0.5][c]["true"] for c in TYPES])
        for x in Q:
            w.writerow([x["name"], "11/4 の表から引用", ""] + [x["f"][t] for t in TOLS] + [round(x["f"][t] - base_ii[t], 3) for t in TOLS]
                       + [x["true"], x["pred"], x["hit"], "", x["prec"], "", "", x["agree"], x["ep"], ""] + [x["tech"][t][c] for t in TOLS for c in TYPES])
        for k, v in VC.items():
            w.writerow([f"ビブラート 複合の区間（{k}）", k, "複合の区間（前後の遊び＝許容）"] + [round(v[t]["f1"], 3) for t in TOLS] + ["", "", ""]
                       + [v[0.5]["true"], v[0.5]["pred"], v[0.5]["hit"], round(v[0.5]["recall"], 3)] + [round(v[t]["precision"], 3) for t in TOLS])

    shift_rows = list(csv.reader(open(os.path.join(OUT, "labels", "shift_table_20261105.csv"), encoding="utf-8-sig")))
    L = ["# C2 補正比較2（正解を D 側＝カウンター−M へずらして学習し直す）2026-11-05", "", "**Opus 参考値・seed 1本（seed 0）・本人未検証**", "",
         "条件＝WSL・8エポック・seed 0・現行の入力・しきい値は dev（20260911・30曲）・試験＝20260924 の28曲。照合＝span の内側・技法ごとの貪欲一致（11/4 と同じ）。",
         "M＝カウンター時刻−D の技法別中央値（9/24 と 10/14 の合算・D がある件）。C の正解＝11/4 の補正版（約0.85秒前）。引用行は 11/4 の表の値（再計算なし）。", "",
         "## 全体", "", "| 評価 | F値±0.5秒 | F値±1.0秒 | F値±2.0秒 | (ii) との差（±0.5／±1.0／±2.0） | 当たり／予測／正解（±0.5秒） | 再現率 | 適合率（±0.5／±1.0／±2.0） | 件数の一致度 | 採用エポック |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in R:
        e = r["ev"]
        L.append(f"| {r['name']} | {e[0.5]['ALL']['f1']:.3f} | {e[1.0]['ALL']['f1']:.3f} | {e[2.0]['ALL']['f1']:.3f} | "
                 + "／".join(f"{e[t]['ALL']['f1'] - base_ii[t]:+.3f}" for t in TOLS)
                 + f" | {e[0.5]['ALL']['hit']}／{e[0.5]['ALL']['pred']}／{e[0.5]['ALL']['true']} | {e[0.5]['ALL']['recall']:.3f} | "
                 + "／".join(f"{e[t]['ALL']['precision']:.3f}" for t in TOLS) + f" | {e[0.5]['ALL']['agree']:.3f} | {r['best_ep']} |")
    for x in Q:
        L.append(f"| {x['name']} | {x['f'][0.5]:.3f} | {x['f'][1.0]:.3f} | {x['f'][2.0]:.3f} | " + "／".join(f"{x['f'][t] - base_ii[t]:+.3f}" for t in TOLS)
                 + f" | {x['hit']}／{x['pred']}／{x['true']} | — | {x['prec']:.3f}／—／— | {x['agree']:.3f} | {x['ep']} |")
    for t in TOLS:
        L += ["", f"## 技法ごとの F値（許容±{t}秒）・括弧＝適合率／予測件数", "", "| 評価 | " + " | ".join(JP[c] for c in TYPES) + " | 全体 |", "|---|" + "---|" * (len(TYPES) + 1)]
        for r in R:
            e = r["ev"][t]
            L.append(f"| {r['name']} | " + " | ".join(f"{e[c]['f1']:.3f}（{e[c]['precision']:.3f}／{e[c]['pred']}）" for c in TYPES) + f" | {e['ALL']['f1']:.3f} |")
        for x in Q:
            L.append(f"| {x['name']} | " + " | ".join(f"{x['tech'][t][c]:.3f}" for c in TYPES) + f" | {x['f'][t]:.3f} |")
    L += ["", "## ビブラート＝複合の区間で採点（1010 報告 根拠C と同じ作法・区間の中か前後の許容内なら当たり）", "",
          f"複合の区間＝labels_composite_20260924.csv の vib_start〜vib_end。index.json の 9/24 ビブラートと対応 {nmatch}件・対応なし {nfb}件は埋め（印なし）の中央値でずらした。", "",
          "| モデル | ±0.5秒 F（適合／再現） | ±1.0秒 F | ±2.0秒 F | 当たり／予測／正解（±0.5秒） |", "|---|---|---|---|---|"]
    for k, v in VC.items():
        L.append(f"| {k} | {v[0.5]['f1']:.3f}（{v[0.5]['precision']:.3f}／{v[0.5]['recall']:.3f}） | {v[1.0]['f1']:.3f} | {v[2.0]['f1']:.3f} | {v[0.5]['hit']}／{v[0.5]['pred']}／{v[0.5]['true']} |")
    L += ["", "## M（秒・前へ）", "", "| " + " | ".join(shift_rows[0]) + " |", "|" + "---|" * len(shift_rows[0])]
    L += ["| " + " | ".join(s) + " |" for s in shift_rows[1:]]
    L += ["", "## しきい値（dev で決定）", ""]
    for k in preds:
        L.append(f"- {k}＝" + "・".join(f"{JP[c]} {metrics[k]['thr'][c]:.3f}" for c in TYPES))
    open(os.path.join(OUT, "C2補正比較2_2026-11-05.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    for r in R:
        e = r["ev"]
        print(f"{r['name']}: F {e[0.5]['ALL']['f1']:.3f}/{e[1.0]['ALL']['f1']:.3f}/{e[2.0]['ALL']['f1']:.3f} pred {e[0.5]['ALL']['pred']} prec {e[0.5]['ALL']['precision']:.3f} agree {e[0.5]['ALL']['agree']:.3f} "
              + " ".join(f"{c[:4]} {e[0.5][c]['f1']:.3f}" for c in TYPES))
    for k, v in VC.items():
        print(f"vib 複合 {k}: " + " ".join(f"±{t} {v[t]['f1']:.3f}" for t in TOLS))
    print("書いた＝", cpath)


if __name__ == "__main__":
    main()
