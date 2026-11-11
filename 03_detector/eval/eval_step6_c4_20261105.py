# -*- coding: utf-8 -*-
r"""手順6＝モデルの比較の表（2026-11-05 Opus・c4-compare 追補1）。学習も推論もしない＝保存済みの pred_test.csv を照合し直すだけ。
行＝モデル。列＝正解の数・モデルが出した数・当てた数・余計に出した数・見落とした数（合計と技法ごと・許容±0.5秒／±1.0秒は参考）・重みの数・学習の所要。
正解＝全体1つの遅れの正解（index_c3_5tech_20261105）。試験＝20260924 の28曲。照合＝span の内側・技法ごとの貪欲一致。
  基準 CRNN（c3_5tech_8ep_s0）・6a 決め事（rule_detect_c4 の pred_rule_test）・6b 畳み込みだけ・6c GRU だけ・6d 幅を半分・6e 幅を2倍・
  6f 技法ごとに1本ずつ（5本の pred_test をまとめて数える・重みと所要は5本の和）・
  6g 過去の候補（走らせ直さない＝各 metrics.json の数をそのまま写す・参考の行）＝v2a（モデルA・9/28）・v2d（吸着ラベル・9/28）・c2_shift2（今朝・6技法）。
重みの数＝model.pt の数の和（torch で読むだけ）。学習の所要＝metrics.json の elapsed_sec（学習台本の t0 から＝特徴の読み込みを除く）。
自己確認＝学習台本の metrics.json（test_tol0.5・1.0）と、照合し直した数を照らす（6f は技法ごと）。合わなければ止める。終わっていない本は飛ばす。
使い方: python eval_step6_c4_20261105.py
出力＝out/c4_compare_20261105/手順6_モデルの比較_c4_2026-11-05.md ・ .csv ・ 手順6_モデルの比較_参考F_c4_2026-11-05.csv
"""
import csv, json, os, sys
from common_c4_20261105 import ROOT, TYPES, JP, TEST, truth_of, load_index, load_pred_csv, count_pred, finish, resolve_mdir
NOROWS = []   # 追補6＝数を作れない打ち切りの本（名, 注記）

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
OUT = os.path.join(ROOT, "out", "c4_compare_20261105")
BASE = os.path.join(ROOT, "out", "c3_5tech_global_20261105", "c3_5tech_8ep_s0")
IXG = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
TOLS = (0.5, 1.0)
RUNS = [
    ("base", "基準 CRNN（畳み込み3層＋双方向 GRU 2層・幅 128）", [BASE]),
    ("rule", "6a 学習なしの決め事（声の高さと音量）", None),
    ("conv", "6b 畳み込みだけ（GRU を外す）", [os.path.join(OUT, "c4_conv_8ep_s0")]),
    ("gru", "6c GRU だけ（畳み込みを外す）", [os.path.join(OUT, "c4_gru_8ep_s0")]),
    ("half", "6d CRNN の幅を半分（64）", [os.path.join(OUT, "c4_half_8ep_s0")]),
    ("wide", "6e CRNN の幅を2倍（256）", [os.path.join(OUT, "c4_wide_8ep_s0")]),
    ("only", "6f 技法ごとに1本ずつ（5本）", [os.path.join(OUT, f"c4_only_{c}_8ep_s0") for c in TYPES]),
]
PAST = [
    ("v2a", "6g 参考 v2a＝モデルA（9/28・6技法で学習・当時の正解の時刻＝遅れなし）", os.path.join(ROOT, "out", "v2a")),
    ("v2d", "6g 参考 v2d＝吸着ラベル CRNN（9/28・6技法・正解を音の終わりに吸着・吸着した正解で数えた）", os.path.join(ROOT, "out", "v2d")),
    ("c2s2", "6g 参考 c2_shift2（11/5 朝・6技法で学習・技法ごとの遅れの正解で数えた）", os.path.join(ROOT, "out", "c2_shift2_20261105", "c2_shift2_8ep_s0")),
]


def n_params(mdir):
    try:
        import torch
        sd = torch.load(os.path.join(mdir, "model.pt"), map_location="cpu")
        return int(sum(v.numel() for k, v in sd.items() if not k.endswith("num_batches_tracked") and "running_" not in k))
    except Exception as e:
        return None


def main():
    tr = truth_of(load_index(IXG), TEST)
    R, notes, skipped = {}, [], []
    NOROWS.clear()
    for key, name, mdirs in RUNS:
        if key == "rule":
            p = os.path.join(OUT, "pred_rule_test_c4_2026-11-05.csv")
            if not os.path.exists(p):
                skipped.append(key); continue
            pred = load_pred_csv(p)
            js = json.load(open(os.path.join(OUT, "手順6a_決め事_c4_2026-11-05.json"), encoding="utf-8"))
            res = {tol: count_pred(pred, tr, tol) for tol in TOLS}
            for tol in TOLS:
                for c in TYPES + ["ALL"]:
                    m = js["R"][str(tol)][c]
                    if (m["true"], m["pred"], m["hit"]) != (res[tol][c]["true"], res[tol][c]["pred"], res[tol][c]["hit"]):
                        raise SystemExit(f"止める＝6a の不一致 ±{tol} {c}")
            notes.append("rule＝手順6a の json と一致（2許容×6行）")
            R[key] = dict(name=name, res=res, params="0（学習なし・決め事の数値 " + str(sum(len(v["params"]) for v in js["best"].values())) + "個）", sec="学習なし", made=set(js["best"]))
            continue
        rs = [resolve_mdir(m) for m in mdirs]   # 追補6＝打ち切りの本は partial_c4 へ振り替える
        if any(d is None for d, _ in rs):
            nts = [nt for d, nt in rs if d is None and nt]
            if nts:
                NOROWS.append((name, "・".join(sorted(set(nts)))))
            skipped.append(key); continue
        pnote = "・".join(f"{TYPES[i] if key == 'only' else ''}{nt}" for i, (d, nt) in enumerate(rs) if nt)
        if pnote:
            name = f"{name}［{pnote}］"
        mdirs = [d for d, _ in rs]
        if key == "only":
            pred = {}
            for c, m in zip(TYPES, mdirs):
                pc = load_pred_csv(os.path.join(m, "pred_test.csv"))
                pred.update({k: v for k, v in pc.items() if k[1] == c})
        else:
            pred = load_pred_csv(os.path.join(mdirs[0], "pred_test.csv"))
        res = {tol: count_pred(pred, tr, tol) for tol in TOLS}
        n = 0
        for i, m in enumerate(mdirs):
            mj = json.load(open(os.path.join(m, "metrics.json"), encoding="utf-8"))
            for tol in TOLS:
                mm = mj["metrics"][f"test_tol{tol}"]
                cs = [TYPES[i]] if key == "only" else TYPES + ["ALL"]
                for c in cs:
                    e = res[tol][c]
                    if (e["true"], e["pred"], e["hit"]) != (mm[c]["true"], mm[c]["pred"], mm[c]["hit"]):
                        raise SystemExit(f"止める＝自己確認の不一致 {key} {m} ±{tol} {c} {e} ／ {mm[c]}")
                    n += 1
        notes.append(f"{key}＝metrics.json と一致（{n}行）")
        ps = [n_params(m) for m in mdirs]; secs = [json.load(open(os.path.join(m, "metrics.json"), encoding="utf-8")).get("elapsed_sec") for m in mdirs]
        R[key] = dict(name=name, res=res, params=(f"{sum(ps):,}" + ("（5本の和）" if key == "only" else "")) if all(ps) else "読めず",
                      sec=(f"{sum(secs) / 60:.1f}分" + ("（5本の和）" if key == "only" else "")) if all(secs) else ("打ち切り（所要は連結ジョブのログ）" if pnote else "記録なし"),
                      best=[json.load(open(os.path.join(m, "metrics.json"), encoding="utf-8"))["best"][1] for m in mdirs])
    P = {}
    for key, name, m in PAST:
        mj = json.load(open(os.path.join(m, "metrics.json"), encoding="utf-8"))
        res = {}
        for tol in TOLS:
            mm = mj["metrics"][f"test_tol{tol}"]
            agg = {c: [mm[c]["hit"], mm[c]["pred"], mm[c]["true"]] for c in TYPES}
            res[tol] = finish(agg)
            res[tol]["vib"] = mm.get("vibrato")
        P[key] = dict(name=name, res=res, params=n_params(m), sec=mj.get("elapsed_sec"))

    with open(os.path.join(OUT, "手順6_モデルの比較_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["モデル", "許容秒", "技法", "正解の数", "モデルが出した数", "当てた数", "余計に出した数", "見落とした数", "重みの数", "学習の所要", "参考の行"])
        for src, flag in ((R, ""), (P, "参考（走らせ直していない）")):
            for k, x in src.items():
                for tol in TOLS:
                    for c in TYPES + ["ALL"]:
                        e = x["res"][tol][c]
                        made = (k != "rule") or c == "ALL" or c in x["made"]
                        w.writerow([x["name"], tol, JP[c], e["true"], e["pred"] if made else "作れず", e["hit"] if made else "作れず", e["extra"] if made else "作れず", e["miss"] if made else "作れず",
                                    x["params"] if x["params"] is not None else "読めず", x["sec"] if x["sec"] is not None else "記録なし", flag])
    with open(os.path.join(OUT, "手順6_モデルの比較_参考F_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["モデル", "許容秒", "技法", "適合率", "再現率", "F値"])
        for src in (R, P):
            for k, x in src.items():
                for tol in TOLS:
                    for c in TYPES + ["ALL"]:
                        e = x["res"][tol][c]; w.writerow([x["name"], tol, JP[c], round(e["precision"], 3), round(e["recall"], 3), round(e["f1"], 3)])

    L = ["# 手順6＝モデルの比較（5技法・全体1つの遅れの正解・試験 20260924 の28曲）2026-11-05", "", "**Opus 参考値・本人未検証**", "",
         "全部の行＝同じ正解・同じ28曲・同じ照合（span の内側・技法ごとの貪欲一致）。学習の行＝学習150曲・調整30曲・seed 0・8エポック。参考の行（6g）は除く。", ""]
    if skipped:
        L += ["まだ終わっていない本（表に無い）＝" + "・".join(skipped), ""]
    for tol in TOLS:
        L += [f"## 合計（許容±{tol}秒）" + ("（主）" if tol == 0.5 else "（参考）"), "",
              "| モデル | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 | 重みの数 | 学習の所要 |", "|---|---|---|---|---|---|---|---|"]
        for k, x in R.items():
            e = x["res"][tol]["ALL"]
            L.append(f"| {x['name']} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} | {x['params']} | {x['sec']} |")
        for nmr, nt in NOROWS:
            L.append(f"| {nmr} | 982 | {nt} | | | | | |")
        for k, x in P.items():
            e = x["res"][tol]["ALL"]
            L.append(f"| {x['name']} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} | {x['params'] if x['params'] else '読めず'} | {f'{x[chr(115)+chr(101)+chr(99)] / 60:.1f}分' if x['sec'] else '記録なし'} |")
        L += ["", f"## 技法ごと（許容±{tol}秒）", "", "| モデル | 技法 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|"]
        for src in (R, P):
            for k, x in src.items():
                for c in TYPES:
                    e = x["res"][tol][c]
                    if k == "rule" and c not in x["made"]:
                        L.append(f"| {x['name']} | {JP[c]} | {e['true']} | 作れず | 作れず | 作れず | 作れず |")
                    else:
                        L.append(f"| {x['name']} | {JP[c]} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
        L.append("")
    L += ["## 参考＝F値（合計・技法ごと・許容±0.5秒）", "", "| モデル | " + " | ".join(JP[c] for c in TYPES + ["ALL"]) + " |", "|---|" + "---|" * 6]
    for src in (R, P):
        for k, x in src.items():
            L.append(f"| {x['name']} | " + " | ".join(f"{x['res'][0.5][c]['f1']:.3f}" for c in TYPES + ["ALL"]) + " |")
    L += ["", "## 注記", "",
          "- 6a の5技法の合計＝作れなかった技法は出した数 0・当てた数 0 として足した（正解の数は入れた）。",
          "- 6f＝技法ごとに出力1本の網を5本学習（網の形と幅は基準と同じ）。境目と採用エポックも技法ごとの dev の F で決まる。合計は5本の予測をまとめて数えた。",
          "- 6g の参考の行＝走らせ直していない。数は各 metrics.json の試験の数をそのまま写した（5技法だけ・ビブラートは抜いた）。",
          "  正解の時刻と技法の数が違う＝v2a と v2d は 9/28 の正解（遅れなし・当時の版・試験の正解 974件）で、6技法で学習。v2d は正解を音の終わりに吸着した版で数えた値。c2_shift2 は技法ごとの遅れの正解で数えた値。",
          "  文書の値（卒論計画 v2 の D2 の行）＝モデルA F値 0.11（±1秒・6技法）・音程ガイド表示を使った例 F値 0.25（1曲）・件数の一致度 0.647。報告_G3v2（9/28）の最終値＝F ±0.5秒 0.129・±1秒 0.199・±2秒 0.271（3モデル平均）。",
          "- 重みの数＝model.pt の学習される数の和（BatchNorm の running の数は除く）。学習の所要＝metrics.json の elapsed_sec（特徴の読み込みを除く）。", "",
          "自己確認＝" + "／".join(notes)]
    open(os.path.join(OUT, "手順6_モデルの比較_c4_2026-11-05.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    for nt in notes:
        print(nt)
    for src in (R, P):
        for k, x in src.items():
            e = x["res"][0.5]["ALL"]
            print(f"{k}: {e['true']}/{e['pred']}/{e['hit']}/{e['extra']}/{e['miss']} F{e['f1']:.3f} 重み {x['params']} 所要 {x['sec']}")
    print("飛ばした＝", skipped)


if __name__ == "__main__":
    main()
