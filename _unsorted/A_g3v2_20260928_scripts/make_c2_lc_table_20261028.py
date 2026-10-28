# -*- coding: utf-8 -*-
r"""C2（検出器）の学習曲線を1表にする（2026-10-28）。学習も推論もしない＝記録を読み直すだけ。

読むもの。
  out/<タグ>/metrics.json ＝試験（9/24 の28曲・未見）と dev の成績・採用エポック・所要秒
  out/<タグ>/persong_test.csv ＝曲ごとの正解件数と予測件数（件数の一致度を曲単位で出すのに使う）
  out/c2_learning_curve_20261028/学習量の内訳_2026-10-28.csv ＝学習曲数と学習イベント件数（技法別）

条件。
  100% ＝ c2a_base_8ep（10/23・8エポック・seed 0・現行の入力）＝基準。再実行しない。
  75%/50%/25% ＝ c2_lc75_8ep / c2_lc50_8ep / c2_lc25_8ep（index を差し替えただけ・他は同条件）。

判定（指示④の規則）。
  75%→100% の ±0.5秒 F値の伸びが 0.01 未満＝頭打ち（量は足りている）
  0.02 以上＝まだ伸びる（量が効く）／その間＝判定保留。

使い方:
  python make_c2_lc_table_20261028.py
出力: out/c2_learning_curve_20261028/C2学習曲線_2026-10-28.csv ＋ .md
"""
import csv
import json
import os
import sys

if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
LCDIR = os.path.join(OUT, "c2_learning_curve_20261028")
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート",
      "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング"}
CONDS = [("25%", "c2_lc25_8ep"), ("50%", "c2_lc50_8ep"), ("75%", "c2_lc75_8ep"), ("100%", "c2a_base_8ep")]


def load_volume():
    """学習量の内訳（学習曲数・イベント件数・技法別）を条件名で引ける形にする。"""
    path = os.path.join(LCDIR, "学習量の内訳_2026-10-28.csv")
    vol = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            vol[r["条件"]] = r
    return vol


def count_agree(tag):
    """件数の一致度（曲単位）＝1 − Σ|予測−正解| ÷ Σ正解。試験の28曲について。"""
    path = os.path.join(OUT, tag, "persong_test.csv")
    if not os.path.exists(path):
        return None
    d = t = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d += abs(int(r["n_pred"]) - int(r["n_true"]))
            t += int(r["n_true"])
    return 1 - d / max(1, t)


def main():
    vol = load_volume()
    rows = []
    for name, tag in CONDS:
        mpath = os.path.join(OUT, tag, "metrics.json")
        if not os.path.exists(mpath):
            print(f"未完了のため飛ばす＝{name}（{tag}）")
            continue
        m = json.load(open(mpath, encoding="utf-8"))
        v = vol.get(name, {})
        rows.append(dict(
            cond=name, tag=tag,
            n_song=int(v.get("学習曲数", m.get("n_train", 0))),
            n_ev=int(v.get("学習イベント件数", 0)),
            per_ev={c: int(v.get(c, 0)) for c in TYPES},
            f05=m["metrics"]["test_tol0.5"]["ALL"]["f1"],
            f10=m["metrics"]["test_tol1.0"]["ALL"]["f1"],
            f20=m["metrics"]["test_tol2.0"]["ALL"]["f1"],
            tp=m["metrics"]["test_tol0.5"]["ALL"]["hit"],
            pred=m["metrics"]["test_tol0.5"]["ALL"]["pred"],
            true=m["metrics"]["test_tol0.5"]["ALL"]["true"],
            rec=m["metrics"]["test_tol0.5"]["ALL"]["recall"],
            prec=m["metrics"]["test_tol0.5"]["ALL"]["precision"],
            per_f={c: m["metrics"]["test_tol0.5"][c]["f1"] for c in TYPES},
            dev_f=m["best"][0], best_ep=m["best"][1],
            agree=count_agree(tag), sec=m.get("elapsed_sec"),
        ))

    # 判定（75%→100% の ±0.5秒の伸び）
    g = {r["cond"]: r for r in rows}
    verdict = "判定できない（75% か 100% が欠けている）"
    delta = None
    if "75%" in g and "100%" in g:
        delta = g["100%"]["f05"] - g["75%"]["f05"]
        if delta < 0.01:
            verdict = f"頭打ち＝量は足りている（75%→100% の ±0.5秒 F値の伸び {delta:+.3f} ＜ 0.01）"
        elif delta >= 0.02:
            verdict = f"まだ伸びる＝量が効く（75%→100% の ±0.5秒 F値の伸び {delta:+.3f} ≧ 0.02）"
        else:
            verdict = f"判定保留（75%→100% の ±0.5秒 F値の伸び {delta:+.3f} ＝0.01以上0.02未満）"

    os.makedirs(LCDIR, exist_ok=True)
    cpath = os.path.join(LCDIR, "C2学習曲線_2026-10-28.csv")
    with open(cpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["条件", "学習曲数", "学習イベント件数", "F値±0.5秒", "F値±1.0秒", "F値±2.0秒",
                    "正解", "予測", "当たり", "再現率", "適合率", "件数の一致度",
                    "dev F値", "採用エポック", "学習の所要秒", "タグ"]
                   + [JP[c] + "のF値" for c in TYPES] + [JP[c] + "の学習件数" for c in TYPES])
        for r in rows:
            w.writerow([r["cond"], r["n_song"], r["n_ev"], r["f05"], r["f10"], r["f20"],
                        r["true"], r["pred"], r["tp"], r["rec"], r["prec"],
                        "" if r["agree"] is None else round(r["agree"], 3),
                        r["dev_f"], r["best_ep"], r["sec"], r["tag"]]
                       + [r["per_f"][c] for c in TYPES] + [r["per_ev"][c] for c in TYPES])

    L = [f"# C2 学習曲線（検出器・学習データ量だけを変える）2026-10-28", "",
         f"**判定＝{verdict}**", "",
         "条件＝8エポック・現行の入力（お手本の音程差なし）・乱数の種 0・スレッド 12・"
         "しきい値は dev（20260911・30曲）で決定・試験＝20260924 の28曲（未見・学習に混ぜない）。",
         "学習曲の部分集合は曲名でそろえてから random.Random(0) で1回混ぜ先頭から取る＝25% ⊂ 50% ⊂ 75% ⊂ 100% の入れ子。",
         "100% は 10/23 の c2a_base_8ep をそのまま基準に使った（再実行していない）。", "",
         "## 試験（9/24 の28曲・未見）の成績",
         "| 条件 | 学習曲数 | 学習イベント件数 | F値±0.5秒 | F値±1.0秒 | F値±2.0秒 | 当たり | 予測 | 再現率 | 適合率 | 件数の一致度 | 採用エポック | 学習の所要秒 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['cond']} | {r['n_song']} | {r['n_ev']} | {r['f05']:.3f} | {r['f10']:.3f} | {r['f20']:.3f} | "
                 f"{r['tp']} | {r['pred']} | {r['rec']} | {r['prec']} | "
                 f"{'—' if r['agree'] is None else format(r['agree'], '.3f')} | {r['best_ep']} | {r['sec']} |")
    L += ["", "## 1本あたり（25%→50%→75%→100%）の伸び（±0.5秒）"]
    for i in range(1, len(rows)):
        d = rows[i]["f05"] - rows[i - 1]["f05"]
        dn = rows[i]["n_song"] - rows[i - 1]["n_song"]
        L.append(f"- {rows[i - 1]['cond']} → {rows[i]['cond']}（学習曲 +{dn}本）＝F値 {d:+.3f}")
    L += ["", "## 技法ごとの F値（許容±0.5秒・試験）",
          "| 条件 | " + " | ".join(JP[c] for c in TYPES) + " | 全体 |",
          "|---|" + "---|" * (len(TYPES) + 1)]
    for r in rows:
        L.append(f"| {r['cond']} | " + " | ".join(f"{r['per_f'][c]:.3f}" for c in TYPES) + f" | {r['f05']:.3f} |")
    L += ["", "## 技法ごとの学習イベント件数",
          "| 条件 | " + " | ".join(JP[c] for c in TYPES) + " | 合計 |",
          "|---|" + "---|" * (len(TYPES) + 1)]
    for r in rows:
        L.append(f"| {r['cond']} | " + " | ".join(str(r["per_ev"][c]) for c in TYPES) + f" | {r['n_ev']} |")
    mpath = os.path.join(LCDIR, "C2学習曲線_2026-10-28.md")
    open(mpath, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE", cpath, mpath)


if __name__ == "__main__":
    main()
