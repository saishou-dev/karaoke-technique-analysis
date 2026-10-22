# -*- coding: utf-8 -*-
"""C2 比較表（入力を変える軸）を CSV と Markdown にまとめる（2026-10-23）。

何をするか。
学習の結果ファイル（out/<tag>/metrics.json）を条件ごとに読み、
「見つけ方の正しさ（F値）」「当たり（TP）」「空振り（FP）」「見落とし（FN）」を技法ごとと全体で並べます。
さらに、同じ予測を観測した生の時刻に対して評価する台本（eval_raw.py）も条件ごとに走らせ、全体の行だけを取り込みます。

使い方:
  python make_c2_table_20261023.py --rows "現行の入力=c2a_base_4ep,現行＋お手本との音程差=c2b_refdiff_4ep" --out_name C2比較表_v1_2026-10-23
"""
import os
import sys
import csv
import json
import argparse
import subprocess

ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"
HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール",
      "accent": "アクセント", "hammering": "ハンマリング", "ALL": "全体"}


def raw_all(tag, lag, valid_range):
    """eval_raw.py を走らせ、許容ごとの全体行（再現率・適合率・F値・件数整合）を取り出す。"""
    cmd = [sys.executable, os.path.join(HERE, "eval_raw.py"), "--tag", tag, "--lag", str(lag), "--valid_range", valid_range]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    got = {}
    tol = None
    head = None
    # eval_raw.py の表の列は版によって増えている（2026-10-20 版は TP/FP/FN の列が付く）。
    # そこで見出し行を読んで列の位置を決める。見出しが読めないときは左から数える。
    for line in (r.stdout or "").splitlines():
        if line.startswith("## 許容±"):
            tol = line.split("±")[1].replace("s", "").strip()
        if line.startswith("| 技法 "):
            head = [x.strip() for x in line.strip().strip("|").split("|")]
        if line.startswith("| ALL ") and tol:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            def pick(name, fallback):
                if head and name in head:
                    i = head.index(name)
                    if i < len(c):
                        return c[i]
                return c[fallback] if fallback < len(c) else ""
            got[tol] = dict(true=pick("正解", 1), pred=pick("予測", 2), hit=pick("一致", 3),
                            recall=pick("再現率", 4), precision=pick("適合率", 5),
                            f1=pick("F1", 6), count_agree=pick("件数整合", 7))
    if not got:
        got["error"] = (r.stderr or "")[-300:]
    return got


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="『表示名=タグ』をカンマ区切りで（上が主たる条件）")
    ap.add_argument("--out_name", default="C2比較表_v1")
    ap.add_argument("--out_dir", default=f"{ROOT}/out/c2_compare_20261023")
    ap.add_argument("--lag", type=float, default=0.30)
    ap.add_argument("--valid_range", default="exclude_band")
    ap.add_argument("--no_raw", action="store_true", help="eval_raw.py を走らせない")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for item in a.rows.split(","):
        label, tag = item.split("=", 1)
        p = f"{ROOT}/out/{tag}/metrics.json"
        if not os.path.exists(p):
            print(f"!! まだ結果が無い: {p}", flush=True)
            continue
        m = json.load(open(p, encoding="utf-8"))
        rows.append((label, tag, m, ({} if a.no_raw else raw_all(tag, a.lag, a.valid_range))))

    # ---- CSV（1行＝条件×許容×技法）----
    csv_path = f"{a.out_dir}/{a.out_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["条件", "タグ", "評価", "許容秒", "技法", "正解数", "予測数", "当たりTP", "空振りFP", "見落としFN", "再現率", "適合率", "F値",
                    "入力の列数", "学習曲数", "お手本ありの学習曲数", "エポック", "乱数の種"])
        for label, tag, m, rw in rows:
            meta = [m.get("cin", ""), m.get("n_train", ""), (m.get("n_ref") or {}).get("train", ""), m.get("epochs", ""), m.get("seed", "")]
            for tol in ("0.5", "1.0", "2.0"):
                d = m["metrics"].get(f"test_tol{tol}")
                if not d:
                    continue
                for c in TYPES + ["ALL"]:
                    x = d[c]
                    w.writerow([label, tag, "吸着ラベル時刻", tol, JP[c], x["true"], x["pred"], x["hit"], x["pred"] - x["hit"], x["true"] - x["hit"],
                                x["recall"], x["precision"], x["f1"]] + meta)
            for tol, x in rw.items():
                if tol == "error":
                    continue
                w.writerow([label, tag, f"観測した生の時刻（{a.valid_range}・lag+{a.lag}s）", tol, "全体", x["true"], x["pred"], x["hit"],
                            int(x["pred"]) - int(x["hit"]), int(x["true"]) - int(x["hit"]), x["recall"], x["precision"], x["f1"]] + meta)

    # ---- Markdown ----
    md = [f"# C2 比較表 v1（入力を変える軸・2026-10-23）", "",
          "比べているもの＝歌声から加点技法を見つける機械（検出器）に入れる材料の違いです。",
          "行が条件・列が成績です。試験に使った録音は同じ 28 本（9/24 のセッション）で、評価のしかたも同じです。",
          "F値＝見つけたものの正しさ（適合率）と、見落としの少なさ（再現率）を合わせた指標で、1 に近いほど良い値です。",
          "TP＝当たり・FP＝空振り（鳴っていないのに出した）・FN＝見落とし（あったのに出さなかった）。", ""]
    md.append("## 全体の成績（試験＝9/24 の 28 本・吸着ラベルの時刻に対して）")
    md.append("| 条件 | 許容 | F値 | 当たりTP | 空振りFP | 見落としFN | 再現率 | 適合率 | 入力の列数 | 学習曲数 | お手本ありの学習曲数 | エポック |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, tag, m, rw in rows:
        for tol in ("0.5", "1.0", "2.0"):
            d = m["metrics"].get(f"test_tol{tol}")
            if not d:
                continue
            x = d["ALL"]
            md.append(f"| {label} | ±{tol}秒 | {x['f1']} | {x['hit']} | {x['pred'] - x['hit']} | {x['true'] - x['hit']} | {x['recall']} | {x['precision']} | "
                      f"{m.get('cin','')} | {m.get('n_train','')} | {(m.get('n_ref') or {}).get('train','')} | {m.get('epochs','')} |")
    md.append("")
    md.append("## 技法ごとの F値（許容±0.5秒・試験＝9/24 の 28 本）")
    md.append("| 条件 | " + " | ".join(JP[c] for c in TYPES) + " | 全体 |")
    md.append("|---" * (len(TYPES) + 2) + "|")
    for label, tag, m, rw in rows:
        d = m["metrics"].get("test_tol0.5")
        if not d:
            continue
        md.append(f"| {label} | " + " | ".join(str(d[c]["f1"]) for c in TYPES) + f" | {d['ALL']['f1']} |")
    md.append("")
    if not a.no_raw:
        md.append(f"## 全体の成績（観測した生の時刻に対して・{a.valid_range}・遅れ +{a.lag}秒）")
        md.append("| 条件 | 許容 | 正解 | 予測 | 当たり | 再現率 | 適合率 | F値 | 件数整合 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for label, tag, m, rw in rows:
            for tol in ("0.5", "1.0", "2.0"):
                x = rw.get(tol)
                if not x:
                    continue
                md.append(f"| {label} | ±{tol}秒 | {x['true']} | {x['pred']} | {x['hit']} | {x['recall']} | {x['precision']} | {x['f1']} | {x['count_agree']} |")
        md.append("")
    md.append("## 条件のそろえかた")
    for label, tag, m, rw in rows:
        md.append(f"- {label}（タグ {tag}）＝入力の列数 {m.get('cin','')}・学習 {m.get('n_train','')} 本・しきい値決定 {m.get('n_dev','')} 本・試験 {m.get('n_test','')} 本・"
                  f"エポック {m.get('epochs','')}・乱数の種 {m.get('seed','')}・バッチ {m.get('bs','')}・お手本ありの本数 {m.get('n_ref','')}・学習にかかった時間 {m.get('elapsed_sec','')}秒")
    md_path = f"{a.out_dir}/{a.out_name}.md"
    open(md_path, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("DONE", csv_path, md_path, flush=True)
