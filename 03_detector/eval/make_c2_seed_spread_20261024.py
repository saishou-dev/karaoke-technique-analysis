# -*- coding: utf-8 -*-
"""C2 の乱数の種（seed）によるばらつきを測る表を作る。

目的＝「お手本との音程差（refdiff）は効くか」を判定するとき、
現行の入力と＋お手本の差が「種を変えただけで出る揺れ」の中かどうかを数で示す。

入力＝out/<タグ>/metrics.json（学習済みの成果物を読むだけ・学習も推論もしない）
出力＝out/c2_compare_20261023/<--out_name>.md と .csv

使い方（例）
  python scripts/make_c2_seed_spread_20261024.py \
    --base c2a_base_8ep,c2a_base_8ep_s1,c2a_base_8ep_s2 \
    --ref  c2b_refdiff_8ep_v5_cov30,c2b_refdiff_8ep_v5_cov30_s1,c2b_refdiff_8ep_v5_cov30_s2 \
    --single c2b_refdiff_8ep_v6_cov45 \
    --out_name C2種のばらつき_2026-10-24
"""
import argparse
import csv
import json
import os
import subprocess
import sys

# Windows の既定（cp932）では「—」などが印字できず落ちるため、標準出力を UTF-8 にする。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
TOLS = ["0.5", "1.0", "2.0"]
WAZA = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
WAZA_JA = {
    "shakuri": "しゃくり",
    "kobushi": "こぶし",
    "vibrato": "ビブラート",
    "fall": "フォール",
    "accent": "アクセント",
    "hammering": "ハンマリング",
}


def load_metrics(tag):
    p = f"{ROOT}/out/{tag}/metrics.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def overall_f(m, tol):
    """吸着ラベルの時刻に対する全体F値（試験＝9/24 の28本）。"""
    return m["metrics"][f"test_tol{tol}"]["ALL"]["f1"]


def waza_f(m, tol, w):
    return m["metrics"][f"test_tol{tol}"][w]["f1"]


def counts(m, tol):
    d = m["metrics"][f"test_tol{tol}"]["ALL"]
    tp = d["hit"]
    return {
        "TP": tp,
        "FP": d["pred"] - tp,
        "FN": d["true"] - tp,
        "再現率": d["recall"],
        "適合率": d["precision"],
        "予測総数": d["pred"],
    }


def raw_agreement(tag, lag=0.30, valid_range="exclude_band"):
    """件数の一致度（件数整合）＝生の時刻に対する評価（eval_raw.py）から読む。

    学習も推論もせず、保存済みの予測ファイルを読み直すだけ。
    許容によらず1つの値なので、最初に読めた値を返す。読めなければ None（推定で埋めない）。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "eval_raw.py"), "--tag", tag,
           "--lag", str(lag), "--valid_range", valid_range]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    head = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("| 技法 "):
            head = [x.strip() for x in line.strip().strip("|").split("|")]
        if line.startswith("| ALL ") and head and "件数整合" in head:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            i = head.index("件数整合")
            if i < len(c):
                try:
                    return float(c[i])
                except ValueError:
                    return None
    return None


def stats(vals):
    return {
        "平均": sum(vals) / len(vals),
        "最小": min(vals),
        "最大": max(vals),
        "範囲": max(vals) - min(vals),
    }


def fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="現行の入力のタグ（種0,1,2 の順・カンマ区切り）")
    ap.add_argument("--ref", required=True, help="＋お手本のタグ（種0,1,2 の順・カンマ区切り）")
    ap.add_argument("--single", default="", help="種1本だけの条件のタグ（カンマ区切り・v6 など）")
    ap.add_argument("--out_name", default="C2種のばらつき")
    ap.add_argument("--out_dir", default=f"{ROOT}/out/c2_compare_20261023")
    a = ap.parse_args()

    base_tags = [t.strip() for t in a.base.split(",") if t.strip()]
    ref_tags = [t.strip() for t in a.ref.split(",") if t.strip()]
    single_tags = [t.strip() for t in a.single.split(",") if t.strip()]
    if len(base_tags) != len(ref_tags):
        raise SystemExit("--base と --ref の本数をそろえてください")

    M = {t: load_metrics(t) for t in base_tags + ref_tags + single_tags}
    AG = {t: raw_agreement(t) for t in M}

    out = []
    w = out.append
    w(f"# {a.out_name}（C2・乱数の種によるばらつき）\n")
    w("乱数の種（seed）を 0・1・2 と変えただけで、同じ入力・同じ分割・同じエポック数でも成績は動きます。")
    w("「お手本との音程差」の効果を言うには、その差が種の揺れより大きいことが要ります。")
    w("試験に使った録音は7条件すべて同じ 28 本（9/24）・吸着ラベルの時刻に対する評価です。")
    w("「範囲」＝最大−最小＝種を変えただけで動く幅。\n")

    # --- 1. 種ごとの全体F値 ---
    w("## 1. 全体F値（種ごと・吸着ラベル）\n")
    w("| 条件 | 種 | ±0.5秒 | ±1.0秒 | ±2.0秒 | 当たりTP | 空振りFP | 見落としFN | 再現率 | 適合率 | 件数の一致度 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    rows_csv = []
    for label, tags in (("現行の入力", base_tags), ("＋お手本（被覆30%以上の62曲）", ref_tags)):
        for i, t in enumerate(tags):
            c = counts(M[t], "0.5")
            w(
                f"| {label} | {M[t].get('seed', i)} | "
                + " | ".join(fmt(overall_f(M[t], tol)) for tol in TOLS)
                + f" | {c['TP']} | {c['FP']} | {c['FN']} | {fmt(c['再現率'])} | {fmt(c['適合率'])} | {fmt(AG[t])} |"
            )
            rows_csv.append([label, M[t].get("seed", i), t] + [overall_f(M[t], tol) for tol in TOLS]
                            + [c["TP"], c["FP"], c["FN"], c["再現率"], c["適合率"], AG[t]])
    for t in single_tags:
        frac = M[t].get("min_ref_frac", "")
        used = (M[t].get("n_ref_used") or {}).get("train", "")
        label = f"＋お手本（被覆{int(float(frac) * 100)}%以上の{used}曲）" if frac != "" else t
        c = counts(M[t], "0.5")
        w(
            f"| {label} | {M[t].get('seed', 0)} | "
            + " | ".join(fmt(overall_f(M[t], tol)) for tol in TOLS)
            + f" | {c['TP']} | {c['FP']} | {c['FN']} | {fmt(c['再現率'])} | {fmt(c['適合率'])} | {fmt(AG[t])} |"
        )
        rows_csv.append([label, M[t].get("seed", 0), t] + [overall_f(M[t], tol) for tol in TOLS]
                        + [c["TP"], c["FP"], c["FN"], c["再現率"], c["適合率"], AG[t]])
    w("")
    w("TP/FP/FN と再現率・適合率は許容±0.5秒のものです。")
    w("件数の一致度（件数整合）は生の時刻に対する評価の値で、0 に近いほど出す件数が正解の件数に合っています。\n")

    # --- 2. 揺れの幅と差の判定 ---
    w("## 2. 種の揺れの幅と「お手本の差」の判定\n")
    w("| 指標 | 現行 平均 | 現行 最小 | 現行 最大 | 現行 範囲 | ＋お手本 平均 | ＋お手本 最小 | ＋お手本 最大 | ＋お手本 範囲 | 平均の差（お手本−現行） | 揺れの中か |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    judge = {}
    for tol in TOLS:
        bv = [overall_f(M[t], tol) for t in base_tags]
        rv = [overall_f(M[t], tol) for t in ref_tags]
        bs, rs = stats(bv), stats(rv)
        diff = rs["平均"] - bs["平均"]
        inside = abs(diff) <= max(bs["範囲"], rs["範囲"])
        judge[f"全体F値 ±{tol}秒"] = (diff, bs["範囲"], rs["範囲"], inside)
        w(
            f"| 全体F値 ±{tol}秒 | {fmt(bs['平均'])} | {fmt(bs['最小'])} | {fmt(bs['最大'])} | {fmt(bs['範囲'])} "
            f"| {fmt(rs['平均'])} | {fmt(rs['最小'])} | {fmt(rs['最大'])} | {fmt(rs['範囲'])} "
            f"| {diff:+.3f} | {'揺れの中' if inside else '揺れを超える'} |"
        )
    bv = [AG[t] for t in base_tags]
    rv = [AG[t] for t in ref_tags]
    if all(v is not None for v in bv + rv):
        bs, rs = stats(bv), stats(rv)
        diff = rs["平均"] - bs["平均"]
        inside = abs(diff) <= max(bs["範囲"], rs["範囲"])
        judge["件数の一致度"] = (diff, bs["範囲"], rs["範囲"], inside)
        w(
            f"| 件数の一致度 | {fmt(bs['平均'])} | {fmt(bs['最小'])} | {fmt(bs['最大'])} | {fmt(bs['範囲'])} "
            f"| {fmt(rs['平均'])} | {fmt(rs['最小'])} | {fmt(rs['最大'])} | {fmt(rs['範囲'])} "
            f"| {diff:+.3f} | {'揺れの中' if inside else '揺れを超える'} |"
        )
    w("")
    w("判定の規則＝平均の差の大きさが、現行と＋お手本のどちらかの範囲（最大−最小）以下なら「揺れの中」。")
    w("「揺れの中」＝その指標では差があるとは言えません。\n")

    # --- 3. 技法別 ---
    for tol in ("0.5", "2.0"):
        w(f"## {3 if tol == '0.5' else 4}. 技法ごとのF値（許容±{tol}秒）\n")
        head = "| 条件 | 種 | " + " | ".join(WAZA_JA[x] for x in WAZA) + " | 全体 |"
        w(head)
        w("|---|---|" + "---|" * (len(WAZA) + 1))
        for label, tags in (("現行の入力", base_tags), ("＋お手本（62曲）", ref_tags)):
            for i, t in enumerate(tags):
                w(f"| {label} | {M[t].get('seed', i)} | "
                  + " | ".join(fmt(waza_f(M[t], tol, x)) for x in WAZA)
                  + f" | {fmt(overall_f(M[t], tol))} |")
        for t in single_tags:
            w(f"| ＋お手本（46曲・被覆45%） | {M[t].get('seed', 0)} | "
              + " | ".join(fmt(waza_f(M[t], tol, x)) for x in WAZA)
              + f" | {fmt(overall_f(M[t], tol))} |")
        w("")
        w(f"### 技法ごとの「現行＜＋お手本」が何本の種で成り立つか（±{tol}秒）\n")
        w("| 技法 | 種0 | 種1 | 種2 | 成り立った種の数 | 現行 範囲 | お手本 範囲 | 平均の差 | 判定 |")
        w("|---|---|---|---|---|---|---|---|---|")
        for x in WAZA + ["ALL"]:
            name = WAZA_JA.get(x, "全体")
            marks, bvs, rvs = [], [], []
            for tb, tr in zip(base_tags, ref_tags):
                b = waza_f(M[tb], tol, x) if x != "ALL" else overall_f(M[tb], tol)
                r = waza_f(M[tr], tol, x) if x != "ALL" else overall_f(M[tr], tol)
                bvs.append(b)
                rvs.append(r)
                marks.append("○" if r > b else ("＝" if r == b else "×"))
            bs, rs = stats(bvs), stats(rvs)
            diff = rs["平均"] - bs["平均"]
            n = marks.count("○")
            inside = abs(diff) <= max(bs["範囲"], rs["範囲"])
            w(f"| {name} | " + " | ".join(marks) + f" | {n}/3 | {fmt(bs['範囲'])} | {fmt(rs['範囲'])} "
              f"| {diff:+.3f} | {'揺れの中' if inside else '揺れを超える'} |")
        w("")
        w("○＝その種で＋お手本が現行より高い・×＝低い。3/3 でなければ「種を変えても成り立つ」とは言えません。\n")

    # --- 5. 採用エポックと所要時間 ---
    w("## 5. 条件のそろえかた（実測）\n")
    w("| 条件 | タグ | 種 | 入力の列数 | お手本として使った学習曲 | 被覆の足切り | 採用エポック | 学習にかかった時間 |")
    w("|---|---|---|---|---|---|---|---|")
    for t in base_tags + ref_tags + single_tags:
        m = M[t]
        if m.get("refdiff"):
            used = (m.get("n_ref_used") or m.get("n_ref") or {}).get("train", "")
        else:
            used = "—（お手本の列を入れない入力）"
        w(f"| {t} | `{t}` | {m.get('seed', '')} | {m.get('cin', '')} | {used} "
          f"| {m.get('min_ref_frac', 0)} | {(m.get('best') or ['', ''])[1]} | {m.get('elapsed_sec', '')}秒 |")
    w("")

    os.makedirs(a.out_dir, exist_ok=True)
    md = f"{a.out_dir}/{a.out_name}.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    cs = f"{a.out_dir}/{a.out_name}.csv"
    with open(cs, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["条件", "種", "タグ", "F値±0.5秒", "F値±1.0秒", "F値±2.0秒",
                     "当たりTP", "空振りFP", "見落としFN", "再現率", "適合率", "件数の一致度"])
        wr.writerows(rows_csv)
    print("\n".join(out))
    print("DONE", md, cs)


if __name__ == "__main__":
    main()
