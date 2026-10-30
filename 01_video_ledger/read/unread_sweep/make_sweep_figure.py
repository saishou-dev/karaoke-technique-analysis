#!/usr/bin/env python3
"""make_sweep_figure.py ＝ 掃引表から図1枚（横＝「読まない」率・縦＝誤読率・点＝条件）を作る（2026-10-30）

コマ単位の「読まない」率・誤読率は安定判定コマ数（後処理）に依らないため、44通りは
しきい値11点の上に4つずつ重なる。そこで点＝しきい値、点の横の文字＝安定判定 4/5/6/8 コマでの曲末一致数とした。
補足測定（数字でない見本を照合から外す）を2本目の系列として重ねる。
Windows 側の python（日本語フォントがある側）で実行する。
使い方: python make_sweep_figure.py --result <unread_sweep_20261030/result> --out <png>
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

C1 = "#2a78d6"   # 既定パレット 分類色1（現行の見本一式）
C2 = "#eb6834"   # 既定パレット 分類色2（補足：数字でない見本を外す）
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURF = "#fcfcfb"


def pick_font():
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return None


def by_t(rows):
    d = {}
    for r in rows:
        d.setdefault(float(r["ncc_min"]), {})[int(r["stable_min"])] = r
    return dict(sorted(d.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = pick_font()
    if f:
        plt.rcParams["font.family"] = f
    main_rows = by_t(csv.DictReader(open(os.path.join(a.result, "sweep_table.csv"), encoding="utf-8")))
    sp = os.path.join(a.result, "supp_digit_only.csv")
    supp_rows = by_t(csv.DictReader(open(sp, encoding="utf-8"))) if os.path.exists(sp) else {}
    sel = json.load(open(os.path.join(a.result, "selection.json"), encoding="utf-8"))
    cur = sel["current"]

    fig, ax = plt.subplots(figsize=(10.0, 6.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)

    def series(d, color, marker, label, annotate):
        xs = [float(v[6]["abstain"]) for v in d.values()]
        ys = [float(v[6]["misread"]) for v in d.values()]
        ax.plot(xs, ys, "-", color=color, lw=2, zorder=2)
        ax.scatter(xs, ys, s=52, marker=marker, color=color, edgecolor=SURF, linewidth=2, zorder=3, label=label)
        if annotate:
            for t, v in d.items():
                fin = "/".join(str(v[s]["final_hit"]) for s in (4, 5, 6, 8))
                ax.annotate(f"{t:.2f}｜曲末 {fin}", (float(v[6]["abstain"]), float(v[6]["misread"])),
                            textcoords="offset points", xytext=(-8, -13), ha="right", fontsize=8, color=INK2)

    series(main_rows, C1, "o", "現行の見本一式（しきい値だけを変える）", True)
    if supp_rows:
        series(supp_rows, C2, "s", "補足：「数字でない」見本を照合から外す", False)

    ax.axhline(float(cur["misread"]), color=INK2, lw=1, ls="--", zorder=1)
    ax.text(82.5, float(cur["misread"]) + 0.03, f"現行の誤読率 {float(cur['misread']):.2f}%", va="bottom", ha="right",
            fontsize=9, color=INK2)

    def ring(x, y):
        ax.scatter([x], [y], s=220, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=4)

    ring(float(cur["abstain"]), float(cur["misread"]))
    ax.annotate("現行＝0.75（誤読率が現行以下で読まない率が最小の点も同じ）\n"
                f"読まない {float(cur['abstain']):.2f}%・誤読 {float(cur['misread']):.2f}%",
                (float(cur["abstain"]), float(cur["misread"])), textcoords="offset points", xytext=(30, 62),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    bb = sel.get("best_b")
    if bb:
        ring(float(bb["abstain"]), float(bb["misread"]))
        ax.annotate(f"曲末24/24で読まない率が最小＝{bb['ncc_min']:.2f}・安定{bb['stable_min']}コマ\n"
                    f"読まない {float(bb['abstain']):.2f}%・誤読 {float(bb['misread']):.2f}%",
                    (float(bb["abstain"]), float(bb["misread"])), textcoords="offset points", xytext=(-125, -230),
                    fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

    ax.set_xlabel("「読まない」率（値を出さなかったコマの割合・％）", color=INK)
    ax.set_ylabel("誤読率（値を出したが違っていたコマの割合・％）", color=INK)
    ax.set_title("見本照合のしきい値 0.65〜0.85 × 安定判定 4/5/6/8 コマ（44通り・正解確定 41,760件）\n"
                 "点＝しきい値。コマ単位の2つの率は安定判定に依らない。点の横＝安定判定 4/5/6/8 コマでの曲末一致（24中）",
                 fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.set_xlim(22, 88)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, facecolor=fig.get_facecolor())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
