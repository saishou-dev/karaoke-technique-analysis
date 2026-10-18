#!/usr/bin/env python3
"""make_figure.py ＝ 比較表から図1枚を作る（2026-10-18）

何を見せる図か
  左＝「値を出したコマのうち正しかった割合」（全体／正解が2桁のコマだけ）＝読み取りの正しさ。
  右＝「値を出したコマの割合（回答率）」＝どれだけ読めたか。
  この2つがそろって初めて使える手法になる（正しいだけ・読めるだけでは足りない）。
  各手法の右端に、軌跡の確定まで通したときの曲末一致（24カウンタ中）を添える。

日本語の見出しを入れるため、日本語フォントがある側（Windows の python3.12）で実行する。
使い方: python make_figure.py --result <result dir> --out <png>
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SHORT = {
    "ncc": "現行 自作の見本照合（NCC・v0.8e）",
    "nn_l2": "対照 生ピクセル最近傍（正規化なし）",
    "tess_psm7": "Tesseract psm7（前処理あり）",
    "tess_psm8": "Tesseract psm8（前処理あり）",
    "tess_psm8_raw": "Tesseract psm8（前処理なし）",
    "easyocr_rec": "EasyOCR 認識のみ（前処理なし）",
    "easyocr_rec_prep": "EasyOCR 認識のみ（前処理あり）",
    "easyocr_readtext": "EasyOCR 検出＋認識",
    "paddle_rec": "PaddleOCR 認識のみ",
    "paddle_full": "PaddleOCR 検出＋認識（部分集合）",
}


def pick_font():
    for name in ("Meiryo", "Yu Gothic", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = pick_font()
    if f:
        plt.rcParams["font.family"] = f
    plt.rcParams["axes.unicode_minus"] = False

    rows = list(csv.DictReader(open(os.path.join(a.result, "compare_table.csv"),
                                    encoding="utf-8", newline="")))
    rows = [r for r in rows if r["cand"] in SHORT]
    rows.sort(key=lambda r: float(r["acc_answered"]))
    names = [SHORT[r["cand"]] for r in rows]
    y = list(range(len(rows)))
    h = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    b1 = ax.barh([i + h / 2 for i in y], [float(r["acc_answered"]) for r in rows], h,
                 label="全体", color="#1f4e79")
    b2 = ax.barh([i - h / 2 for i in y], [float(r.get("acc_2digit_answered", 0)) for r in rows], h,
                 label="正解が2桁のコマだけ", color="#c00000")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("値を出したコマのうち正しかった割合（％）")
    ax.set_xlim(0, 126)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=1.0)
    ax.set_title("① 読み取りの正しさ", fontsize=11)
    for bars in (b1, b2):
        for bb in bars:
            w = bb.get_width()
            ax.text(w + 1.0, bb.get_y() + bb.get_height() / 2, f"{w:.1f}",
                    va="center", fontsize=8)

    ax2 = axes[1]
    b3 = ax2.barh(y, [100.0 - float(r["abstain"]) for r in rows], h * 1.6, color="#7f7f7f")
    ax2.set_xlabel("値を出したコマの割合（％）")
    ax2.set_xlim(0, 160)
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_title("② どれだけ読めたか／1,000コマあたり秒", fontsize=11)
    for bb, r in zip(b3, rows):
        w = bb.get_width()
        sec = r["sec_per_1000_cells"]
        ax2.text(w + 1.0, bb.get_y() + bb.get_height() / 2,
                 f"{w:.1f}　{float(sec):.1f} 秒/千コマ" if sec else f"{w:.1f}",
                 va="center", fontsize=8)

    fig.suptitle("録画の加点回数表示（技法別の加点回数）の読み取り：現行の見本照合と既製OCRの比較\n"
                 "2026-10-14 収録 LIVE DAM WAO! の4曲・同一の43,170件（正解が確定している41,760件で集計）",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(a.out, dpi=160)
    print("wrote", a.out, "font=", f)


if __name__ == "__main__":
    main()
