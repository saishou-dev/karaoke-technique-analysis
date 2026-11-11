# -*- coding: utf-8 -*-
r"""手順10＝アクセントの行を並べる（2026-11-06 Opus・c4-compare 追補7・学習しない）。
行＝アクセントだけ 8エポック（c4_only_accent_8ep_s0）・アクセントだけ 16エポック（c4_only_accent_16ep_s0）・まとめて1本 8エポック（基準 c3_5tech_8ep_s0）。
数え方＝今の境目（各 metrics.json の thr）・8a（dev で出す数を正解の数に合わせる境目・0.01 刻み）・8b（境目に依らない＝確率の高い順に正解の数だけ）。
確率の列＝out/c4_compare_20261105/probs/（base・only_accent・only_accent16）。正解＝全体1つの遅れ。試験＝20260924 の28曲・許容±0.5秒。
自己確認＝確率の列から今の境目で数えた 正解・出した・当てた が metrics.json と一致。打ち切りの本は partial_c4 へ振り替え（追補6）。
使い方: python eval_step10_c4_20261105.py
出力＝out/c4_compare_20261105/手順10_アクセントのエポック_c4_2026-11-06.md ・ .csv
"""
import csv, json, os, re, sys
import numpy as np
from common_c4_20261105 import ROOT, TYPES, DEV, TEST, truth_of, load_index, peaks, match, resolve_mdir

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
OUT = os.path.join(ROOT, "out", "c4_compare_20261105")
PR = os.path.join(OUT, "probs")
IXG = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
C = "accent"
ROWS = [
    ("アクセントだけ 8エポック", os.path.join(OUT, "c4_only_accent_8ep_s0"), "only_accent", 0, os.path.join(OUT, "c4_only_accent_8ep_s0.log")),
    ("アクセントだけ 16エポック", os.path.join(OUT, "c4_only_accent_16ep_s0"), "only_accent16", 0, os.path.join(OUT, "c4_only_accent_16ep_s0.log")),
    ("まとめて1本 8エポック（基準）", os.path.join(ROOT, "out", "c3_5tech_global_20261105", "c3_5tech_8ep_s0"), "base", TYPES.index(C),
     os.path.join(ROOT, "out", "c3_5tech_global_20261105", "c3_5tech_8ep_s0.log")),
]
FINE = [round(x, 2) for x in np.arange(0.01, 0.995, 0.01)]


def curve(log):
    out = []
    for ln in open(log, encoding="utf-8"):
        m = re.match(r"ep(\d+) loss=\S+ dev F1\(±0.5s\)=([\d.]+) .*?acce=([\d.]+)", ln.strip())
        if m:
            out.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def main():
    ix = load_index(IXG); td, tt = truth_of(ix, DEV), truth_of(ix, TEST)
    L = ["# 手順10＝アクセントの行（エポックの比べ）2026-11-06", "", "**Opus 参考値・本人未検証**", "",
         "試験＝20260924 の28曲・許容±0.5秒・正解＝全体1つの遅れ・アクセントの正解の数 381。", "",
         "| モデル | 数え方 | 境目 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|---|---|"]
    rows, notes, curves = [], [], []
    for name, mdir, pn, k, log in ROWS:
        md, pnote = resolve_mdir(mdir)
        if md is None:
            L.append(f"| {name} | {pnote or 'まだ終わっていない'} | | | | | | |"); continue
        if pnote:
            name = f"{name}［{pnote}］"
        mj = json.load(open(os.path.join(md, "metrics.json"), encoding="utf-8"))
        pk = {sp: (lambda z: {v: peaks(z[v][k]) for v in z.files})(np.load(os.path.join(PR, f"probs_{pn}_{sp}.npz"))) for sp in ("dev", "test")}

        def cnt(sp, tr, th):
            h = n = t = 0
            for v, (span, T) in tr.items():
                lo, hi = span; pr = [x for x, p in pk[sp][v] if p >= th and lo <= x <= hi]
                h += match(pr, T[C], 0.5); n += len(pr); t += len(T[C])
            return h, n, t
        th0 = mj["thr"][C]; h, n, t = cnt("test", tt, th0); mm = mj["metrics"]["test_tol0.5"][C]
        if (t, n, h) != (mm["true"], mm["pred"], mm["hit"]):
            raise SystemExit(f"止める＝自己確認の不一致 {name} {(t, n, h)} ／ {mm}")
        notes.append(f"{name}＝metrics.json と一致")
        best = None
        for th in FINE:
            hd, nd, tdv = cnt("dev", td, th); p = hd / nd if nd else 0; r = hd / tdv if tdv else 0; f = 2 * p * r / (p + r) if p + r else 0
            key = (abs(nd - tdv), -f, th)
            if best is None or key < best[0]: best = (key, th)
        tha = best[1]; ha, na, ta = cnt("test", tt, tha)
        cand = sorted((-p, v, x) for v, (span, T) in tt.items() for x, p in pk["test"][v] if span[0] <= x <= span[1])
        sel = {}
        for _, v, x in cand[:t]: sel.setdefault(v, []).append(x)
        hb = sum(match(sel.get(v, []), T[C], 0.5) for v, (span, T) in tt.items()); nb = min(t, len(cand))
        for way, th, (hh, nn, tt_) in (("今の境目", f"{th0:.3f}", (h, n, t)), ("8a", f"{tha:.2f}", (ha, na, ta)), ("8b（境目に依らない）", "", (hb, nb, t))):
            L.append(f"| {name} | {way} | {th} | {tt_} | {nn} | {hh} | {nn - hh} | {tt_ - hh} |")
            rows.append([name, way, th, tt_, nn, hh, nn - hh, tt_ - hh])
        cv = curve(log)
        curves.append(f"- {name}＝採用エポック {mj['best'][1]}・dev F（5技法 or 1技法の全体）＝" + "・".join(f"{f:.3f}" for _, f, _ in cv)
                      + "／dev のアクセントの F＝" + "・".join(f"{a:.2f}" for _, _, a in cv))
    L += ["", "8b 注記＝試験の曲の正解の数を使う物差し。境目を選ばない比べ用・報告書の主の表には使わない。", "",
          "エポックごとの dev F（学習台本のログ・±0.5秒）"] + curves + [
          "", "手順10b＝基準 CRNN は 6エポック目が最大（dev 0.239・c3_5tech_8ep_s0.log）＝回さず。", "", "自己確認＝" + "／".join(notes)]
    open(os.path.join(OUT, "手順10_アクセントのエポック_c4_2026-11-06.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "手順10_アクセントのエポック_c4_2026-11-06.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["モデル", "数え方", "境目", "正解の数", "モデルが出した数", "当てた数", "余計に出した数", "見落とした数"]); w.writerows(rows)
    print("\n".join(L))


if __name__ == "__main__":
    main()
