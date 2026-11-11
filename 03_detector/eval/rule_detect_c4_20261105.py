# -*- coding: utf-8 -*-
r"""手順6a＝学習なしの基準（決め事で拾う物）（2026-11-05 Opus・c4-compare 追補）。classic_detectors.py（9/28・しゃくりの決め事）の作りを写した新規1本。
入力＝feats/<曲>.npz の声の高さ（st＝半音・voiced）と音量（rms）だけ。学習しない。決め事の数値（下の格子）だけ dev（20260911・30曲）で選び、試験（20260924・28曲）で数える。
正解＝全体1つの遅れの正解（index_c3_5tech_20261105）。照合＝span の内側・技法ごとの貪欲一致（common_c4_20261105.match）。
決め事
  音の切れ目＝classic_detectors の segments と同じ（有声の連続・中央値5で平滑・0.8半音を超える段差で切る・8フレーム以上）。
  しゃくり＝classic_detectors の det_shakuri と同じ（音の出だし win フレームの最低点から到達の高さまで rise_min 半音以上上がる・最低点が前の方）。時刻＝音の出だし。
  フォール＝音の終わり win フレームで、その前の高さより drop_min 半音以上下がる（最低点が後ろの方）。時刻＝音の終わり。
  こぶし＝音の中で、声の高さから31フレームの中央値を引いた残りが depth 半音以上の山（短い持ち上げ＝頂の前後 0.15秒の内で高さの半分より下へ戻る）。時刻＝山の頂。山どうしは 0.3秒以上離す。
  アクセント＝音量（feats の rms＝既に log）の立ち上がり（k フレーム後−k フレーム前）が曲の中で z 標準偏差以上の山・山の所が有声。時刻＝山。山どうしは 0.3秒以上離す。
  ハンマリング＝作れず（声の高さと音量だけで形を決める決め事を置けない・grep 済みの記録は報告に書く）。
数値の選び方＝dev の30曲で、技法ごとに F（±0.5秒）最大の格子点（同じ F なら出した数と正解の数の差が小さい方）。
低い優先度（Idle）・スレッド 1 で回す（学習の邪魔をしない）。
使い方: python rule_detect_c4_20261105.py
出力＝out/c4_compare_20261105/手順6a_決め事_c4_2026-11-05.md ・ .csv ・ pred_rule_test.csv
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import csv, ctypes, itertools, json, sys, time
import numpy as np
from scipy.ndimage import median_filter
from common_c4_20261105 import ROOT, TYPES, JP, DEV, TEST, FR, truth_of, load_index, match, finish

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()), 0x40)
except Exception:
    pass
OUT = os.path.join(ROOT, "out", "c4_compare_20261105")
IX = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"


def segments(st, v, min_len=8, jump=0.8):
    sm = st.copy(); ok = ~np.isnan(sm)
    idx = np.where(ok)[0]
    if len(idx) < 10: return [], sm
    sm[idx] = median_filter(sm[idx], size=5, mode="nearest")
    segs = []; i = 0; n = len(sm)
    while i < n:
        if not ok[i]: i += 1; continue
        j = i
        while j + 1 < n and ok[j + 1]:
            a_ = np.nanmedian(sm[max(i, j - 4):j + 1]); b_ = np.nanmedian(sm[j + 1:j + 6])
            if abs(b_ - a_) > jump and (j - i) >= min_len: break
            j += 1
        if (j - i + 1) >= min_len: segs.append((i, j))
        i = j + 1
    return segs, sm


def det_shakuri(S, rise_min, win, gap_min):
    sm, segs = S["sm"], S["segs"]; out = []; prev_end = -10 ** 9
    for i, j in segs:
        n = j - i + 1; gap = i - prev_end; prev_end = j
        if n < win + 5 or gap < gap_min: continue
        head = sm[i:i + win]; tgt = np.nanmedian(sm[i + win:i + min(n, win + 25)])
        if tgt - np.nanmin(head) >= rise_min and np.nanargmin(head) < win * 0.7: out.append(i / FR)
    return out


def det_fall(S, drop_min, win):
    sm, segs = S["sm"], S["segs"]; out = []
    for i, j in segs:
        n = j - i + 1
        if n < win + 10: continue
        tail = sm[j - win + 1:j + 1]; body = np.nanmedian(sm[max(i, j - win - 25):j - win + 1])
        if body - np.nanmin(tail) >= drop_min and np.nanargmin(tail) > win * 0.3: out.append((j + 1) / FR)
    return out


def peaks_1d(x, thr, w):
    n = len(x); out = []
    if n < 3: return out
    pad = np.concatenate([np.full(w, -np.inf), x, np.full(w, -np.inf)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1).max(axis=1)
    return np.nonzero((x >= thr) & (x >= win))[0]


def det_kobushi(S, depth):
    """短い山だけ＝頂の前後 15 フレーム（0.15秒）の内で、残りが高さの半分より下へ戻る。"""
    r = S["res"]; n = len(r); out = []
    for i in peaks_1d(r, depth, 30):
        a, b = r[max(0, i - 15):i], r[i + 1:min(n, i + 16)]
        if len(a) and len(b) and a.min() < depth * 0.5 and b.min() < depth * 0.5:
            out.append(i / FR)
    return out


def det_accent(S, z, k):
    o = S["on"][k]
    idx = peaks_1d(o, z, 30)
    v = S["v"]
    return [i / FR for i in idx if v[min(len(v) - 1, i + k)]]


def prep(video):
    f = np.load(os.path.join(ROOT, "feats", video + ".npz"))
    st = f["st"].astype(float); v = f["voiced"] > 0; st[~v] = np.nan
    segs, sm = segments(st, v)
    res = np.full(len(sm), -np.inf)
    for i, j in segs:
        s = sm[i:j + 1]
        if len(s) < 15: continue
        s2 = np.interp(np.arange(len(s)), np.where(~np.isnan(s))[0], s[~np.isnan(s)]) if np.isnan(s).any() else s
        res[i:j + 1] = s2 - median_filter(s2, size=31, mode="nearest")
    lr = f["rms"].astype(float)   # feats の rms は既に log（extract_feats.py＝np.log(rms+1e-6)）
    on = {}
    for k in (3, 5, 10):
        d = np.full(len(lr), -np.inf); d[k:-k] = lr[2 * k:] - lr[:-2 * k]
        fin = d[np.isfinite(d)]; on[k] = (d - fin.mean()) / (fin.std() + 1e-9)
    return dict(sm=sm, segs=segs, res=res, on=on, v=v)


DET = {
    "shakuri": (det_shakuri, list(itertools.product([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], [10, 15, 20, 30], [0, 5, 10, 20]))),
    "fall": (det_fall, list(itertools.product([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0], [10, 15, 20, 30]))),
    "kobushi": (det_kobushi, [(x,) for x in (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0)]),
    "accent": (det_accent, list(itertools.product([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], [3, 5, 10]))),
}
NAMES = {"shakuri": "（上げ幅 半音, 出だしの窓 フレーム, 前の無声 フレーム）", "fall": "（下げ幅 半音, 終わりの窓 フレーム）",
         "kobushi": "（山の高さ 半音）", "accent": "（立ち上がり z, 幅 k フレーム）"}


def score(det, params, cache, truth, tol=0.5):
    h = npd = nt = 0
    for v, (span, tr) in truth.items():
        lo, hi = span; c = det["c"]
        pr = [t for t in det["f"](cache[v], *params) if lo <= t <= hi]
        h += match(pr, tr[c], tol); npd += len(pr); nt += len(tr[c])
    return h, npd, nt


def main():
    t0 = time.time()
    ix = load_index(IX)
    tr_dev, tr_test = truth_of(ix, DEV), truth_of(ix, TEST)
    cache = {v: prep(v) for v in list(tr_dev) + list(tr_test)}
    print(f"特徴を読んだ＝dev {len(tr_dev)}曲・試験 {len(tr_test)}曲・{time.time() - t0:.0f}秒", flush=True)
    best = {}
    for c, (f, grid) in DET.items():
        bb = None
        for p in grid:
            h, npd, nt = score(dict(c=c, f=f), p, cache, tr_dev)
            pr = h / npd if npd else 0; rc = h / nt if nt else 0; F = 2 * pr * rc / (pr + rc) if pr + rc else 0
            key = (F, -abs(npd - nt))
            if bb is None or key > bb[0]: bb = (key, p, (h, npd, nt))
        best[c] = bb
        print(f"{c} 選んだ数値 {bb[1]} dev F {bb[0][0]:.3f} dev 当てた/出した/正解 {bb[2]}", flush=True)
    TOLS = (0.25, 0.5, 1.0, 2.0)
    R = {}
    for tol in TOLS:
        agg = {}
        for c in TYPES:
            if c in best:
                h, npd, nt = score(dict(c=c, f=DET[c][0]), best[c][1], cache, tr_test, tol)
            else:
                h, npd, nt = 0, 0, sum(len(tr[c]) for _, tr in tr_test.values())
            agg[c] = [h, npd, nt]
        R[tol] = finish(agg)
    with open(os.path.join(OUT, "pred_rule_test_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo); w.writerow(["video", "type", "wav_time_sec"])
        for v in tr_test:
            for c in best:
                for t in DET[c][0](cache[v], *best[c][1]): w.writerow([v, c, f"{t:.2f}"])
    with open(os.path.join(OUT, "手順6a_決め事_c4_2026-11-05.csv"), "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo); w.writerow(["許容秒", "技法", "正解の数", "モデルが出した数", "当てた数", "余計に出した数", "見落とした数", "F値", "決め事の数値"])
        for tol in TOLS:
            for c in TYPES + ["ALL"]:
                e = R[tol][c]
                w.writerow([tol, JP[c], e["true"], e["pred"] if (c in best or c == "ALL") else "作れず", e["hit"], e["extra"], e["miss"], round(e["f1"], 3), str(best[c][1]) if c in best else ("" if c == "ALL" else "作れず")])
    L = ["# 手順6a＝学習なしの基準（声の高さと音量の決め事）2026-11-05", "", "**Opus 参考値・本人未検証**", "",
         "決め事の数値だけ dev（20260911・30曲）で選び（技法ごとに F ±0.5秒 最大）、試験（20260924・28曲）で数えた。正解＝全体1つの遅れ。", ""]
    for tol in (0.5, 1.0):
        L += [f"## 許容±{tol}秒", "", "| 技法 | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|"]
        for c in TYPES + ["ALL"]:
            e = R[tol][c]
            if c in best or c == "ALL":
                L.append(f"| {JP[c]} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
            else:
                L.append(f"| {JP[c]} | {e['true']} | 作れず | 作れず | 作れず | 作れず |")
        L.append("")
    L += ["5技法の合計の行＝ハンマリングは出した数 0・当てた数 0 として足した（正解の数は入れた）。", "", "選んだ数値（dev）"]
    for c in best:
        L.append(f"- {JP[c]}＝{best[c][1]} {NAMES[c]}・dev F {best[c][0][0]:.3f}・dev 当てた {best[c][2][0]}／出した {best[c][2][1]}／正解 {best[c][2][2]}")
    L += ["- ハンマリング＝作れず", "", "参考 F（試験・±0.5秒）＝" + "・".join(f"{JP[c]} {R[0.5][c]['f1']:.3f}" for c in TYPES + ["ALL"]),
          "参考 F（試験・±1.0秒）＝" + "・".join(f"{JP[c]} {R[1.0][c]['f1']:.3f}" for c in TYPES + ["ALL"]), "", f"所要 {time.time() - t0:.0f}秒"]
    open(os.path.join(OUT, "手順6a_決め事_c4_2026-11-05.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(dict(best={c: dict(params=list(best[c][1]), dev=list(best[c][2])) for c in best}, R={str(t): R[t] for t in TOLS}),
              open(os.path.join(OUT, "手順6a_決め事_c4_2026-11-05.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for tol in TOLS:
        print(f"±{tol}: " + " ".join(f"{c[:4]} {R[tol][c]['true']}/{R[tol][c]['pred']}/{R[tol][c]['hit']}" for c in TYPES + ["ALL"]))
    print(f"所要 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()
