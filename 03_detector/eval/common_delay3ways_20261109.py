# -*- coding: utf-8 -*-
r"""delay-3ways の共通の部品（2026-11-09 Opus）。数え方は 11/5 の手順8（rescore_step8_c4_20261105.py）の写し＝同じ答えになる。
  count_thr＝確率の山（common_c4.peaks）をしきい値で絞って技法ごとに貪欲一致（common_c4.match）。
  choose_ii＝8a の境目（dev の合計で出した数が正解の数に最も近い・同じなら F ±0.5秒 が高い方・なお同じなら低い境目・0.01〜0.99）。
  topn＝8b（確率の高い順に正解の数と同じ件数だけ出した時の当てた数・同じ確率は（曲名・時刻）の順）。
"""
import json, os
import numpy as np
from common_c4_20261105 import ROOT, TYPES, JP, DEV, TEST, FR, truth_of, load_index, peaks, match, finish, row

SHIFT = 1.6722
IXG = "out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json"
OUT = os.path.join(ROOT, "out", "delay_3ways_20261109")
FINE = [round(x, 2) for x in np.arange(0.01, 0.995, 0.01)]


def pk_of(z, vids=None):
    return {v: [peaks(z[v][k]) for k in range(len(TYPES))] for v in (vids or z.files)}


def count_thr(pk, truth, thr, tol, only=None):
    agg = {}
    for c in (only or TYPES):
        k = TYPES.index(c); h = npd = nt = 0
        for v, (span, tr) in truth.items():
            lo, hi = span
            pr = [t for t, p in pk[v][k] if p >= thr[c] and lo <= t <= hi]
            h += match(pr, tr[c], tol); npd += len(pr); nt += len(tr[c])
        agg[c] = [h, npd, nt]
    return finish(agg) if only is None else {c: row(*agg[c]) for c in agg}


def choose_ii(pk, truth):
    thr = {}
    for c in TYPES:
        best = None
        for th in FINE:
            e = count_thr(pk, truth, {c: th}, 0.5, only=[c])[c]
            key = (abs(e["pred"] - e["true"]), -e["f1"], th)
            if best is None or key < best[0]:
                best = (key, th)
        thr[c] = best[1]
    return thr


def topn(pk, truth, tol):
    agg = {}
    for c in TYPES:
        k = TYPES.index(c)
        cand = []
        for v, (span, tr) in truth.items():
            lo, hi = span
            cand += [(-p, v, t) for t, p in pk[v][k] if lo <= t <= hi]
        nt = sum(len(tr[c]) for _, tr in truth.values())
        cand.sort()
        sel = {}
        for _, v, t in cand[:nt]:
            sel.setdefault(v, []).append(t)
        h = sum(match(sel.get(v, []), tr[c], tol) for v, (span, tr) in truth.items())
        agg[c] = [h, min(nt, len(cand)), nt]
    return finish(agg)


def qstat(xs):
    if len(xs) == 0:
        return dict(n=0, median=None, q1=None, q3=None)
    a = np.asarray(xs, float)
    return dict(n=int(len(a)), median=round(float(np.median(a)), 4), q1=round(float(np.percentile(a, 25)), 4), q3=round(float(np.percentile(a, 75)), 4))
