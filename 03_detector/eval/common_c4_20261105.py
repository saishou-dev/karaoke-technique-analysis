# -*- coding: utf-8 -*-
r"""c4-compare の共通の部品（2026-11-05 Opus）。照合の作法は train_crnn_refdiff_c3_5tech_20261105.py と同じ。
  pick＝確率の列から山を拾う（しきい値以上・前後 min_dist 秒の中で最大）。
  match＝技法ごとの貪欲一致（予測を時刻順に・許容の内の未使用の正解で最も近い物・同じ距離なら一覧で先の物）。
  ここでは山の候補を先に1回だけ数えて（しきい値 0）、しきい値ごとに確率で絞る＝pick と同じ結果（山の条件はしきい値に依らない）。
"""
import bisect, csv, json, os, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TYPES = ["shakuri", "kobushi", "fall", "accent", "hammering"]
JP = {"shakuri": "しゃくり", "kobushi": "こぶし", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング", "ALL": "5技法の合計"}
FR = 100
DEV, TEST = "20260911", "20260924"


def peaks(prob, min_dist=0.3):
    """pick(prob, 0, min_dist) と同じ山の一覧（時刻, 確率）。"""
    w = int(min_dist * FR); n = len(prob)
    if n < 3:
        return []
    pad = np.concatenate([np.full(w, -np.inf, np.float32), prob.astype(np.float32), np.full(w, -np.inf, np.float32)])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1).max(axis=1)   # 長さ n・[i-w, i+w] の最大（端は切り詰め）
    idx = np.nonzero(prob[1:n - 1] >= win[1:n - 1])[0] + 1
    return [(i / FR, float(prob[i])) for i in idx]


def match(pred, truth, tol):
    """元の match と同じ答え（正解の並びは元のまま・最も近い物・同じ距離なら先の物）。"""
    order = sorted(range(len(truth)), key=lambda k: truth[k]); ts = [truth[k] for k in order]
    used = set(); hit = 0
    for p in sorted(pred):
        lo = bisect.bisect_left(ts, p - tol - 1e-9); hi = bisect.bisect_right(ts, p + tol + 1e-9)
        best = None
        for j in range(lo, hi):
            k = order[j]; t = truth[k]
            if k in used or abs(t - p) > tol:
                continue
            if best is None or abs(t - p) < abs(truth[best] - p) or (abs(t - p) == abs(truth[best] - p) and k < best):
                best = k
        if best is not None:
            used.add(best); hit += 1
    return hit


def load_index(rel):
    return json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))


def truth_of(index, session):
    """{video: {技法: [span 内の正解の時刻]}}（span も返す）。"""
    out = {}
    for d in index:
        if d["session"] != session or not os.path.exists(os.path.join(ROOT, "feats", d["video"] + ".npz")):
            continue
        lo, hi = d["span"]
        out[d["video"]] = (d["span"], {c: [t for cc, t in d["events"] if cc == c and lo <= t <= hi] for c in TYPES})
    return out


def peaks_of(npz_path):
    z = np.load(npz_path)
    return {v: [peaks(z[v][k]) for k in range(len(TYPES))] for v in z.files}


def count(pk, truth, thr, tol):
    """pk＝peaks_of の結果・truth＝truth_of の結果・thr＝{技法: しきい値}。技法ごとと合計の 正解・出した・当てた・余計・見落とし・F。"""
    agg = {c: [0, 0, 0] for c in TYPES}
    for v, (span, tr) in truth.items():
        lo, hi = span
        for k, c in enumerate(TYPES):
            pr = [t for t, p in pk[v][k] if p >= thr[c] and lo <= t <= hi]
            agg[c][0] += match(pr, tr[c], tol); agg[c][1] += len(pr); agg[c][2] += len(tr[c])
    return finish(agg)


def finish(agg):
    out = {}
    for c in list(agg):
        out[c] = row(*agg[c])
    H, NP, NT = (sum(x[i] for x in agg.values()) for i in range(3))
    out["ALL"] = row(H, NP, NT)
    return out


def row(h, npd, nt):
    p = h / npd if npd else 0.0; r = h / nt if nt else 0.0
    return dict(true=nt, pred=npd, hit=h, extra=npd - h, miss=nt - h, precision=p, recall=r, f1=(2 * p * r / (p + r) if p + r else 0.0))


def load_pred_csv(path):
    pred = collections.defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        pred[(r["video"], r["type"])].append(float(r["wav_time_sec"]))
    return pred


def count_pred(pred, truth, tol):
    """保存済みの pred_test.csv（しきい値で絞り済み）を数える。"""
    agg = {c: [0, 0, 0] for c in TYPES}
    for v, (span, tr) in truth.items():
        lo, hi = span
        for c in TYPES:
            pr = [t for t in pred.get((v, c), []) if lo <= t <= hi]
            agg[c][0] += match(pr, tr[c], tol); agg[c][1] += len(pr); agg[c][2] += len(tr[c])
    return finish(agg)


def resolve_mdir(mdir):
    """追補6（11/6）＝打ち切られた本の置き場へ振り替える。
    返す物＝(使う所, 注記)。metrics.json が元の所にあれば (mdir, "")。
    無くて partial_c4/<tag>/metrics.json があれば (その所, "打ち切り（N エポック目まで）・打ち切り時点の最良エポック M")。
    どちらも無く partial_c4/<tag>/ だけある（重みも無い）か、連結ジョブのログに打ち切りの行があれば (None, "打ち切り・数なし（重みの保存なし）")。それも無ければ (None, "")＝まだ終わっていない。"""
    if os.path.exists(os.path.join(mdir, "metrics.json")):
        return mdir, ""
    tag = os.path.basename(mdir.rstrip("/\\"))
    out = os.path.join(ROOT, "out", "c4_compare_20261105")
    p = os.path.join(out, "partial_c4", tag)
    if os.path.exists(os.path.join(p, "metrics.json")):
        pj = json.load(open(os.path.join(p, "metrics.json"), encoding="utf-8")).get("partial", {})
        return p, f"打ち切り（{pj.get('last_epoch')}エポック目まで）・打ち切り時点の最良エポック {pj.get('best_epoch')}"
    for nm in ("c4_models_chain_20261105.log", "c4_compare_chain_20261105.log"):
        lp = os.path.join(out, nm)
        if os.path.exists(lp) and f"!!! {tag} " in open(lp, encoding="utf-8", errors="replace").read():
            if not os.path.exists(os.path.join(p, "model.pt")):
                return None, "打ち切り・数なし（重みの保存なし）"
            return None, "打ち切り・採点待ち"
    return None, ""


def md_table(res, title_col="技法"):
    L = [f"| {title_col} | 正解の数 | モデルが出した数 | 当てた数 | 余計に出した数 | 見落とした数 |", "|---|---|---|---|---|---|"]
    for c in TYPES + ["ALL"]:
        e = res[c]
        L.append(f"| {JP[c]} | {e['true']} | {e['pred']} | {e['hit']} | {e['extra']} | {e['miss']} |")
    return L
