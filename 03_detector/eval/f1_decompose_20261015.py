# -*- coding: utf-8 -*-
r"""F値 が動いた理由を数字で切り分ける（2026-10-15）。

build_index.py を 10/15 の追記後に流し直すと、test 9/24 の正解が 1,182 → 1,194 件になる。
このとき F値 は 2 つの経路で動く。
  (i)  正解そのものが増える（TP／FN が動く）
  (ii) 評価区間 span＝[最初の正解−5s, 最後の正解+2s] が (i) の影響で広がる → 区間内の予測が増えて FP が動く
さらに 10/15 に入れた
  (iii) --valid_range exclude（観測できた区間＝dataset/song_valid_range.csv との重なりに狭める）
も効きうる。この 3 つを分離して出す。

予測は build_ledgers_20261008.py の --prob_cache に保存した確率をそのまま読む（推論はしない）。
pick／match／しきい値は build_ledgers_20261008.py・render_pitchbar.py と同一の規則を逐語で写してある。

使い方:
  python f1_decompose_20261015.py [--cache <probcache dir>] [--old_index <index.json のバックアップ>] [--out <md>]
"""
import os, sys, json, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from song_valid_range_20261015 import clip_span
sys.stdout.reconfigure(encoding="utf-8")

BASE = "G:/研究アーカイブ/成果物/動画解析"; ROOT = f"{BASE}/g3v2_20260928"; FR = 100
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JA = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング"}
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=f"{ROOT}/out/probcache_20261015")
ap.add_argument("--old_index", default=f"{ROOT}/index.json.pre_rebuild_20261015.bak")
ap.add_argument("--new_index", default=f"{ROOT}/index.json")
ap.add_argument("--tags", default="v2d,v2d2,v2e"); ap.add_argument("--lag", type=float, default=0.3)
ap.add_argument("--tol", type=float, default=1.0); ap.add_argument("--hid", type=int, default=128)
ap.add_argument("--test", default="20260924")
ap.add_argument("--out", default=f"{BASE}/dataset/ledgers_20261008/F値_切り分け_20260924_20261015.md")
a = ap.parse_args(); TAGS = a.tags.split(",")

best = {c: tuple(v) for c, v in json.load(open(f"{ROOT}/out/pitchbar_thr_{'+'.join(TAGS)}_lag{a.lag}.json", encoding="utf-8")).items()}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def pick(prob, thr, md):   # build_ledgers_20261008.py と逐語同一
    w = int(md * FR); n = len(prob)
    return [(i / FR + a.lag, float(prob[i])) for i in range(1, n - 1) if prob[i] >= thr and prob[i] >= prob[max(0, i - w):min(n, i + w + 1)].max()]


def match(pred, truth, tol):   # build_ledgers_20261008.py と逐語同一（貪欲1対1）
    used = set(); pairs = []
    for pi in sorted(range(len(pred)), key=lambda i: pred[i]):
        bk = None
        for k, t in enumerate(truth):
            if k in used or abs(t - pred[pi]) > tol: continue
            if bk is None or abs(t - pred[pi]) < abs(truth[bk] - pred[pi]): bk = k
        if bk is not None: used.add(bk); pairs.append((pi, bk))
    return pairs


def load(p):
    return {d["video"]: d for d in json.load(open(p, encoding="utf-8")) if d["session"] == a.test}


OLD = load(a.old_index); NEW = load(a.new_index)
assert set(OLD) == set(NEW), "曲集合が変わっている＝評価条件が同一でない"
PROB = {}
for v in NEW:
    p = f"{a.cache}/{v}__{'+'.join(TAGS)}_h{a.hid}.npy"
    assert os.path.exists(p), f"確率キャッシュなし: {p}（先に build_ledgers_20261008.py --prob_cache を流す）"
    PROB[v] = np.load(p)
PICK = {(v, c): pick(PROB[v][TYPES.index(c)], *best[c]) for v in NEW for c in TYPES}


def run(truth_src, span_src, vr):
    """truth_src／span_src ＝ OLD or NEW（正解と span をどちらから取るか）・vr ＝ valid_range"""
    per = {}
    for v in sorted(NEW):
        d = span_src[v]; off = float(d.get("offset", 0.0))
        lo, hi = clip_span(v, d["span"], off, vr)[0]
        for c in TYPES:
            tr = [t for cc, t in truth_src[v]["events"] if cc == c and lo <= t <= hi]
            pr = [t for t, _ in PICK[(v, c)] if lo <= t <= hi]
            m = len(match(pr, tr, a.tol))
            per[(v, c)] = (m, len(pr) - m, len(tr) - m)
    tot = tuple(sum(x[i] for x in per.values()) for i in range(3))
    bytech = {c: tuple(sum(per[(v, c)][i] for v in NEW) for i in range(3)) for c in TYPES}
    return tot, bytech, per


CASES = [
    ("(a) 従来＝正解1,182・span従来・keep", OLD, OLD, "keep"),
    ("(a') 従来の正解・span従来・exclude", OLD, OLD, "exclude"),
    ("(b1) 正解1,194・span従来のまま・keep", NEW, OLD, "keep"),
    ("(b) 正解1,194・span再計算・exclude（既定）", NEW, NEW, "exclude"),
    ("(c) 正解1,194・span再計算・keep（除外なし）", NEW, NEW, "keep"),
]
R = {}
L = [f"# F値 の切り分け（test {a.test}・28曲・±{a.tol}s・2026-10-15）", "",
     "予測確率は 1 度だけ推論して使い回している（build_ledgers_20261008.py --prob_cache）。しきい値・lag・照合規則は全ケース同一。", "",
     "| 条件 | 正解 | 予測 | TP | FP | FN | 適合率 | 再現率 | **F値** | 未検出率 | 誤検出率 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for name, ts, ss, vr in CASES:
    tot, bt, per = run(ts, ss, vr); R[name] = (tot, bt)
    tp, fp, fn = tot; p, r, f = prf(tp, fp, fn)
    L.append(f"| {name} | {tp + fn} | {tp + fp} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | **{f:.3f}** | {1 - r:.3f} | {1 - p:.3f} |")
    print(name, tot, f"F={f:.4f}")

# 寄与の分解
def d(x, y): return tuple(y[i] - x[i] for i in range(3))
A0 = R["(a) 従来＝正解1,182・span従来・keep"][0]
B1 = R["(b1) 正解1,194・span従来のまま・keep"][0]
B = R["(b) 正解1,194・span再計算・exclude（既定）"][0]
C = R["(c) 正解1,194・span再計算・keep（除外なし）"][0]
L += ["", "## 寄与の分解（ΔTP／ΔFP／ΔFN）", "", "| 要因 | ΔTP | ΔFP | ΔFN | F値 |", "|---|---|---|---|---|"]
for nm, x, y in [("① 正解の追加 12 件（span は従来のまま）", A0, B1),
                 ("② span の再計算（①で区間が広がった分の予測）", B1, C),
                 ("③ 観測区間の除外（--valid_range exclude）", C, B)]:
    dd = d(x, y)
    L.append(f"| {nm} | {dd[0]:+d} | {dd[1]:+d} | {dd[2]:+d} | {prf(*y)[2]:.4f} |")
L.append(f"| **合計（従来 → 既定）** | {d(A0,B)[0]:+d} | {d(A0,B)[1]:+d} | {d(A0,B)[2]:+d} | {prf(*B)[2]:.4f} |")

# 技法別
L += ["", "## 技法別 F値（従来 → 追記後exclude → 追記後keep）", "",
      "| 技法 | 正解 前→後 | TP 前→後 | FP 前→後 | FN 前→後 | F値 従来 | F値 exclude | F値 keep | ΔF |", "|---|---|---|---|---|---|---|---|---|"]
bA = R["(a) 従来＝正解1,182・span従来・keep"][1]; bB = R["(b) 正解1,194・span再計算・exclude（既定）"][1]; bC = R["(c) 正解1,194・span再計算・keep（除外なし）"][1]
for c in TYPES:
    x = bA[c]; y = bB[c]; z = bC[c]
    L.append(f"| {JA[c]} | {x[0]+x[2]}→{y[0]+y[2]} | {x[0]}→{y[0]} | {x[1]}→{y[1]} | {x[2]}→{y[2]} | {prf(*x)[2]:.3f} | {prf(*y)[2]:.3f} | {prf(*z)[2]:.3f} | {prf(*y)[2]-prf(*x)[2]:+.3f} |")
x = A0; y = B; z = C
L.append(f"| **全体** | {x[0]+x[2]}→{y[0]+y[2]} | {x[0]}→{y[0]} | {x[1]}→{y[1]} | {x[2]}→{y[2]} | **{prf(*x)[2]:.3f}** | **{prf(*y)[2]:.3f}** | **{prf(*z)[2]:.3f}** | **{prf(*y)[2]-prf(*x)[2]:+.3f}** |")

# span の変化と観測区間の効き
L += ["", "## 評価区間 span の変化（録音WAV時刻・秒）", "",
      "| 曲 | 正解 前→後 | span 従来 | span 再計算 | 観測区間と重ねた後（exclude） | 追加区間の長さ |", "|---|---|---|---|---|---|"]
n_span = 0
for v in sorted(NEW):
    so = OLD[v]["span"]; sn = NEW[v]["span"]; off = float(NEW[v].get("offset", 0.0))
    sc = clip_span(v, sn, off, "exclude")[0]
    if abs(so[0] - sn[0]) < 1e-6 and abs(so[1] - sn[1]) < 1e-6 and abs(sc[0] - sn[0]) < 1e-6 and abs(sc[1] - sn[1]) < 1e-6: continue
    n_span += 1
    L.append(f"| {v} | {OLD[v]['n']}→{NEW[v]['n']} | {so[0]:.3f}〜{so[1]:.3f} | {sn[0]:.3f}〜{sn[1]:.3f} | {sc[0]:.3f}〜{sc[1]:.3f} | {(sn[1]-sn[0])-(so[1]-so[0]):+.3f}s |")
L.append("")
L.append(f"- span または観測区間で区間が動いた曲＝{n_span}/{len(NEW)}")

# exclude で実際に落ちた正解・予測
drop_t = drop_p = 0; rows_x = []
for v in sorted(NEW):
    d_ = NEW[v]; off = float(d_.get("offset", 0.0)); lo, hi = d_["span"]; lo2, hi2 = clip_span(v, d_["span"], off, "exclude")[0]
    if abs(lo - lo2) < 1e-6 and abs(hi - hi2) < 1e-6: continue
    nt = sum(1 for cc, t in d_["events"] if (lo <= t <= hi) and not (lo2 <= t <= hi2))
    npd = sum(1 for c in TYPES for t, _ in PICK[(v, c)] if (lo <= t <= hi) and not (lo2 <= t <= hi2))
    drop_t += nt; drop_p += npd
    rows_x.append(f"| {v} | {lo:.3f}〜{hi:.3f} | {lo2:.3f}〜{hi2:.3f} | {nt} | {npd} |")
L += ["", "## --valid_range exclude で評価から外れたもの", "", "| 曲 | span 再計算 | 観測区間と重ねた後 | 外れた正解 | 外れた予測 |", "|---|---|---|---|---|"] + rows_x
L.append("")
L.append(f"- 合計＝正解 {drop_t} 件・予測 {drop_p} 件が外れた（どちらも 0 なら exclude と keep は同じ数字になる）")

# 10/14 が評価用データに入っていないことの確認
idx_all = json.load(open(a.new_index, encoding="utf-8"))
ses = collections.Counter(d["session"] for d in idx_all)
L += ["", "## 10/14 が評価用データに入っていないことの確認", "", "| セッション | index.json のエントリ | 件数 |", "|---|---|---|"]
for s in sorted(ses): L.append(f"| {s} | {ses[s]} | {sum(d['n'] for d in idx_all if d['session']==s)} |")
L.append(f"| **20261014** | **{ses.get('20261014', 0)}** | **{sum(d['n'] for d in idx_all if d['session']=='20261014')}** |")
L.append("")
L.append("- build_index.py が読む同期表は sync_offsets.csv（6/28〜9/2）＋ sync_offsets_20260911.csv ＋ sync_offsets_20260924.csv だけ。"
         "10/14（と 9/15）は sync_offsets_<ses>.csv があっても読まれないので index.json に入らない＝学習にも評価にも使われない。")
open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("WROTE", a.out)
