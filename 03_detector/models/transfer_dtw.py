# -*- coding: utf-8 -*-
"""仮説「加点は曲固有」：同じ曲の過去録音とDTWで位置合わせし、過去の加点イベントを検証曲の時間軸へ転写して予測にする。
   使い方: python transfer_dtw.py --test 20260924 [--hop 0.1] [--k 1]
   出力: out/dtw/pred_test.csv, out/dtw/report.md（±0.5/1/2s F1・件数整合・CRNN v2b との同一曲比較）"""
import os, sys, json, argparse, csv, time
import numpy as np, librosa, soundfile as sf
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"; OUT = f"{ROOT}/out/dtw"; os.makedirs(OUT, exist_ok=True)
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser(); ap.add_argument("--test", default="20260924"); ap.add_argument("--hop", type=float, default=0.1); ap.add_argument("--k", type=int, default=1, help="使う過去演奏の数（新しい順）"); ap.add_argument("--vote", type=float, default=0.5, help="複数演奏のとき採用に必要な支持率")
ap.add_argument("--min_dist", type=float, default=0.3)
a = ap.parse_args()
index = json.load(open(f"{ROOT}/index.json", encoding="utf-8")); titles = json.load(open(f"{ROOT}/titles.json", encoding="utf-8"))
SR = 22050; HOP = int(round(a.hop * SR))
def chroma(d):
    cache = f"{ROOT}/feats/{d['video']}_chroma{int(a.hop*1000)}.npy"
    if os.path.exists(cache): return np.load(cache)
    y, sr = sf.read(d["wav"], dtype="float32", always_2d=True); y = y.mean(1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    C = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=4096, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=HOP)[0]
    X = np.vstack([C, (np.log(rms + 1e-6) / 5.0)[None, :]]).astype(np.float32); np.save(cache, X); return X
def align(src, dst):
    """src(過去)→dst(検証) の時刻写像。DTW（cosine）の経路から単調写像を作る"""
    D, wp = librosa.sequence.dtw(X=src, Y=dst, metric="cosine", subseq=False)
    wp = wp[::-1]  # (i_src, j_dst) 昇順
    si = wp[:, 0] * a.hop; dj = wp[:, 1] * a.hop
    # 同じ src 時刻に複数 dst → 平均
    u, inv = np.unique(si, return_inverse=True); m = np.zeros(len(u)); cnt = np.zeros(len(u))
    np.add.at(m, inv, dj); np.add.at(cnt, inv, 1); m /= cnt
    cost = float(D[-1, -1] / (src.shape[1] + dst.shape[1]))
    return (lambda t: np.interp(t, u, m)), cost
def match(pred, truth, tol):
    used = set(); hit = 0
    for p in sorted(pred):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - p) > tol: continue
            if best is None or abs(t - p) < abs(truth[best] - p): best = k
        if best is not None: used.add(best); hit += 1
    return hit
test = [d for d in index if d["session"] == a.test]; hist = [d for d in index if d["session"] != a.test]
by_title = {}
for d in hist: by_title.setdefault(titles[d["video"]], []).append(d)
preds = {}; log = []; t0 = time.time()
for d in test:
    past = sorted(by_title.get(titles[d["video"]], []), key=lambda x: x["session"], reverse=True)[:a.k]
    if not past: continue
    Xd = chroma(d); ev_lists = []
    for p in past:
        f, cost = align(chroma(p), Xd)
        lo, hi = p["span"]; ev = [(c, float(f(t))) for c, t in p["events"] if lo <= t <= hi]; ev_lists.append(ev)
        log.append(dict(video=d["video"], past=p["video"], cost=round(cost, 4), n_past_events=len(ev)))
    # 複数演奏の合議：技法別に時刻をクラスタ（min_dist）し支持率>=vote で採用
    out = []
    for c in TYPES:
        ts = sorted([(t, i) for i, ev in enumerate(ev_lists) for cc, t in ev if cc == c])
        clusters = []
        for t, i in ts:
            if clusters and t - clusters[-1][-1][0] <= a.min_dist: clusters[-1].append((t, i))
            else: clusters.append([(t, i)])
        for cl in clusters:
            support = len({i for _, i in cl}) / len(ev_lists)
            if support >= a.vote: out.append((c, float(np.mean([t for t, _ in cl]))))
    preds[d["video"]] = out
    print(d["video"], titles[d["video"]], "past", len(past), "pred", len(out), f"{time.time()-t0:.0f}s", flush=True)
# 評価
vids = list(preds); rep = [f"# DTW転写（test={a.test}・k={a.k}・vote={a.vote}）", f"対象＝過去に同曲がある {len(vids)}/{len(test)} 曲", ""]
for tol in (0.5, 1.0, 2.0):
    agg = {c: [0, 0, 0] for c in TYPES}
    for v in vids:
        d = next(x for x in index if x["video"] == v); lo, hi = d["span"]
        for c in TYPES:
            truth = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]; pred = [t for cc, t in preds[v] if cc == c and lo <= t <= hi]
            h = match(pred, truth, tol); agg[c][0] += h; agg[c][1] += len(pred); agg[c][2] += len(truth)
    rep.append(f"## 事象一致 許容±{tol}s"); rep.append("| 技法 | 正解 | 予測 | 一致 | 再現率 | 適合率 | F1 |"); rep.append("|---|---|---|---|---|---|---|")
    H = NP = NT = 0
    for c in TYPES:
        h, npd, nt = agg[c]; H += h; NP += npd; NT += nt
        p = h / npd if npd else 0; r = h / nt if nt else 0; f = 2 * p * r / (p + r) if p + r else 0
        rep.append(f"| {c} | {nt} | {npd} | {h} | {r:.3f} | {p:.3f} | {f:.3f} |")
    p = H / NP if NP else 0; r = H / NT if NT else 0; rep.append(f"| ALL | {NT} | {NP} | {H} | {r:.3f} | {p:.3f} | {2*p*r/(p+r) if p+r else 0:.3f} |"); rep.append("")
# 件数整合（同一曲で CRNN v2b 積分と比較）
crnn = {}
pth = f"{ROOT}/out/v2b/counts_test_integral.csv"
if os.path.exists(pth):
    for r in csv.DictReader(open(pth, encoding="utf-8-sig")): crnn[(r["video"], r["type"])] = float(r["pred"])
rep.append("## 件数整合（同じ曲で比較）"); rep.append("| 技法 | Σ正解 | DTW転写 | CRNN v2b積分 |"); rep.append("|---|---|---|---|")
T = [0, 0, 0]
for c in TYPES:
    s1 = s2 = t = 0
    for v in vids:
        d = next(x for x in index if x["video"] == v); lo, hi = d["span"]
        nt = sum(1 for cc, tt in d["events"] if cc == c and lo <= tt <= hi); npd = sum(1 for cc, tt in preds[v] if cc == c and lo <= tt <= hi)
        s1 += abs(npd - nt); s2 += abs(round(crnn.get((v, c), nt)) - nt); t += nt
    T[0] += t; T[1] += s1; T[2] += s2
    rep.append(f"| {c} | {t} | {1 - s1 / t if t else float('nan'):.3f} | {1 - s2 / t if t else float('nan'):.3f} |")
rep.append(f"| ALL | {T[0]} | {1 - T[1] / T[0]:.3f} | {1 - T[2] / T[0]:.3f} |")
txt = "\n".join(rep); print(txt); open(f"{OUT}/report_k{a.k}.md", "w", encoding="utf-8").write(txt)
with open(f"{OUT}/pred_test_k{a.k}.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["video", "type", "wav_time_sec"])
    for v in vids:
        for c, t in sorted(preds[v], key=lambda x: x[1]): w.writerow([v, c, f"{t:.2f}"])
with open(f"{OUT}/align_log_k{a.k}.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["video", "past", "cost", "n_past_events"]); w.writeheader(); w.writerows(log)
