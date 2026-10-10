# -*- coding: utf-8 -*-
"""学習済みモデル（吸着ラベル）の確率から、dev（9/11）で「G1生時刻 ±1s F1 最大」のしきい値と最短間隔を技法別に決め、test（9/24）で評価する。
   使い方: python rethreshold_raw.py --tag v2d [--lag 0.3]"""
import sys, json, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"; FR = 100
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True, help="カンマ区切りで複数指定するとアンサンブル（確率平均）"); ap.add_argument("--lag", type=float, default=0.3); ap.add_argument("--hid", type=int, default=128); ap.add_argument("--dev", default="20260911"); ap.add_argument("--test", default="20260924")
a = ap.parse_args(); torch.set_num_threads(4)
TAGS = a.tag.split(","); OUT = f"{ROOT}/out/{TAGS[0]}"; index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
norm = np.load(f"{OUT}/norm.npy"); mu = torch.tensor(norm[0][:, None]); sd = torch.tensor(norm[1][:, None]); CIN = norm.shape[1]
def feats(d):
    z = np.load(f"{ROOT}/feats/{d['video']}.npz"); lm = z["logmel"].astype(np.float32); T = lm.shape[1]
    st = z["st"]; v = z["voiced"]; stn = np.where(v > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
    dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (v[:-1] > 0); dst = np.where(v > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
    return np.concatenate([lm, z["rms"][None, :], stn[None, :], v[None, :], z["conf"][None, :], dst[None, :]], 0).astype(np.float32)
class Net(nn.Module):
    def __init__(s, cin=CIN, h=a.hid, nout=len(TYPES)):
        super().__init__()
        s.c1 = nn.Sequential(nn.Conv1d(cin, h, 5, padding=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Conv1d(h, h, 5, padding=2, stride=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15))
        s.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15); s.head = nn.Linear(2 * h, nout)
    def forward(s, x):
        T = x.shape[-1]; h = s.c1(x); h, _ = s.gru(h.transpose(1, 2)); o = s.head(h).transpose(1, 2)
        return Fn.interpolate(o, size=T, mode="linear", align_corners=False)
nets = []
for tg in TAGS:
    n_ = Net(); n_.load_state_dict(torch.load(f"{ROOT}/out/{tg}/model.pt")); n_.eval(); nm = np.load(f"{ROOT}/out/{tg}/norm.npy"); nets.append((n_, torch.tensor(nm[0][:, None]), torch.tensor(nm[1][:, None])))
def probs(d):
    x = torch.tensor(feats(d)[None])
    with torch.no_grad(): return np.mean([torch.sigmoid(n_((x - m_) / s_))[0].numpy() for n_, m_, s_ in nets], 0)
def pick(prob, thr, md):
    w = int(md * FR); n = len(prob); return [i / FR + a.lag for i in range(1, n - 1) if prob[i] >= thr and prob[i] >= prob[max(0, i - w):min(n, i + w + 1)].max()]
def match(pred, truth, tol):
    used = set(); hit = 0
    for p in sorted(pred):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - p) > tol: continue
            if best is None or abs(t - p) < abs(truth[best] - p): best = k
        if best is not None: used.add(best); hit += 1
    return hit
sets = {n: [d for d in index if d["session"] == s] for n, s in (("dev", a.dev), ("test", a.test))}
P = {d["video"]: probs(d) for s in sets.values() for d in s}
def stats(ds, c, thr, md, tol):
    k = TYPES.index(c); H = NP = NT = 0
    for d in ds:
        lo, hi = d["span"]; tr = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]; pr = [t for t in pick(P[d["video"]][k], thr, md) if lo <= t <= hi]
        H += match(pr, tr, tol); NP += len(pr); NT += len(tr)
    p = H / NP if NP else 0; r = H / NT if NT else 0; return (2 * p * r / (p + r) if p + r else 0), p, r, NP, NT, H
best = {}
for c in TYPES:
    cand = max(((stats(sets["dev"], c, th, md, 1.0)[0], th, md) for th in np.arange(0.2, 0.96, 0.05) for md in (0.3, 0.5, 0.8)), key=lambda x: x[0]); best[c] = (cand[1], cand[2])
print("dev で決めた (しきい値, 最短間隔s):", {c: (round(v[0], 2), v[1]) for c, v in best.items()})
for tol in (0.5, 1.0, 2.0):
    print(f"## test 許容±{tol}s"); print("| 技法 | 正解 | 予測 | 一致 | 再現率 | 適合率 | F1 |"); print("|---|---|---|---|---|---|---|"); HH = NNP = NNT = 0
    for c in TYPES:
        f, p, r, NP, NT, H = stats(sets["test"], c, best[c][0], best[c][1], tol); HH += H; NNP += NP; NNT += NT
        print(f"| {c} | {NT} | {NP} | {H} | {r:.3f} | {p:.3f} | {f:.3f} |")
    p = HH / NNP if NNP else 0; r = HH / NNT if NNT else 0; print(f"| ALL | {NNT} | {NNP} | {HH} | {r:.3f} | {p:.3f} | {2*p*r/(p+r) if p+r else 0:.3f} |")
