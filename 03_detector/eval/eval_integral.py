# -*- coding: utf-8 -*-
"""学習済みモデル（out/<tag>/model.pt）の確率を曲・窓で積分して件数を推定し、正解件数と比べる。
   使い方: python eval_integral.py --tag v2b [--dev 20260911 --test 20260924 --win 30]
   件数推定＝ Σprob（採点区間内）× 係数（devで Σ正解/Σprob として技法別に校正）。ピーク抽出（件数校正しきい値）との比較も出す。"""
import os, sys, json, argparse, csv
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--dev", default="20260911"); ap.add_argument("--test", default="20260924")
ap.add_argument("--win", type=float, default=30.0); ap.add_argument("--hid", type=int, default=128); ap.add_argument("--no_mel", action="store_true"); ap.add_argument("--min_dist", type=float, default=0.3)
a = ap.parse_args(); torch.set_num_threads(8)
OUT = f"{ROOT}/out/{a.tag}"; FR = 100
index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
norm = np.load(f"{OUT}/norm.npy"); mu = torch.tensor(norm[0][:, None]); sd = torch.tensor(norm[1][:, None])
def feats(d):
    z = np.load(f"{ROOT}/feats/{d['video']}.npz"); lm = z["logmel"].astype(np.float32); T = lm.shape[1]
    st = z["st"]; v = z["voiced"]; stn = np.where(v > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
    dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (v[:-1] > 0); dst = np.where(v > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
    parts = [] if a.no_mel else [lm]; parts += [z["rms"][None, :], stn[None, :], v[None, :], z["conf"][None, :], dst[None, :]]
    return np.concatenate(parts, 0).astype(np.float32)
CIN = norm.shape[1]
class Net(nn.Module):
    def __init__(s, cin=CIN, h=a.hid, nout=len(TYPES)):
        super().__init__()
        s.c1 = nn.Sequential(nn.Conv1d(cin, h, 5, padding=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Conv1d(h, h, 5, padding=2, stride=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15))
        s.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15); s.head = nn.Linear(2 * h, nout)
    def forward(s, x):
        T = x.shape[-1]; h = s.c1(x); h, _ = s.gru(h.transpose(1, 2)); o = s.head(h).transpose(1, 2)
        return Fn.interpolate(o, size=T, mode="linear", align_corners=False)
net = Net(); net.load_state_dict(torch.load(f"{OUT}/model.pt")); net.eval()
def probs(d):
    with torch.no_grad(): return torch.sigmoid(net((torch.tensor(feats(d)[None]) - mu) / sd))[0].numpy()
def pick(prob, thr, min_dist):
    w = int(min_dist * FR); n = len(prob); return [i / FR for i in range(1, n - 1) if prob[i] >= thr and prob[i] >= prob[max(0, i - w):min(n, i + w + 1)].max()]
sets = {"dev": [d for d in index if d["session"] == a.dev], "test": [d for d in index if d["session"] == a.test]}
P = {d["video"]: probs(d) for s in sets.values() for d in s}
def truth(d, c):
    lo, hi = d["span"]; return [t for cc, t in d["events"] if cc == c and lo <= t <= hi]
# 校正（dev）: 積分係数
scale = {}
for k, c in enumerate(TYPES):
    sp = sum(P[d["video"]][k, max(0, int(d["span"][0] * FR)):int(d["span"][1] * FR)].sum() for d in sets["dev"]); st_ = sum(len(truth(d, c)) for d in sets["dev"])
    scale[c] = st_ / sp if sp > 0 else 0.0
# 校正（dev）: ピークしきい値（件数差最小）
thr = {}
for k, c in enumerate(TYPES):
    best = None
    for th in np.arange(0.05, 0.96, 0.025):
        err = sum(abs(len([t for t in pick(P[d["video"]][k], th, a.min_dist) if d["span"][0] <= t <= d["span"][1]]) - len(truth(d, c))) for d in sets["dev"])
        if best is None or err < best[0]: best = (err, float(th))
    thr[c] = best[1]
print("積分係数", {c: round(v, 3) for c, v in scale.items()}); print("件数校正しきい値", thr)
lines = []
for name in ("dev", "test"):
    ds = sets[name]
    for mode in ("integral", "peaks"):
        agg = {c: [0, 0, 0, 0] for c in TYPES}; wagg = {c: [0, 0, 0, 0] for c in TYPES}; rows = []
        for d in ds:
            lo, hi = d["span"]; edges = np.arange(lo, hi + a.win, a.win); pr = P[d["video"]]
            for k, c in enumerate(TYPES):
                tr = truth(d, c); ht, _ = np.histogram(tr, edges)
                if mode == "integral":
                    seg = pr[k]; t = np.arange(len(seg)) / FR
                    hp = np.array([seg[(t >= e0) & (t < e1)].sum() * scale[c] for e0, e1 in zip(edges[:-1], edges[1:])]); n_pred = hp.sum(); hp = np.round(hp)
                else:
                    pk = [t for t in pick(pr[k], thr[c], a.min_dist) if lo <= t <= hi]; hp, _ = np.histogram(pk, edges); n_pred = len(pk)
                n_true = len(tr); npr = int(round(n_pred))
                agg[c][0] += abs(npr - n_true); agg[c][1] += n_true; agg[c][2] += 1; agg[c][3] += int(npr == n_true)
                wagg[c][0] += np.abs(hp - ht).sum(); wagg[c][1] += ht.sum(); wagg[c][2] += int((hp == ht).sum()); wagg[c][3] += len(ht)
                rows.append(dict(video=d["video"], type=c, true=n_true, pred=npr))
        lines.append(f"## {name}（{a.dev if name=='dev' else a.test}）方式＝{mode}"); lines.append(f"| 技法 | Σ正解 | Σ|差| | 件数整合 | 曲で完全一致 | {int(a.win)}秒窓の整合 | 窓の完全一致 |"); lines.append("|---|---|---|---|---|---|---|")
        T0 = T1 = W0 = W1 = 0
        for c in TYPES:
            s, t, n, ex = agg[c]; ws, wt, wex, wn = wagg[c]; T0 += s; T1 += t; W0 += ws; W1 += wt
            lines.append(f"| {c} | {t} | {s} | {1 - s / t if t else float('nan'):.3f} | {ex}/{n} | {1 - ws / wt if wt else float('nan'):.3f} | {wex}/{wn} |")
        lines.append(f"| ALL | {T1} | {T0} | {1 - T0 / T1:.3f} |  | {1 - W0 / W1:.3f} |  |"); lines.append("")
        with open(f"{OUT}/counts_{name}_{mode}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
txt = "\n".join(lines); print(txt); open(f"{OUT}/counts_report.md", "w", encoding="utf-8").write(f"# 件数整合 {a.tag}\n\n積分係数 {scale}\nしきい値 {thr}\n\n" + txt)
