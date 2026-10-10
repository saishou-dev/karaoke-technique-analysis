# -*- coding: utf-8 -*-
"""DTW転写イベント（位置の事前知識）× CRNN確率（今回の歌唱の証拠）のゲート。
   転写イベントの±w秒内のCRNN最大確率が τ 以上なら採用。τ は dev（9/11）でF1最大に決め、test（9/24）で測る。
   使い方: python gate_transfer.py --tag v2b"""
import os, sys, json, argparse, csv
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"; FR = 100
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v2b"); ap.add_argument("--w", type=float, default=1.0); ap.add_argument("--hid", type=int, default=128)
a = ap.parse_args(); torch.set_num_threads(8)
OUT = f"{ROOT}/out/{a.tag}"; index = {d["video"]: d for d in json.load(open(f"{ROOT}/index.json", encoding="utf-8"))}
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
net = Net(); net.load_state_dict(torch.load(f"{OUT}/model.pt")); net.eval()
def probs(d):
    with torch.no_grad(): return torch.sigmoid(net((torch.tensor(feats(d)[None]) - mu) / sd))[0].numpy()
def load_pred(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")): out.setdefault(r["video"], []).append((r["type"], float(r["wav_time_sec"])))
    return out
def match(pred, truth, tol):
    used = set(); hit = 0
    for p in sorted(pred):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - p) > tol: continue
            if best is None or abs(t - p) < abs(truth[best] - p): best = k
        if best is not None: used.add(best); hit += 1
    return hit
def gated(pred, P, c, tau):
    k = TYPES.index(c); out = []
    for cc, t in pred:
        if cc != c: continue
        i0 = max(0, int((t - a.w) * FR)); i1 = min(P.shape[1], int((t + a.w) * FR))
        if i1 > i0 and P[k, i0:i1].max() >= tau: out.append(t)
    return out
def evaluate(preds, P_all, tau, tol):
    agg = {c: [0, 0, 0] for c in TYPES}
    for v, pred in preds.items():
        d = index[v]; lo, hi = d["span"]
        for c in TYPES:
            truth = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]; pr = [t for t in gated(pred, P_all[v], c, tau[c]) if lo <= t <= hi]
            h = match(pr, truth, tol); agg[c][0] += h; agg[c][1] += len(pr); agg[c][2] += len(truth)
    res = {}
    for c in TYPES + ["ALL"]:
        h, npd, nt = (sum(x[i] for x in agg.values()) for i in range(3)) if c == "ALL" else agg[c]
        p = h / npd if npd else 0; r = h / nt if nt else 0; res[c] = dict(true=nt, pred=npd, hit=h, precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / (p + r), 3) if p + r else 0)
    return res
dev = load_pred(f"{ROOT}/out/dtw/pred_dev0911_k1.csv"); test = load_pred(f"{ROOT}/out/dtw/pred_test_k1.csv")
P = {v: probs(index[v]) for v in list(dev) + list(test)}
tau = {}
for c in TYPES:
    best = None
    for t_ in np.arange(0.0, 0.96, 0.05):
        r = evaluate(dev, P, {cc: (t_ if cc == c else 0.0) for cc in TYPES}, 1.0)[c]
        if best is None or r["f1"] > best[0]: best = (r["f1"], float(t_))
    tau[c] = best[1]
print("τ（dev 9/11・±1s F1最大）", tau)
rep = [f"# ゲート付きDTW転写（CRNN {a.tag}・w=±{a.w}s）", f"τ={tau}", ""]
for name, preds in (("dev 9/11", dev), ("test 9/24", test)):
    for tol in (0.5, 1.0, 2.0):
        r0 = evaluate(preds, P, {c: 0.0 for c in TYPES}, tol); r1 = evaluate(preds, P, tau, tol)
        rep.append(f"## {name} 許容±{tol}s（ゲートなし → ゲートあり）"); rep.append("| 技法 | 正解 | 予測 | 一致 | 再現率 | 適合率 | F1 | ← ゲートなしF1 |"); rep.append("|---|---|---|---|---|---|---|---|")
        for c in TYPES + ["ALL"]:
            x = r1[c]; rep.append(f"| {c} | {x['true']} | {x['pred']} | {x['hit']} | {x['recall']} | {x['precision']} | {x['f1']} | {r0[c]['f1']} |")
        rep.append("")
txt = "\n".join(rep); print(txt); open(f"{ROOT}/out/dtw/gate_report_{a.tag}.md", "w", encoding="utf-8").write(txt)
