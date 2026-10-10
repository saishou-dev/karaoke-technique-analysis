# -*- coding: utf-8 -*-
"""G3 v2: フレーム列モデル（Conv1d + BiGRU）で6技法の加点ヒートマップを学習し、事象単位で照合する。
   使い方: python train_crnn.py --tag v2a --test 20260924 --dev 20260911 --epochs 12
   出力: out/<tag>/model.pt, report.md, pred_<split>.csv, persong_<split>.csv, metrics.json"""
import os, sys, json, time, argparse, random, csv
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v2a"); ap.add_argument("--test", default="20260924"); ap.add_argument("--dev", default="20260911")
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--sigma", type=float, default=10); ap.add_argument("--chunk", type=int, default=800); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--hid", type=int, default=128); ap.add_argument("--lagshift", type=float, default=0.0, help="ラベルを何秒前へずらすか（技法共通）")
ap.add_argument("--min_dist", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--no_mel", action="store_true", help="log-melを使わずピッチ系＋RMSだけ")
ap.add_argument("--eval_only", action="store_true", help="学習せず out/<tag>/model.pt を評価")
ap.add_argument("--index", default="index.json", help="使うindexファイル名（index_snapped.json など）")
ap.add_argument("--exclude", default="", help="学習から外すセッション（カンマ区切り）")
ap.add_argument("--pw_cap", type=float, default=30.0, help="pos_weightの上限")
ap.add_argument("--thr_mode", default="f1", choices=["f1", "count"], help="devでのしきい値決定：F1最大 or 件数差最小")
a = ap.parse_args(); torch.set_num_threads(a.threads); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
OUT = f"{ROOT}/out/{a.tag}"; os.makedirs(OUT, exist_ok=True)
index = json.load(open(f"{ROOT}/{a.index}", encoding="utf-8"))
FR = 100  # frames/sec

def load(d):
    z = np.load(f"{ROOT}/feats/{d['video']}.npz")
    lm = z["logmel"].astype(np.float32); T = lm.shape[1]
    st = z["st"]; v = z["voiced"]
    stn = np.where(v > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
    dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (v[:-1] > 0); dst = np.where(v > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
    parts = [] if a.no_mel else [lm]
    parts += [z["rms"][None, :], stn[None, :], v[None, :], z["conf"][None, :], dst[None, :]]
    X = np.concatenate(parts, 0).astype(np.float32)
    Y = np.zeros((len(TYPES), T), np.float32); M = np.zeros(T, np.float32)
    lo, hi = d["span"]; M[max(0, int(lo * FR)):min(T, int(hi * FR))] = 1.0
    g = np.arange(-40, 41); ker = np.exp(-0.5 * (g / a.sigma) ** 2).astype(np.float32)
    ev = []
    for c, t in d["events"]:
        if c not in TYPES: continue
        f = int(round((t - a.lagshift) * FR)); ev.append((c, t))
        i0 = f - 40; s0 = max(0, -i0); e0 = min(81, T - i0)
        if e0 > s0:
            k = TYPES.index(c); Y[k, i0 + s0:i0 + e0] = np.maximum(Y[k, i0 + s0:i0 + e0], ker[s0:e0])
    return X, Y, M, ev

data = {}
for d in index:
    if not os.path.exists(f"{ROOT}/feats/{d['video']}.npz"): continue
    data[d["video"]] = (d,) + load(d)
CIN = next(iter(data.values()))[1].shape[0]
EXC = set(x for x in a.exclude.split(",") if x)
train = [v for v, (d, *_) in data.items() if d["session"] not in (a.test, a.dev) and d["session"] not in EXC]
dev = [v for v, (d, *_) in data.items() if d["session"] == a.dev]
test = [v for v, (d, *_) in data.items() if d["session"] == a.test]
print(f"train {len(train)} dev {len(dev)} test {len(test)} cin {CIN}", flush=True)
allX = np.concatenate([data[v][1] for v in train], 1); mu = allX.mean(1, keepdims=True); sd = allX.std(1, keepdims=True) + 1e-5; del allX
np.save(f"{OUT}/norm.npy", np.stack([mu[:, 0], sd[:, 0]]))
pos = np.array([sum((data[v][2][k] > 0.5).sum() for v in train) for k in range(len(TYPES))], float); tot = sum(data[v][2].shape[1] for v in train)
pw = torch.tensor(np.clip((tot - pos) / np.maximum(pos, 1), 1, a.pw_cap), dtype=torch.float32); print("pos_weight", [round(float(x), 1) for x in pw], flush=True)

class Net(nn.Module):
    def __init__(s, cin=CIN, h=a.hid, nout=len(TYPES)):
        super().__init__()
        s.c1 = nn.Sequential(nn.Conv1d(cin, h, 5, padding=2), nn.BatchNorm1d(h), nn.ReLU(),
                             nn.Conv1d(h, h, 5, padding=2, stride=2), nn.BatchNorm1d(h), nn.ReLU(),
                             nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15))
        s.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        s.head = nn.Linear(2 * h, nout)
    def forward(s, x):
        T = x.shape[-1]; h = s.c1(x); h, _ = s.gru(h.transpose(1, 2)); o = s.head(h).transpose(1, 2)
        return Fn.interpolate(o, size=T, mode="linear", align_corners=False)

net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
mu_t = torch.tensor(mu); sd_t = torch.tensor(sd)

def batches(vids, chunk, bs):
    items = []
    for v in vids:
        T = data[v][1].shape[1]
        for s in range(0, max(1, T - chunk), chunk // 2): items.append((v, s))
    random.shuffle(items)
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        X = np.zeros((len(b), CIN, chunk), np.float32); Y = np.zeros((len(b), len(TYPES), chunk), np.float32); M = np.zeros((len(b), chunk), np.float32)
        for j, (v, s) in enumerate(b):
            x, y, m = data[v][1][:, s:s + chunk], data[v][2][:, s:s + chunk], data[v][3][s:s + chunk]; L = x.shape[1]
            X[j, :, :L] = x; Y[j, :, :L] = y; M[j, :L] = m
        yield (torch.tensor(X) - mu_t) / sd_t, torch.tensor(Y), torch.tensor(M)

def predict(v):
    net.eval()
    with torch.no_grad():
        x = (torch.tensor(data[v][1][None]) - mu_t) / sd_t
        return torch.sigmoid(net(x))[0].numpy()

def pick(prob, thr, min_dist):
    w = int(min_dist * FR); out = []
    n = len(prob)
    for i in range(1, n - 1):
        if prob[i] >= thr and prob[i] >= prob[max(0, i - w):min(n, i + w + 1)].max(): out.append((i / FR, float(prob[i])))
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

def evaluate(vids, thr, tol, min_dist, probs):
    agg = {c: [0, 0, 0] for c in TYPES}; persong = []
    for v in vids:
        d = data[v][0]; lo, hi = d["span"]; P = probs[v]
        for k, c in enumerate(TYPES):
            truth = [t for cc, t in data[v][4] if cc == c and lo <= t <= hi]
            pred = [t for t, _ in pick(P[k], thr[c], min_dist) if lo <= t <= hi]
            h = match(pred, truth, tol); agg[c][0] += h; agg[c][1] += len(pred); agg[c][2] += len(truth)
            persong.append(dict(video=v, type=c, n_true=len(truth), n_pred=len(pred), hit=h))
    res = {}
    for c in TYPES:
        h, npred, ntrue = agg[c]
        p = h / npred if npred else 0.0; r = h / ntrue if ntrue else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        res[c] = dict(hit=h, pred=npred, true=ntrue, precision=round(p, 3), recall=round(r, 3), f1=round(f, 3))
    H = sum(x[0] for x in agg.values()); NP = sum(x[1] for x in agg.values()); NT = sum(x[2] for x in agg.values())
    p = H / NP if NP else 0; r = H / NT if NT else 0
    res["ALL"] = dict(hit=H, pred=NP, true=NT, precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / (p + r), 3) if (p + r) else 0)
    return res, persong

def sweep_thr(vids, probs, tol=0.5):
    thr = {}
    for c in TYPES:
        bestf = None
        for th in np.arange(0.05, 0.96, 0.025):
            r, ps = evaluate(vids, {cc: (th if cc == c else 0.99) for cc in TYPES}, tol, a.min_dist, probs)
            if a.thr_mode == "count":
                score = -sum(abs(x["n_pred"] - x["n_true"]) for x in ps if x["type"] == c)
            else:
                score = r[c]["f1"]
            if bestf is None or score > bestf[0]: bestf = (score, float(th))
        thr[c] = bestf[1]
    return thr

t0 = time.time(); best = None
for ep in range(0 if a.eval_only else a.epochs):
    net.train(); tl = 0; nb = 0
    for X, Y, M in batches(train, a.chunk, a.bs):
        logit = net(X)
        loss = (Fn.binary_cross_entropy_with_logits(logit, Y, pos_weight=pw[None, :, None], reduction="none") * M[:, None, :]).sum() / (M.sum() * len(TYPES) + 1)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); tl += loss.item(); nb += 1
    probs = {v: predict(v) for v in dev}; thr = sweep_thr(dev, probs)
    rd, ps = evaluate(dev, thr, 0.5, a.min_dist, probs)
    cagree = 1 - sum(abs(x["n_pred"] - x["n_true"]) for x in ps) / max(1, sum(x["n_true"] for x in ps))
    score = cagree if a.thr_mode == "count" else rd["ALL"]["f1"]
    print(f"ep{ep + 1} loss={tl / nb:.4f} dev F1(±0.5s)={rd['ALL']['f1']:.3f} dev件数整合={cagree:.3f} " + " ".join(f"{c[:4]}={rd[c]['f1']:.2f}" for c in TYPES) + f" thr={ {c[:4]: round(t, 2) for c, t in thr.items()} } {time.time() - t0:.0f}s", flush=True)
    torch.save(net.state_dict(), f"{OUT}/last.pt")
    if best is None or score > best[0]:
        best = (score, ep + 1, dict(thr)); torch.save(net.state_dict(), f"{OUT}/model.pt")
if a.eval_only:
    net.load_state_dict(torch.load(f"{OUT}/model.pt")); probs = {v: predict(v) for v in dev}; thr = sweep_thr(dev, probs); rd, _ = evaluate(dev, thr, 0.5, a.min_dist, probs); best = (rd["ALL"]["f1"], -1, thr)
print("best dev", best, flush=True)
net.load_state_dict(torch.load(f"{OUT}/model.pt")); thr = best[2]
rep = [f"# G3 v2 {a.tag}（{time.strftime('%Y-%m-%d %H:%M')}）", "",
       "学習＝" + ",".join(sorted({data[v][0]['session'] for v in train})) + f"／dev（しきい値決定）＝{a.dev}／test（未見）＝{a.test}",
       f"epochs={a.epochs} sigma={a.sigma}f lagshift={a.lagshift}s chunk={a.chunk} hid={a.hid} no_mel={a.no_mel} best_ep={best[1]}", f"しきい値={thr}", ""]
metrics = {}
for name, vids in (("dev", dev), ("test", test)):
    probs = {v: predict(v) for v in vids}
    for tol in (0.5, 1.0, 2.0):
        r, ps = evaluate(vids, thr, tol, a.min_dist, probs); metrics[f"{name}_tol{tol}"] = r
        rep.append(f"## {name}（{'/'.join(sorted({data[v][0]['session'] for v in vids}))}）許容±{tol}s")
        rep.append("| 技法 | 正解 | 予測 | 一致 | 再現率 | 適合率 | F1 |"); rep.append("|---|---|---|---|---|---|---|")
        for c in TYPES + ["ALL"]:
            x = r[c]; rep.append(f"| {c} | {x['true']} | {x['pred']} | {x['hit']} | {x['recall']} | {x['precision']} | {x['f1']} |")
        rep.append("")
        if tol == 0.5:
            with open(f"{OUT}/persong_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(ps[0].keys())); w.writeheader(); w.writerows(ps)
    with open(f"{OUT}/pred_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["video", "type", "wav_time_sec", "prob"])
        for v in vids:
            for k, c in enumerate(TYPES):
                for t, p in pick(probs[v][k], thr[c], a.min_dist): w.writerow([v, c, f"{t:.2f}", f"{p:.3f}"])
open(f"{OUT}/report.md", "w", encoding="utf-8").write("\n".join(rep))
json.dump(dict(metrics=metrics, thr=thr, best=best), open(f"{OUT}/metrics.json", "w"), indent=1)
print("\n".join(rep)); print("DONE", OUT)
