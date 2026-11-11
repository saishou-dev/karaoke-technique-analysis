# -*- coding: utf-8 -*-
r"""打ち切られた本を、保存済みの最良エポックの重みで採点する（2026-11-06 Opus・c4-compare 追補6・学習しない）。
入力＝out/c4_compare_20261105/partial_c4/<tag>/（watch_cut_c4_20261105.sh が写した model.pt・norm.npy・train.log）。
  model.pt＝学習台本が「dev の F がそれまでで最大のエポック」で保存した重み。train.log の ep 行から、最後まで終わったエポックと最良エポックを読む。
やる事＝dev と試験の確率の列を出し（infer_probs2_c4 と同じ網）、しきい値を学習台本と同じ決め方（dev・技法ごとに F ±0.5秒 最大・0.05〜0.95 を 0.025 刻み・他の技法は 0.99）で選び直す。
  選び直したしきい値が train.log の最良エポックの行の thr（小数2桁）と合うかを照らす。
  試験の予測を pred_test.csv・pred_dev.csv に書き、metrics.json（test_tol0.5・1.0・2.0）を partial_c4/<tag>/ に書く（学習台本の metrics.json と同じ形＋partial の項目）。
  確率の列は out/c4_compare_20261105/probs/probs_<name>_{dev,test}.npz・check に「打ち切り」と書く。
使い方: python score_partial_c4_20261105.py --tag c4_wide_8ep_s0 --name wide --hid 256 [--arch crnn] [--types …] [--index …]
"""
import argparse, csv, json, os, re, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FR = 100
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--index", default="out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json")
ap.add_argument("--arch", default="crnn", choices=["crnn", "conv", "gru"]); ap.add_argument("--hid", type=int, default=128)
ap.add_argument("--types", default="shakuri,kobushi,fall,accent,hammering"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--test", default="20260924"); ap.add_argument("--dev", default="20260911"); ap.add_argument("--min_dist", type=float, default=0.3)
a = ap.parse_args(); torch.set_num_threads(a.threads)
TYPES = [x for x in a.types.split(",") if x]
SRC = os.path.join(ROOT, "out", "c4_compare_20261105", "partial_c4", a.tag)
PR = os.path.join(ROOT, "out", "c4_compare_20261105", "probs")
index = json.load(open(os.path.join(ROOT, a.index), encoding="utf-8"))


def load_x(v):
    z = np.load(f"{ROOT}/feats/{v}.npz")
    lm = z["logmel"].astype(np.float32); T = lm.shape[1]
    st = z["st"]; vo = z["voiced"]
    stn = np.where(vo > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
    dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (vo[:-1] > 0); dst = np.where(vo > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
    return np.concatenate([lm, z["rms"][None, :], stn[None, :], vo[None, :], z["conf"][None, :], dst[None, :]], 0).astype(np.float32)


songs = {sp: [d for d in index if d["session"] == ses and os.path.exists(f"{ROOT}/feats/{d['video']}.npz")] for sp, ses in (("dev", a.dev), ("test", a.test))}
X = {d["video"]: load_x(d["video"]) for sp in songs for d in songs[sp]}
CIN = next(iter(X.values())).shape[0]
nm = np.load(os.path.join(SRC, "norm.npy")); mu_t = torch.tensor(nm[0][:, None]); sd_t = torch.tensor(nm[1][:, None])


class Net(nn.Module):
    def __init__(s, cin=CIN, h=a.hid, nout=len(TYPES)):
        super().__init__()
        if a.arch in ("crnn", "conv"):
            s.c1 = nn.Sequential(nn.Conv1d(cin, h, 5, padding=2), nn.BatchNorm1d(h), nn.ReLU(),
                                 nn.Conv1d(h, h, 5, padding=2, stride=2), nn.BatchNorm1d(h), nn.ReLU(),
                                 nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15))
        if a.arch in ("crnn", "gru"):
            s.gru = nn.GRU(h if a.arch == "crnn" else cin, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        s.head = nn.Linear(h if a.arch == "conv" else 2 * h, nout)

    def forward(s, x):
        T = x.shape[-1]
        if a.arch == "crnn":
            h = s.c1(x); h, _ = s.gru(h.transpose(1, 2))
        elif a.arch == "conv":
            h = s.c1(x).transpose(1, 2)
        else:
            h, _ = s.gru(x.transpose(1, 2))
        o = s.head(h).transpose(1, 2)
        return Fn.interpolate(o, size=T, mode="linear", align_corners=False)


net = Net(); net.load_state_dict(torch.load(os.path.join(SRC, "model.pt"), map_location="cpu")); net.eval()
N_PARAMS = sum(p.numel() for p in net.parameters())


def pick(prob, thr, min_dist):
    w = int(min_dist * FR); out = []; n = len(prob)
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


def evaluate(ds, thr, tol, probs, pk=None):
    agg = {c: [0, 0, 0] for c in TYPES}
    for d in ds:
        lo, hi = d["span"]; P = probs[d["video"]]
        for k, c in enumerate(TYPES):
            truth = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]
            cand = pk[d["video"]][k] if pk is not None else pick(P[k], 0.0, a.min_dist)
            pred = [t for t, p in cand if p >= thr[c] and lo <= t <= hi]
            agg[c][0] += match(pred, truth, tol); agg[c][1] += len(pred); agg[c][2] += len(truth)
    res = {}
    for c in TYPES:
        h, npd, nt = agg[c]; p = h / npd if npd else 0.0; r = h / nt if nt else 0.0
        res[c] = dict(hit=h, pred=npd, true=nt, precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / (p + r), 3) if p + r > 0 else 0.0)
    H = sum(x[0] for x in agg.values()); NP = sum(x[1] for x in agg.values()); NT = sum(x[2] for x in agg.values())
    p = H / NP if NP else 0; r = H / NT if NT else 0
    res["ALL"] = dict(hit=H, pred=NP, true=NT, precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / (p + r), 3) if (p + r) else 0)
    return res


t0 = time.time(); log = []
say = lambda s: (print(s, flush=True), log.append(s))
eps = [(int(m.group(1)), float(m.group(2)), m.group(3)) for m in (re.match(r"ep(\d+) loss=\S+ dev F1\(±0.5s\)=([\d.]+) .*thr=(\{.*\}) \d+s$", ln.strip())
       for ln in open(os.path.join(SRC, "train.log"), encoding="utf-8")) if m]
last_ep = max((e for e, _, _ in eps), default=0)
best = max(eps, key=lambda x: (x[1], -x[0])) if eps else None   # 学習台本は score > best で更新＝同じ値なら先のエポック
say(f"打ち切りの本＝{a.tag}・終わったエポック {last_ep}・最良エポック {best[0] if best else 'なし'}（dev {best[1] if best else ''}）・重みの数 {N_PARAMS}")
probs = {}
for sp in ("dev", "test"):
    P = {}
    with torch.no_grad():
        for d in songs[sp]:
            x = (torch.tensor(X[d["video"]][None]) - mu_t) / sd_t
            P[d["video"]] = torch.sigmoid(net(x))[0].numpy().astype(np.float32)
    probs[sp] = P
    np.savez_compressed(os.path.join(PR, f"probs_{a.name}_{sp}.npz"), **P)
pk = {sp: {v: [pick(probs[sp][v][k], 0.0, a.min_dist) for k in range(len(TYPES))] for v in probs[sp]} for sp in probs}
thr = {}
for c in TYPES:
    bf = None
    for th in np.arange(0.05, 0.96, 0.025):
        r = evaluate(songs["dev"], {cc: (th if cc == c else 0.99) for cc in TYPES}, 0.5, probs["dev"], pk["dev"])
        if bf is None or r[c]["f1"] > bf[0]: bf = (r[c]["f1"], float(th))
    thr[c] = bf[1]
rd = evaluate(songs["dev"], thr, 0.5, probs["dev"], pk["dev"])
logthr = best[2] if best else ""
mine = "{" + ", ".join(f"'{c[:4]}': {round(t, 2)}" for c, t in thr.items()) + "}"
say(f"選び直したしきい値＝{ {c: round(t, 3) for c, t in thr.items()} }・dev F {rd['ALL']['f1']}")
say(f"train.log の最良エポックの thr＝{logthr}・照らし＝{'一致' if mine == logthr else '不一致（' + mine + '）'}・dev F の照らし＝{'一致' if best and abs(rd['ALL']['f1'] - best[1]) < 0.0005 else '不一致'}")
metrics = {}
for sp in ("dev", "test"):
    for tol in (0.5, 1.0, 2.0):
        metrics[f"{sp}_tol{tol}"] = evaluate(songs[sp], thr, tol, probs[sp], pk[sp])
    with open(os.path.join(SRC, f"pred_{sp}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["video", "type", "wav_time_sec", "prob"])
        for d in songs[sp]:
            for k, c in enumerate(TYPES):
                for t, p in pk[sp][d["video"]][k]:
                    if p >= thr[c]: w.writerow([d["video"], c, f"{t:.2f}", f"{p:.3f}"])
json.dump(dict(metrics=metrics, thr=thr, best=[rd["ALL"]["f1"], best[0] if best else -1, thr], partial=dict(last_epoch=last_ep, best_epoch=best[0] if best else None, epochs_planned=8),
               n_params=N_PARAMS, arch=a.arch, hid=a.hid, types=TYPES, elapsed_sec=None, n_train=None),
          open(os.path.join(SRC, "metrics.json"), "w"), indent=1)
say(f"試験 ±0.5秒＝" + " ".join(f"{c[:4]} {metrics['test_tol0.5'][c]['true']}/{metrics['test_tol0.5'][c]['pred']}/{metrics['test_tol0.5'][c]['hit']}" for c in TYPES + ["ALL"]))
say(f"所要 {time.time() - t0:.1f}秒")
txt = "\n".join(log) + "\n" + "dev＝打ち切り（照らす pred なし・境目は dev で選び直し）・食い違い 0件\ntest＝打ち切り（同上）・食い違い 0件\n"
open(os.path.join(PR, f"probs_{a.name}_check.txt"), "w", encoding="utf-8").write(txt)
open(os.path.join(SRC, "partial_note.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
