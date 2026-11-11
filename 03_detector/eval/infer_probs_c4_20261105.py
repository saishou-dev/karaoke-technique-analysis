# -*- coding: utf-8 -*-
r"""学習済みのモデル（model.pt・norm.npy）から dev と試験の曲の確率の列を書き出す（2026-11-05 Opus・c4-compare・学習しない）。
網の形・入力の列・正規化は train_crnn_refdiff_c3_5tech_20261105.py と同じ（写し）。--refdiff なし・5技法。
出力＝<out>/probs_<name>_dev.npz・probs_<name>_test.npz（曲名→ (5, T) の float32）＋ <out>/probs_<name>_check.txt。
自己確認＝保存済みのしきい値（metrics.json の thr）で同じ pick を通し、モデルの pred_dev.csv・pred_test.csv と1件ずつ照らす。
使い方: python infer_probs_c4_20261105.py --mdir out/c3_5tech_global_20261105/c3_5tech_8ep_s0 --index out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json --name base --out out/c4_compare_20261105/probs
"""
import argparse, csv, json, os, sys, time, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TYPES = ["shakuri", "kobushi", "fall", "accent", "hammering"]
FR = 100
ap = argparse.ArgumentParser()
ap.add_argument("--mdir", required=True); ap.add_argument("--index", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--hid", type=int, default=128)
ap.add_argument("--test", default="20260924"); ap.add_argument("--dev", default="20260911"); ap.add_argument("--min_dist", type=float, default=0.3)
a = ap.parse_args(); torch.set_num_threads(a.threads)
MDIR = os.path.join(ROOT, a.mdir); OUT = os.path.join(ROOT, a.out); os.makedirs(OUT, exist_ok=True)
index = json.load(open(os.path.join(ROOT, a.index), encoding="utf-8"))


def load_x(v):
    z = np.load(f"{ROOT}/feats/{v}.npz")
    lm = z["logmel"].astype(np.float32); T = lm.shape[1]
    st = z["st"]; vo = z["voiced"]
    stn = np.where(vo > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
    dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (vo[:-1] > 0); dst = np.where(vo > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
    parts = [lm, z["rms"][None, :], stn[None, :], vo[None, :], z["conf"][None, :], dst[None, :]]
    return np.concatenate(parts, 0).astype(np.float32)


songs = {"dev": [d for d in index if d["session"] == a.dev and os.path.exists(f"{ROOT}/feats/{d['video']}.npz")],
         "test": [d for d in index if d["session"] == a.test and os.path.exists(f"{ROOT}/feats/{d['video']}.npz")]}
X = {d["video"]: load_x(d["video"]) for sp in songs for d in songs[sp]}
CIN = next(iter(X.values())).shape[0]
nm = np.load(os.path.join(MDIR, "norm.npy")); mu_t = torch.tensor(nm[0][:, None]); sd_t = torch.tensor(nm[1][:, None])


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


net = Net(); net.load_state_dict(torch.load(os.path.join(MDIR, "model.pt"), map_location="cpu")); net.eval()


def pick(prob, thr, min_dist):
    w = int(min_dist * FR); out = []; n = len(prob)
    for i in range(1, n - 1):
        if prob[i] >= thr and prob[i] >= prob[max(0, i - w):min(n, i + w + 1)].max(): out.append((i / FR, float(prob[i])))
    return out


t0 = time.time(); log = []
say = lambda s: (print(s, flush=True), log.append(s))
thr = json.load(open(os.path.join(MDIR, "metrics.json"), encoding="utf-8"))["thr"]
say(f"モデル＝{MDIR}／正解の一覧（曲の選び出しだけに使う）＝{a.index}／cin {CIN}／しきい値 {thr}")
for sp in ("dev", "test"):
    P = {}
    with torch.no_grad():
        for d in songs[sp]:
            x = (torch.tensor(X[d["video"]][None]) - mu_t) / sd_t
            P[d["video"]] = torch.sigmoid(net(x))[0].numpy().astype(np.float32)
    np.savez_compressed(os.path.join(OUT, f"probs_{a.name}_{sp}.npz"), **P)
    # 自己確認＝pred_<sp>.csv（学習台本が書いた予測）と1件ずつ
    ref = collections.Counter()
    for r in csv.DictReader(open(os.path.join(MDIR, f"pred_{sp}.csv"), encoding="utf-8-sig")):
        ref[(r["video"], r["type"], r["wav_time_sec"])] += 1
    mine = collections.Counter()
    for d in songs[sp]:
        for k, c in enumerate(TYPES):
            for t, _ in pick(P[d["video"]][k], thr[c], a.min_dist):
                mine[(d["video"], c, f"{t:.2f}")] += 1
    diff = (ref - mine) + (mine - ref)
    say(f"{sp}＝{len(P)}曲・予測 {sum(mine.values())}件（pred_{sp}.csv {sum(ref.values())}件）・食い違い {sum(diff.values())}件")
    if diff:
        say("食い違いの例＝" + str(list(diff.items())[:10]))
say(f"所要 {time.time() - t0:.1f}秒")
open(os.path.join(OUT, f"probs_{a.name}_check.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
