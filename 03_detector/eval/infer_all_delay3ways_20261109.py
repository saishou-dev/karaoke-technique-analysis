# -*- coding: utf-8 -*-
r"""学習済みのモデルから全208曲（学習150・調整30・試験28）の確率の列を書き出す（2026-11-09 Opus・delay-3ways・学習しない）。
infer_probs2_c4_20261105.py の写し（crnn の形だけ）。違い＝曲を分けずに全部・1つの npz に書く・dev と試験は pred_*.csv と1件ずつ照らす。
用途＝案B の候補（ii）（今のモデル種0の確率の山）と、案D の1回目の置き直し（基準種0 の学習曲の確率の山）。
出力＝<out>/probs_<name>_all.npz（曲名→ (5, T) の float32・技法の順＝TYPES）＋ probs_<name>_all_check.txt
使い方: python infer_all_delay3ways_20261109.py --mdir out/c3_5tech_global_20261105/c3_5tech_8ep_s0 --name base --out out/delay_3ways_20261109/probs [--threads 2]
"""
import argparse, collections, csv, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FR = 100
TYPES = ["shakuri", "kobushi", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser()
ap.add_argument("--mdir", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--index", default="out/c3_5tech_global_20261105/labels/index_c3_5tech_20261105.json", help="曲の選び出しだけに使う")
ap.add_argument("--hid", type=int, default=128); ap.add_argument("--threads", type=int, default=2)
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
    return np.concatenate([lm, z["rms"][None, :], stn[None, :], vo[None, :], z["conf"][None, :], dst[None, :]], 0).astype(np.float32)


songs = [d for d in index if os.path.exists(f"{ROOT}/feats/{d['video']}.npz")]
nm = np.load(os.path.join(MDIR, "norm.npy")); mu_t = torch.tensor(nm[0][:, None]); sd_t = torch.tensor(nm[1][:, None])
CIN = nm.shape[1]


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
say(f"モデル＝{MDIR}・cin {CIN}・しきい値 {thr}・スレッド {a.threads}・曲 {len(songs)}")
P = {}
with torch.no_grad():
    for d in songs:
        x = (torch.tensor(load_x(d["video"])[None]) - mu_t) / sd_t
        P[d["video"]] = torch.sigmoid(net(x))[0].numpy().astype(np.float32)
np.savez_compressed(os.path.join(OUT, f"probs_{a.name}_all.npz"), **P)
for sp, ses in (("dev", a.dev), ("test", a.test)):
    ref = collections.Counter()
    for r in csv.DictReader(open(os.path.join(MDIR, f"pred_{sp}.csv"), encoding="utf-8-sig")):
        ref[(r["video"], r["type"], r["wav_time_sec"])] += 1
    mine = collections.Counter()
    for d in songs:
        if d["session"] != ses: continue
        for k, c in enumerate(TYPES):
            for t, _ in pick(P[d["video"]][k], thr[c], a.min_dist):
                mine[(d["video"], c, f"{t:.2f}")] += 1
    diff = (ref - mine) + (mine - ref)
    say(f"{sp}＝予測 {sum(mine.values())}件（pred_{sp}.csv {sum(ref.values())}件）・食い違い {sum(diff.values())}件")
say(f"全 {len(P)}曲・所要 {time.time() - t0:.1f}秒")
open(os.path.join(OUT, f"probs_{a.name}_all_check.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
