# -*- coding: utf-8 -*-
"""C2（入力を変える比較）用の学習台本＝現行の入力と「現行＋お手本との音程差」を同じ条件で比べる（2026-10-23）。

もとは train_crnn.py（2026-09-28）で、違いは入力の列を足す選択肢（--refdiff）を付けたことだけです。
--refdiff を付けないときの入力・学習・評価は、もとの台本とまったく同じ順序で動きます（同じ乱数の種・同じ分割・同じしきい値の決め方）。
--refdiff を付けると、各フレームに次の2列が最後に加わります。
  ① お手本との音程差（セント）÷ 1200 ＝ 1オクターブを 1 とした値。お手本が読めていないフレームは 0。
  ② 欠損の印（お手本があり、かつ声が出ているフレームだけ 1・それ以外は 0）。
お手本の読取がまだ無い曲は、その2列が全フレーム 0 と 0 になります（学習からその曲を外すことはしません）。

使い方:
  python train_crnn_refdiff_20261023.py --tag c2a_base    --epochs 4
  python train_crnn_refdiff_20261023.py --tag c2b_refdiff --epochs 4 --refdiff
  python train_crnn_refdiff_20261023.py --tag c2b_refdiff_cov30 --epochs 8 --refdiff --min_ref_frac 0.30
出力: out/<tag>/model.pt, report.md, pred_<split>.csv, persong_<split>.csv, metrics.json, ref_frac.csv

--min_ref_frac（2026-10-24 追加）。
お手本付きとして学習に使う曲を「被覆がこの値以上」の曲だけに絞ります（既定 0 ＝従来どおり全部使う）。
被覆＝お手本の差の列が値を持つフレーム割合（build_refdiff_feats が書く feats_refdiff/_coverage.csv の both_frac）。
下回った曲は学習から外すのではなく、その曲の「お手本との音程差」と「欠損の印」を全フレーム 0 にします（曲そのものは学習に残る）。
dev・test の曲は絞りません（しきい値の決め方と未見の試験を v4 と同じに保つため）。分割の規則も従来どおりです。
"""
import os, sys, json, time, argparse, random, csv
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="c2a_base"); ap.add_argument("--test", default="20260924"); ap.add_argument("--dev", default="20260911")
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--sigma", type=float, default=10); ap.add_argument("--chunk", type=int, default=800); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--hid", type=int, default=128); ap.add_argument("--lagshift", type=float, default=0.0, help="ラベルを何秒前へずらすか（技法共通）")
ap.add_argument("--min_dist", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--no_mel", action="store_true", help="log-melを使わずピッチ系＋RMSだけ")
ap.add_argument("--eval_only", action="store_true", help="学習せず out/<tag>/model.pt を評価")
ap.add_argument("--index", default="index.json", help="使うindexファイル名（index_snapped.json など）")
ap.add_argument("--exclude", default="", help="学習から外すセッション（カンマ区切り）")
ap.add_argument("--pw_cap", type=float, default=30.0, help="pos_weightの上限")
ap.add_argument("--thr_mode", default="f1", choices=["f1", "count"], help="devでのしきい値決定：F1最大 or 件数差最小")
ap.add_argument("--refdiff", action="store_true", help="入力に「お手本との音程差」と「欠損の印」の2列を足す（2026-10-23 追加）")
ap.add_argument("--only_with_ref", action="store_true", help="お手本が読めている曲だけを使う（2026-10-23 追加・比較の土俵をそろえたい時）")
ap.add_argument("--min_ref_frac", type=float, default=0.0,
                help="お手本付きとして学習に使う曲の被覆の下限（0〜1・既定0＝従来どおり全部使う）。"
                     "被覆＝お手本の差の列が値を持つフレーム割合。これを下回る曲は学習側だけ「お手本なし」として扱う（dev・test と分割は変えない）。")
a = ap.parse_args(); torch.set_num_threads(a.threads); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
OUT = f"{ROOT}/out/{a.tag}"; os.makedirs(OUT, exist_ok=True)
index = json.load(open(f"{ROOT}/{a.index}", encoding="utf-8"))
FR = 100  # frames/sec
REFD = f"{ROOT}/feats_refdiff"
EXC = set(x for x in a.exclude.split(",") if x)


def has_ref(v):
    return os.path.exists(f"{REFD}/{v}.npz")


def is_train_song(d):
    """学習に回る曲か（test・dev・--exclude のどれでもない）。分割の規則は従来どおり。"""
    return d["session"] not in (a.test, a.dev) and d["session"] not in EXC


# 曲ごとの被覆。build_refdiff_feats_20261023.py の記録（_coverage.csv の both_frac
# ＝お手本があり、かつ声が出ているフレームの割合）を正本とし、記録に無い曲は特徴量から直接数える。
_COV = {}
_COVCSV = f"{REFD}/_coverage.csv"
if os.path.exists(_COVCSV):
    with open(_COVCSV, encoding="utf-8-sig") as _f:
        for _row in csv.DictReader(_f):
            try:
                _COV[_row["video"]] = float(_row["both_frac"])
            except (KeyError, TypeError, ValueError):
                pass


def ref_frac(v):
    """この曲の被覆（0〜1）。お手本の読取が無い曲は 0。"""
    if v in _COV:
        return _COV[v]
    if not has_ref(v):
        _COV[v] = 0.0
    else:
        z = np.load(f"{REFD}/{v}.npz")
        _COV[v] = float(z["refvalid"].mean()) if "refvalid" in z.files else 0.0
    return _COV[v]


def use_ref(d):
    """この曲で「お手本との音程差」の2列を実際に使うか。
    --min_ref_frac を指定したときは、被覆がそれを下回る学習側の曲を「お手本なし」（2列が全部 0）として扱う。"""
    v = d["video"]
    if not (a.refdiff and has_ref(v)):
        return False
    if a.min_ref_frac > 0 and is_train_song(d) and ref_frac(v) < a.min_ref_frac:
        return False
    return True


def load(d):
    v = d["video"]
    # --refdiff のときは列を足した別名ファイル（feats_refdiff）を優先して読む。無ければ従来のファイル＋全欠損の2列。
    src = f"{REFD}/{v}.npz" if use_ref(d) else f"{ROOT}/feats/{v}.npz"
    z = np.load(src)
    lm = z["logmel"].astype(np.float32); T = lm.shape[1]
    st = z["st"]; vo = z["voiced"]
    stn = np.where(vo > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
    dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (vo[:-1] > 0); dst = np.where(vo > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
    parts = [] if a.no_mel else [lm]
    parts += [z["rms"][None, :], stn[None, :], vo[None, :], z["conf"][None, :], dst[None, :]]
    if a.refdiff:
        if "refdiff_cent" in z.files:
            rd = (z["refdiff_cent"] / 1200.0).astype(np.float32); rv = z["refvalid"].astype(np.float32)
        else:
            rd = np.zeros(T, np.float32); rv = np.zeros(T, np.float32)
        parts += [rd[None, :], rv[None, :]]
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
    if a.only_with_ref and not has_ref(d["video"]): continue
    data[d["video"]] = (d,) + load(d)
CIN = next(iter(data.values()))[1].shape[0]
train = [v for v, (d, *_) in data.items() if is_train_song(d)]
dev = [v for v, (d, *_) in data.items() if d["session"] == a.dev]
test = [v for v, (d, *_) in data.items() if d["session"] == a.test]
nref = dict(train=sum(has_ref(v) for v in train), dev=sum(has_ref(v) for v in dev), test=sum(has_ref(v) for v in test))
nused = dict(train=sum(use_ref(data[v][0]) for v in train), dev=sum(use_ref(data[v][0]) for v in dev), test=sum(use_ref(data[v][0]) for v in test))
# 曲ごとの被覆と足切りの結果を記録に残す（--min_ref_frac の検証用）。
with open(f"{OUT}/ref_frac.csv", "w", newline="", encoding="utf-8-sig") as _f:
    _w = csv.writer(_f); _w.writerow(["video", "session", "split", "has_ref", "ref_frac", "used_as_ref"])
    for _v, (_d, *_rest) in data.items():
        _sp = "test" if _d["session"] == a.test else ("dev" if _d["session"] == a.dev else ("excluded" if _d["session"] in EXC else "train"))
        _w.writerow([_v, _d["session"], _sp, int(has_ref(_v)), f"{ref_frac(_v):.3f}", int(use_ref(_d))])
print(f"train {len(train)} dev {len(dev)} test {len(test)} cin {CIN} refdiff={a.refdiff} min_ref_frac={a.min_ref_frac} お手本ありの曲数 {nref} お手本として使った曲数 {nused}", flush=True)
if a.min_ref_frac > 0:
    _drop = {}
    for _v in train:
        _d = data[_v][0]
        if has_ref(_v) and not use_ref(_d): _drop[_d["session"]] = _drop.get(_d["session"], 0) + 1
    print(f"被覆 {a.min_ref_frac} 未満で学習のお手本から外した曲（セッション別）＝{_drop}（計 {sum(_drop.values())}曲）", flush=True)
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
rep = [f"# C2 比較 {a.tag}（{time.strftime('%Y-%m-%d %H:%M')}）", "",
       "学習＝" + ",".join(sorted({data[v][0]['session'] for v in train})) + f"／dev（しきい値決定）＝{a.dev}／test（未見）＝{a.test}",
       f"epochs={a.epochs} sigma={a.sigma}f lagshift={a.lagshift}s chunk={a.chunk} bs={a.bs} hid={a.hid} seed={a.seed} no_mel={a.no_mel} refdiff={a.refdiff} only_with_ref={a.only_with_ref} min_ref_frac={a.min_ref_frac} cin={CIN} best_ep={best[1]}",
       f"曲数＝学習{len(train)}／dev{len(dev)}／test{len(test)}・うちお手本ありの曲数＝{nref}・うちお手本として使った曲数＝{nused}", f"しきい値={thr}", ""]
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
json.dump(dict(metrics=metrics, thr=thr, best=best, cin=CIN, refdiff=bool(a.refdiff), only_with_ref=bool(a.only_with_ref),
               min_ref_frac=float(a.min_ref_frac), n_ref_used=nused,
               n_train=len(train), n_dev=len(dev), n_test=len(test), n_ref=nref, epochs=a.epochs, seed=a.seed, bs=a.bs,
               elapsed_sec=round(time.time() - t0, 1)), open(f"{OUT}/metrics.json", "w"), indent=1)
print("\n".join(rep)); print("DONE", OUT)
