# -*- coding: utf-8 -*-
r"""合否付き台帳 2種を生成する（2026-10-08）。新しい学習はしない。
A. 検出照合台帳（予測 vs 正解・イベント単位）＝ test 9/24（28曲）を render_pitchbar.py と同じ
   学習済みCRNNアンサンブル（v2d+v2d2+v2e・確率平均・lag 0.3）＋同じしきい値キャッシュ（out/pitchbar_thr_*.json・dev 9/11）で予測し、
   render_pitchbar.py / rethreshold_raw.py と同じ貪欲1対1照合（±tol 1.0s）で TP／FP／FN を1行ずつ書き出す。
   出力: 検出照合台帳_test20260924.csv・検出照合_集計_test20260924.csv・検出照合_集計_test20260924.md
B. 検収台帳（本人の目視○×・正解G1イベント単位）＝ 動画解析/kensyu_*.csv（PC版）＋ ledgers_20261008/kensyu_mobile_<sess>_db.json（スマホ検収＝Artifact db kensyu/<sess> の書き出し・
   {"j":{"<event_id>":{"v":"o|x|k","t":<ms epoch>}}}・o→OK／x→NG／k→HOLD・source="mobile_<sess>"・10/8追加）を統合。重複は新しい判定を優先。
   時刻は dataset/G1_events_all.csv から (video, counter, count_value) で結合。
   出力: 検収台帳_統合.csv・検収_集計.md
使い方: python build_ledgers_20261008.py [--tags v2d,v2d2,v2e --lag 0.3 --tol 1.0 --test 20260924 --dev 20260911] [--skip_a] [--skip_b]"""
import os, sys, re, json, csv, glob, time, argparse, warnings, collections
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
T_START = time.time()
BASE = "G:/研究アーカイブ/成果物/動画解析"; ROOT = f"{BASE}/g3v2_20260928"; FR = 100
OUT = f"{BASE}/dataset/ledgers_20261008"; os.makedirs(OUT, exist_ok=True)
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JA = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング"}
ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="v2d,v2d2,v2e"); ap.add_argument("--lag", type=float, default=0.3); ap.add_argument("--tol", type=float, default=1.0)
ap.add_argument("--test", default="20260924"); ap.add_argument("--dev", default="20260911"); ap.add_argument("--hid", type=int, default=128); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--skip_a", action="store_true"); ap.add_argument("--skip_b", action="store_true")
a = ap.parse_args(); TAGS = a.tags.split(",")

def wcsv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

# =====================================================================================
# A. 検出照合台帳
# =====================================================================================
def build_a():
    import torch, torch.nn as nn, torch.nn.functional as Fn
    torch.set_num_threads(a.threads)
    index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
    test = [d for d in index if d["session"] == a.test]
    print(f"[A] test={a.test} {len(test)}曲  tags={TAGS} lag={a.lag} tol=±{a.tol}s")
    def feats_of(z):
        lm = z["logmel"].astype(np.float32); T = lm.shape[1]; st = z["st"]; v = z["voiced"]
        stn = np.where(v > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
        dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (v[:-1] > 0); dst = np.where(v > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
        return np.concatenate([lm, z["rms"][None, :], stn[None, :], v[None, :], z["conf"][None, :], dst[None, :]], 0).astype(np.float32)
    norm0 = np.load(f"{ROOT}/out/{TAGS[0]}/norm.npy"); CIN = norm0.shape[1]
    class Net(nn.Module):  # render_pitchbar.py / rethreshold_raw.py と同一構造
        def __init__(s, cin=CIN, h=a.hid, nout=len(TYPES)):
            super().__init__()
            s.c1 = nn.Sequential(nn.Conv1d(cin, h, 5, padding=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Conv1d(h, h, 5, padding=2, stride=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15))
            s.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15); s.head = nn.Linear(2 * h, nout)
        def forward(s, x):
            T = x.shape[-1]; h = s.c1(x); h, _ = s.gru(h.transpose(1, 2)); o = s.head(h).transpose(1, 2)
            return Fn.interpolate(o, size=T, mode="linear", align_corners=False)
    nets = []
    for tg in TAGS:
        n_ = Net(); n_.load_state_dict(torch.load(f"{ROOT}/out/{tg}/model.pt")); n_.eval(); nm = np.load(f"{ROOT}/out/{tg}/norm.npy")
        nets.append((n_, torch.tensor(nm[0][:, None]), torch.tensor(nm[1][:, None])))
    def probs_of(z):
        x = torch.tensor(feats_of(z)[None])
        with torch.no_grad(): return np.mean([torch.sigmoid(n_((x - m_) / s_))[0].numpy() for n_, m_, s_ in nets], 0)
    def pick(prob, thr, md):
        w = int(md * FR); n = len(prob)
        return [(i / FR + a.lag, float(prob[i])) for i in range(1, n - 1) if prob[i] >= thr and prob[i] >= prob[max(0, i - w):min(n, i + w + 1)].max()]
    def match(pred, truth, tol):
        """貪欲1対1照合（render_pitchbar.py と同じ）：予測を時刻順に、未使用の正解のうち最も近いもの（±tol）を取る"""
        used = set(); pairs = []
        for pi in sorted(range(len(pred)), key=lambda i: pred[i]):
            best = None
            for k, t in enumerate(truth):
                if k in used or abs(t - pred[pi]) > tol: continue
                if best is None or abs(t - pred[pi]) < abs(truth[best] - pred[pi]): best = k
            if best is not None: used.add(best); pairs.append((pi, best))
        return pairs
    # しきい値（render_pitchbar.py のキャッシュを再利用。無ければ同じ手順で dev から決めてキャッシュ）
    thr_file = f"{ROOT}/out/pitchbar_thr_{'+'.join(TAGS)}_lag{a.lag}.json"
    if os.path.exists(thr_file):
        best = {c: tuple(v) for c, v in json.load(open(thr_file, encoding="utf-8")).items()}; print(f"[A] しきい値キャッシュ: {thr_file}")
    else:
        print(f"[A] しきい値を dev({a.dev}) で決定中…"); dev = [d for d in index if d["session"] == a.dev and os.path.exists(f"{ROOT}/feats/{d['video']}.npz")]
        P = {d["video"]: probs_of(np.load(f"{ROOT}/feats/{d['video']}.npz")) for d in dev}
        def f1_dev(c, th, md):
            k = TYPES.index(c); H = NP = NT = 0
            for d in dev:
                lo, hi = d["span"]; tr = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]; pr = [t for t, _ in pick(P[d["video"]][k], th, md) if lo <= t <= hi]
                H += len(match(pr, tr, 1.0)); NP += len(pr); NT += len(tr)
            return prf(H, NP - H, NT - H)[2]
        best = {}
        for c in TYPES:
            cand = max(((f1_dev(c, th, md), th, md) for th in np.arange(0.2, 0.96, 0.05) for md in (0.3, 0.5, 0.8)), key=lambda x: x[0]); best[c] = (round(float(cand[1]), 2), cand[2])
        json.dump(best, open(thr_file, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[A] しきい値(thr, 最短間隔s)=", best)
    rows = []; agg = collections.OrderedDict()
    for d in sorted(test, key=lambda d: d["video"]):
        video = d["video"]; song = video.rsplit("_", 2)[0]; lo, hi = d["span"]; off = float(d.get("offset", 0.0))
        PR = probs_of(np.load(f"{ROOT}/feats/{video}.npz"))
        for c in TYPES:
            pred = [(t, p) for t, p in pick(PR[TYPES.index(c)], *best[c]) if lo <= t <= hi]
            truth = [t for cc, t in d["events"] if cc == c and lo <= t <= hi]
            pt = [t for t, _ in pred]; pairs = match(pt, truth, a.tol); mp = {i for i, _ in pairs}; mt = {k for _, k in pairs}
            ev = []
            for i, k in pairs: ev.append(dict(kind="TP", pred_time_sec=pt[i], truth_time_sec=truth[k], diff_sec=pt[i] - truth[k], score=pred[i][1]))
            for i in range(len(pt)):
                if i not in mp: ev.append(dict(kind="FP", pred_time_sec=pt[i], truth_time_sec="", diff_sec="", score=pred[i][1]))
            for k in range(len(truth)):
                if k not in mt: ev.append(dict(kind="FN", pred_time_sec="", truth_time_sec=truth[k], diff_sec="", score=""))
            ev.sort(key=lambda e: e["pred_time_sec"] if e["pred_time_sec"] != "" else e["truth_time_sec"])
            for e in ev:
                tv = e["truth_time_sec"]
                rows.append(dict(session=a.test, song=song, video=video, technique=c, technique_ja=JA[c], **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()},
                                 truth_video_time_sec=(round(tv - off, 3) if tv != "" else "")))
            agg[(song, video, c)] = dict(tp=len(pairs), fp=len(pt) - len(pairs), fn=len(truth) - len(pairs), n_pred=len(pt), n_truth=len(truth))
        g = [x for (_, vv, _), x in agg.items() if vv == video]
        print(f"  {video:<40} 一致 {sum(x['tp'] for x in g):>3}／誤検知 {sum(x['fp'] for x in g):>3}／見逃し {sum(x['fn'] for x in g):>3}")
    cols = ["session", "song", "video", "technique", "technique_ja", "kind", "pred_time_sec", "truth_time_sec", "diff_sec", "score", "truth_video_time_sec"]
    wcsv(f"{OUT}/検出照合台帳_test{a.test}.csv", rows, cols)
    # 集計（曲×技法・曲ALL・技法ALL・全体）
    def srow(song, video, c, tp, fp, fn):
        p, r, f = prf(tp, fp, fn)
        return dict(song=song, video=video, technique=c, technique_ja=JA.get(c, "全体"), n_truth=tp + fn, n_pred=tp + fp, TP=tp, FP=fp, FN=fn, precision=round(p, 3), recall=round(r, 3), f1=round(f, 3))
    srows = [srow(s, v, c, g["tp"], g["fp"], g["fn"]) for (s, v, c), g in agg.items()]
    vids = sorted({(s, v) for s, v, _ in agg})
    def vid_tot(v):
        g = [x for (_, vv, _), x in agg.items() if vv == v]; return (sum(x["tp"] for x in g), sum(x["fp"] for x in g), sum(x["fn"] for x in g))
    for s, v in vids: srows.append(srow(s, v, "ALL", *vid_tot(v)))
    tech_tot = {}
    for c in TYPES:
        g = [x for (_, _, cc), x in agg.items() if cc == c]; tech_tot[c] = (sum(x["tp"] for x in g), sum(x["fp"] for x in g), sum(x["fn"] for x in g)); srows.append(srow("ALL", "ALL", c, *tech_tot[c]))
    TT = tuple(sum(t[i] for t in tech_tot.values()) for i in range(3)); srows.append(srow("ALL", "ALL", "ALL", *TT))
    scols = ["song", "video", "technique", "technique_ja", "n_truth", "n_pred", "TP", "FP", "FN", "precision", "recall", "f1"]
    wcsv(f"{OUT}/検出照合_集計_test{a.test}.csv", srows, scols)
    # md
    L = [f"# 検出照合 集計（test {a.test}・{len(test)}曲）", "",
         f"- 予測＝学習済みCRNNアンサンブル {'+'.join(TAGS)}（確率平均・lag +{a.lag}s）。しきい値・最短間隔＝dev {a.dev} で技法別に F1(±1s) 最大（{os.path.basename(thr_file)}）。新しい学習なし。",
         f"- 照合＝予測と正解（G1・録音WAV時刻）の ±{a.tol}s 一致・貪欲1対1（render_pitchbar.py と同規則）。採点区間（span）内のみ。",
         "- TP＝一致（予測が正解と一致）／FP＝誤検知（予測だけ）／FN＝見逃し（正解だけ）。適合率=TP/(TP+FP)・再現率=TP/(TP+FN)。",
         f"- しきい値(thr, 最短間隔s)＝{ {JA[c]: best[c] for c in TYPES} }", "",
         "## 技法別（28曲合計）", "", "| 技法 | 正解 | 予測 | TP | FP | FN | 適合率 | 再現率 | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for c in TYPES:
        tp, fp, fn = tech_tot[c]; p, r, f = prf(tp, fp, fn); L.append(f"| {JA[c]} | {tp + fn} | {tp + fp} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {f:.3f} |")
    p, r, f = prf(*TT); L.append(f"| **全体** | {TT[0] + TT[2]} | {TT[0] + TT[1]} | {TT[0]} | {TT[1]} | {TT[2]} | {p:.3f} | {r:.3f} | {f:.3f} |")
    L += ["", "## 曲別（6技法合計）", "", "| 曲 | 正解 | 予測 | TP | FP | FN | 適合率 | 再現率 | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for s, v in vids:
        tp, fp, fn = vid_tot(v); p, r, f = prf(tp, fp, fn)
        L.append(f"| {s} | {tp + fn} | {tp + fp} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {f:.3f} |")
    L += ["", f"- 台帳本体＝検出照合台帳_test{a.test}.csv（{len(rows)}行＝TP {TT[0]}＋FP {TT[1]}＋FN {TT[2]}）。列＝session, song, video, technique, technique_ja, kind, pred_time_sec, truth_time_sec, diff_sec(pred−truth), score(予測確率), truth_video_time_sec(録画時刻＝G1台帳の video_time_sec と対応)。",
          "- 時刻の単位＝秒。pred/truth は録音WAVの時刻（index.json の events＝audio_time_sec）。録画の時刻は truth_video_time_sec（= truth − offset）。",
          f"- 生成＝build_ledgers_20261008.py（{time.strftime('%Y-%m-%d %H:%M')}）"]
    open(f"{OUT}/検出照合_集計_test{a.test}.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    # lulu 一致確認（#14 の図＝一致24／誤検知119／見逃し27）
    lulu = [v for s, v in vids if v.lower().startswith("lulu")]
    if lulu:
        chk = vid_tot(lulu[0])
        print(f"[A] lulu 確認: 一致 {chk[0]}／誤検知 {chk[1]}／見逃し {chk[2]}  → #14 の図(24/119/27) と", "一致" if chk == (24, 119, 27) else "不一致！")
    print(f"[A] 台帳 {len(rows)}行  TP {TT[0]} / FP {TT[1]} / FN {TT[2]}  技法別F1=", {JA[c]: round(prf(*tech_tot[c])[2], 3) for c in TYPES}, f"全体F1={prf(*TT)[2]:.3f}")
    return rows

# =====================================================================================
# B. 検収台帳（PC版 kensyu_*.csv の統合）
# =====================================================================================
def parse_kensyu(path):
    """event_id,video,counter,count_value,judgement。曲名にカンマがあり引用符なしの行（Love me,Love you）も復元する。"""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        lines = [l.rstrip("\r\n") for l in f if l.strip()]
    for l in lines[1:]:
        parts = l.split(",")
        if len(parts) == 5: eid, video, counter, cv, jd = parts
        else:
            jd, cv, counter = parts[-1], parts[-2], parts[-3]; rest = ",".join(parts[:-3]); suf = f"_{counter}_{cv}"
            L = (len(rest) - len(suf) - 1) // 2; video = rest[-L:]; eid = rest[:len(rest) - L - 1]
            if eid != video + suf: print(f"  [B] 復元不可の行をスキップ: {os.path.basename(path)}: {l}"); continue
        if jd not in ("OK", "NG", "HOLD", "EXCLUDED"): print(f"  [B] 判定値が不正の行をスキップ: {os.path.basename(path)}: {l}"); continue
        rows.append(dict(event_id=eid, video=video, counter=counter, count_value=cv, judgement=jd))
    return rows
def parse_mobile(path):
    """スマホ検収の db 書き出し（kensyu_mobile_<sess>_db.json）→ [(event_id, video, counter, count_value, judgement, stamp YYYYMMDDHHMM)]。
       event_id＝<video>_<counter>_<count_value>（build_mobile_review.py の ev_min.id＝review_sample_<sess>_200.html の EV.id と同じ）。t＝ms epoch（UTC）→ JST。"""
    import datetime
    d = json.load(open(path, encoding="utf-8")); d = d.get("data", d); J = d.get("j", d); V = {"o": "OK", "x": "NG", "k": "HOLD"}
    rows = []
    for eid, v in J.items():
        m = re.match(r"^(.*)_(shakuri|kobushi|vibrato|fall|accent|hammering)_(\d+)$", eid)
        if not m: print(f"  [B] event_id を分解できない行をスキップ: {os.path.basename(path)}: {eid}"); continue
        val = v.get("v") if isinstance(v, dict) else v; t = v.get("t") if isinstance(v, dict) else None
        if val not in V: print(f"  [B] 判定値が不正の行をスキップ: {os.path.basename(path)}: {eid}={val}"); continue
        stamp = (datetime.datetime.fromtimestamp(t / 1000, datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y%m%d%H%M") if t else "")
        rows.append(dict(event_id=eid, video=m.group(1), counter=m.group(2), count_value=m.group(3), judgement=V[val], _stamp=stamp))
    return rows
def build_b():
    files = sorted(glob.glob(f"{BASE}/kensyu_*.csv")); mfiles = sorted(glob.glob(f"{OUT}/kensyu_mobile_*_db.json"))
    print(f"[B] 入力 PC版 {len(files)}ファイル ＋ スマホ {len(mfiles)}ファイル")
    g1 = {}
    with open(f"{BASE}/dataset/G1_events_all.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f): g1.setdefault((r["video"], r["counter"], r["count_value"]), r)
    latest = {}; hist = collections.defaultdict(list)
    for p in files:
        m = re.search(r"_(\d{12})\.csv$", os.path.basename(p)); stamp = m.group(1) if m else "202609040000"  # kensyu_judgements.csv＝9/4以前の統合（レジストリの judged_on <=2026-09-04）
        judged_at = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}"
        rs = parse_kensyu(p); print(f"  {os.path.basename(p):<48} {len(rs):>4}行  {dict(collections.Counter(r['judgement'] for r in rs))}")
        for r in rs:
            key = (r["video"], r["counter"], r["count_value"]); hist[key].append((stamp, r["judgement"], os.path.basename(p)))
            if key not in latest or stamp >= latest[key]["_stamp"]:
                latest[key] = dict(r, _stamp=stamp, source=os.path.basename(p), judged_at=judged_at)
    n_mobile = 0
    for p in mfiles:  # スマホ検収（10/8 追加）。判定ごとの t（ms epoch）を時刻に使う。file名の時刻は無い
        m = re.search(r"kensyu_mobile_(\d{8})_db\.json$", os.path.basename(p)); src = f"mobile_{m.group(1) if m else 'unknown'}"
        rs = parse_mobile(p); print(f"  {os.path.basename(p):<48} {len(rs):>4}行  {dict(collections.Counter(r['judgement'] for r in rs))}  source={src}")
        for r in rs:
            key = (r["video"], r["counter"], r["count_value"]); stamp = r["_stamp"] or "209912312359"; hist[key].append((stamp, r["judgement"], src)); n_mobile += 1
            if key not in latest or stamp >= latest[key]["_stamp"]:
                latest[key] = dict(r, _stamp=stamp, source=src, judged_at=(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}" if r["_stamp"] else ""))
    out = []
    for key, r in latest.items():
        video = r["video"]; m = re.search(r"_(\d{8})_\d{6}$", video); sess = m.group(1) if m else ""
        g = g1.get(key); hs = sorted(hist[key])
        prior = ";".join(f"{s[:8]}:{j}" for s, j, _ in hs[:-1]) if len(hs) > 1 else ""
        out.append(dict(session=sess, video=video, counter=r["counter"], counter_ja=JA.get(r["counter"], r["counter"]), count_value=int(r["count_value"]),
                        video_time_sec=(g["video_time_sec"] if g else ""), stable_read_time_sec=(g["stable_read_time_sec"] if g else ""), judgement=r["judgement"],
                        source=r["source"], judged_at=r["judged_at"], n_times_judged=len(hs), prior_judgements=prior, in_G1_ledger=("1" if g else "0")))
    out.sort(key=lambda r: (r["session"], r["video"], float(r["video_time_sec"]) if r["video_time_sec"] != "" else 1e9, r["counter"], r["count_value"]))
    cols = ["session", "video", "counter", "counter_ja", "count_value", "video_time_sec", "stable_read_time_sec", "judgement", "source", "judged_at", "n_times_judged", "prior_judgements", "in_G1_ledger"]
    wcsv(f"{OUT}/検収台帳_統合.csv", out, cols)
    # 集計
    g1_by_sess = collections.Counter(r["session"] for r in g1.values())
    sess = sorted({r["session"] for r in out})
    L = ["# 検収 集計（PC版 kensyu_*.csv ＋ スマホ検収 db の統合）", "",
         "- 検収＝正解G1イベント（カウンタ加点）1件ずつを本人が目視で ○（OK＝加点が歌唱と対応）／×（NG）／保留（HOLD）／EXCLUDED と判定したもの。",
         f"- 入力＝{BASE}/kensyu_*.csv（PC版 {len(files)}ファイル）＋ ledgers_20261008/kensyu_mobile_<sess>_db.json（スマホ検収 {len(mfiles)}ファイル・{n_mobile}判定）＝延べ {sum(len(h) for h in hist.values())}判定。同一イベントの重複は新しい判定（PC版＝ファイル名の時刻・スマホ＝判定ごとの t）を優先。kensyu_judgements.csv は時刻なし＝2026-09-04 00:00 扱い。",
         "- スマホ検収＝Artifact db kensyu/<sess> の書き出し（{\"j\":{event_id:{v,t}}}）。o→OK／x→NG／k→HOLD。source 列＝mobile_<sess>。PC版と判定は別保存＝同じイベントが両方にあれば新しい方を採る。",
         "- 時刻＝dataset/G1_events_all.csv の video_time_sec（録画の時刻）。in_G1_ledger=0 は結合先なし。", "",
         "| セッション | 検収件数 | OK | NG | HOLD | EXCLUDED | ○率 OK/(OK+NG+HOLD) | G1イベント総数 | 検収カバー率 |", "|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for s in sess:
        c = collections.Counter(r["judgement"] for r in out if r["session"] == s); n = sum(c.values()); den = c["OK"] + c["NG"] + c["HOLD"]
        tot.update(c); L.append(f"| {s} | {n} | {c['OK']} | {c['NG']} | {c['HOLD']} | {c['EXCLUDED']} | {c['OK'] / den if den else 0:.3f} | {g1_by_sess.get(s, 0)} | {n / g1_by_sess[s] if g1_by_sess.get(s) else 0:.3f} |")
    n = sum(tot.values()); den = tot["OK"] + tot["NG"] + tot["HOLD"]; g1n = sum(g1_by_sess[s] for s in sess)
    L.append(f"| **合計** | {n} | {tot['OK']} | {tot['NG']} | {tot['HOLD']} | {tot['EXCLUDED']} | {tot['OK'] / den if den else 0:.3f} | {g1n} | {n / g1n if g1n else 0:.3f} |")
    L += ["", "## 技法別", "", "| 技法 | 検収件数 | OK | NG | HOLD | ○率 |", "|---|---|---|---|---|---|"]
    for c in TYPES:
        cc = collections.Counter(r["judgement"] for r in out if r["counter"] == c); den = cc["OK"] + cc["NG"] + cc["HOLD"]
        L.append(f"| {JA[c]} | {sum(cc.values())} | {cc['OK']} | {cc['NG']} | {cc['HOLD']} | {cc['OK'] / den if den else 0:.3f} |")
    chg = [r for r in out if r["prior_judgements"] and any(x.split(":")[1] != r["judgement"] for x in r["prior_judgements"].split(";"))]
    L += ["", f"- 判定が途中で変わったイベント＝{len(chg)}件（prior_judgements 列に旧判定・最終判定は judgement 列）。", f"- 台帳本体＝検収台帳_統合.csv（{len(out)}行・1行＝1イベントの最終判定）。",
          f"- 生成＝build_ledgers_20261008.py（{time.strftime('%Y-%m-%d %H:%M')}）"]
    open(f"{OUT}/検収_集計.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"[B] 台帳 {len(out)}行  ", {s: f"OK {collections.Counter(r['judgement'] for r in out if r['session'] == s)['OK']}/{sum(1 for r in out if r['session'] == s)}" for s in sess})
    return out

if __name__ == "__main__":
    if not a.skip_a: build_a()
    if not a.skip_b: build_b()
    print(f"所要 {time.time() - T_START:.0f}s  出力先 {OUT}")
