# -*- coding: utf-8 -*-
r"""録音WAV 1本を 1コマンドで評価する（2026-10-08・E1）。新しい学習はしない。
   流れ: 特徴量（feats/<名前>.npz・あれば再利用）→ 学習済みCRNNアンサンブル（v2d+v2d2+v2e・確率平均・lag +0.3s・
         しきい値キャッシュ out/pitchbar_thr_v2d+v2d2+v2e_lag0.3.json）で予測 → 検出表 CSV（時刻順）
         → 正解（index.json の G1）があれば ±1s 貪欲1対1照合（build_ledgers_20261008.py と同じ規則）で TP/FP/FN 台帳＋技法別集計
         → DAM 採点CSV があれば技法回数の件数比較 → 音程バー図 PNG（render_pitchbar.py を呼ぶ）→ summary.md
   使い方:
     python evaluate_recording.py --wav "G:\研究アーカイブ\録音\songs_20260924_raw\T02_20260924_101915.wav" --out G:\...\eval_lulu
       省略可: --video lulu._20260924_101433（index.json のエントリ名。省略時は WAV のパス一致で探す）
               --dam-csv "G:\研究アーカイブ\成果物\dam_api\dam_ai-hearts_full_2026-09-24T15-55-17.csv"
               --tags v2d,v2d2,v2e  --lag 0.3  --tol 1.0  --no-fig  --threads 8
               --html  1録音レポート（録音1本につき1枚・report_<録音ID>.html・録音ID＝WAV の stem・自己完結 HTML・画像 base64 埋め込み）を追加出力
   出力（--out 直下・<名前>＝index の video 名、無ければ WAV のファイル名）:
     <名前>_detections.csv   予測した加点イベント（technique, technique_ja, time_sec, score, in_span）時刻順
     <名前>_matching.csv     正解あり時：各加点の判定（kind=TP/FP/FN, technique, pred_time_sec, truth_time_sec, diff_sec, score, truth_video_time_sec）
     <名前>_by_technique.csv 正解あり時：技法別集計（n_truth, n_pred, TP, FP, FN, precision, recall, f1）＋ALL
     <名前>_dam_compare.csv  DAM CSV あり時：技法別の DAM回数／予測件数／正解件数（G1）
     <名前>_pitchbar.png     音程バー図（render_pitchbar.py）
     <名前>_summary.md       上記のまとめと所要時間
     report_<録音ID>.html    --html 時のみ。1録音レポート＝録音だけから検出器で得た独自の分析（録音の情報・DAM 風の集計・音程バー図＝検出マークのみ・
                             加点候補の一覧・候補ごとの波形と音程の小図）。正解との判定・DAM 比較は入れない（それは補助レポート＝上の CSV／summary）"""
import os, sys, re, json, csv, time, argparse, warnings, subprocess, collections, datetime
import numpy as np
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
T0 = time.time()
BASE = "G:/研究アーカイブ/成果物/動画解析"; ROOT = f"{BASE}/g3v2_20260928"
SR = 22050; HOP = 220; NMEL = 80; FR = 100
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
JA = {"shakuri": "しゃくり", "kobushi": "こぶし", "vibrato": "ビブラート", "fall": "フォール", "accent": "アクセント", "hammering": "ハンマリング"}
DAMCOL = {"shakuri": "shakuriCount", "kobushi": "kobushiCount", "vibrato": "vibratoCount", "fall": "fallCount", "accent": "accentCount", "hammering": "hammeringOnCount"}

ap = argparse.ArgumentParser()
ap.add_argument("--wav", required=True); ap.add_argument("--video", default=""); ap.add_argument("--dam-csv", dest="dam_csv", default="")
ap.add_argument("--out", default=f"{ROOT}/out/evaluate"); ap.add_argument("--tags", default="v2d,v2d2,v2e"); ap.add_argument("--lag", type=float, default=0.3)
ap.add_argument("--tol", type=float, default=1.0); ap.add_argument("--dev", default="20260911"); ap.add_argument("--hid", type=int, default=128)
ap.add_argument("--threads", type=int, default=8); ap.add_argument("--no-fig", dest="no_fig", action="store_true")
ap.add_argument("--html", action="store_true", help="1録音レポート（report_<録音ID>.html・録音だけから得た独自分析・自己完結・画像base64埋め込み）を追加出力する。既存の出力は変えない")
a = ap.parse_args(); TAGS = a.tags.split(","); os.makedirs(a.out, exist_ok=True)
LOG = []
def say(s): print(s, flush=True); LOG.append(s)
def wcsv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
def r3(x): return round(x, 3) if isinstance(x, float) else x

# ---------- 1. index との対応（render_pitchbar.py と同じ） ----------
if not os.path.exists(a.wav): sys.exit(f"WAV がありません: {a.wav}")
index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
def norm_path(p): return os.path.normcase(os.path.normpath(p)).replace("\\", "/")
entry = None
if a.video:
    entry = next((d for d in index if d["video"] == a.video), None)
    if entry is None: sys.exit(f"index.json に video={a.video} がありません")
else:
    base = os.path.basename(a.wav)
    hits = [d for d in index if os.path.basename(d["wav"]) == base and norm_path(d["wav"]) == norm_path(a.wav)] or [d for d in index if os.path.basename(d["wav"]) == base]
    entry = hits[0] if hits else None
stem = os.path.splitext(os.path.basename(a.wav))[0]; name = entry["video"] if entry else stem
say(f"# 評価: {name}"); say(f"- WAV: {a.wav}"); say(f"- 正解: {'あり（index.json ' + entry['video'] + '・session ' + entry['session'] + '・' + str(entry['n']) + '件）' if entry else 'なし（予測のみ）'}")

# ---------- 2. 特徴量（extract_feats.py と同一手順・キャッシュ feats/<名前>.npz） ----------
def find_pitch_csv():
    dirs = [f"{BASE}/dataset/{d}" for d in sorted(os.listdir(f"{BASE}/dataset")) if d.startswith("pitch_pyin")]
    names = [stem + "_f0.csv"] + ([entry["video"] + "_f0.csv"] if entry else [])
    for d in dirs:
        for n in names:
            if os.path.exists(f"{d}/{n}"): return f"{d}/{n}"
    if entry and entry.get("pitch") and os.path.exists(entry["pitch"]): return entry["pitch"]
    return None
def build_feats(wav, out):
    import librosa, soundfile as sf, pandas as pd
    y, sr = sf.read(wav, dtype="float32", always_2d=True); y = y.mean(1)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=HOP, n_mels=NMEL, fmin=50, fmax=8000, power=2.0)
    logmel = np.log(S + 1e-6).astype(np.float32)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]; rms = np.log(rms + 1e-6).astype(np.float32)
    T = logmel.shape[1]; t = np.arange(T) * HOP / SR
    pf = find_pitch_csv()
    if pf:
        p = pd.read_csv(pf, encoding="utf-8-sig"); pt = p["audio_time_sec"].values.astype(float); f0 = p["f0_hz"].values.astype(float); conf = p["conf"].values.astype(float); src = "csv"
    else:
        say("  pYIN をいま計算（1〜3分）…")
        f0, vf, vp = librosa.pyin(y, fmin=70, fmax=800, sr=SR, frame_length=2048, hop_length=HOP)
        f0 = np.nan_to_num(f0, nan=0.0); conf = np.nan_to_num(vp, nan=0.0); pt = np.arange(len(f0)) * HOP / SR; src = "pyin_now"
    st = np.where(f0 > 40, 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0), np.nan)
    idx = np.clip(np.searchsorted(pt, t), 0, len(pt) - 1)
    st_g = st[idx]; voiced = (~np.isnan(st_g)).astype(np.float32); conf_g = conf[idx].astype(np.float32)
    np.savez_compressed(out, logmel=logmel.astype(np.float16), rms=rms, st=np.nan_to_num(st_g, nan=0.0).astype(np.float32), voiced=voiced, conf=conf_g, t0=0.0, hop=HOP / SR, src=src)
    return src
fpath = f"{ROOT}/feats/{name}.npz"
t1 = time.time()
if os.path.exists(fpath): say(f"- 特徴量: キャッシュ再利用 {fpath}")
else:
    src = build_feats(a.wav, fpath); say(f"- 特徴量: 生成 {fpath}（ピッチ={src}・{time.time() - t1:.0f}s）")
Z = np.load(fpath); DUR = Z["st"].shape[0] / FR
T_FEAT = time.time() - t1

# ---------- 3. 学習済みモデル（render_pitchbar.py / build_ledgers_20261008.py と同一構造・確率平均） ----------
import torch, torch.nn as nn, torch.nn.functional as Fn
torch.set_num_threads(a.threads)
def feats_of(z):
    lm = z["logmel"].astype(np.float32); T = lm.shape[1]; st = z["st"]; v = z["voiced"]
    stn = np.where(v > 0, (st - 60.0) / 12.0, 0.0).astype(np.float32)
    dst = np.zeros(T, np.float32); dst[1:] = np.diff(st); dst[1:] *= (v[:-1] > 0); dst = np.where(v > 0, np.clip(dst, -3, 3), 0.0).astype(np.float32)
    return np.concatenate([lm, z["rms"][None, :], stn[None, :], v[None, :], z["conf"][None, :], dst[None, :]], 0).astype(np.float32)
norm0 = np.load(f"{ROOT}/out/{TAGS[0]}/norm.npy"); CIN = norm0.shape[1]
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
    n_ = Net(); n_.load_state_dict(torch.load(f"{ROOT}/out/{tg}/model.pt")); n_.eval(); nm = np.load(f"{ROOT}/out/{tg}/norm.npy")
    nets.append((n_, torch.tensor(nm[0][:, None]), torch.tensor(nm[1][:, None])))
def probs_of(z):
    x = torch.tensor(feats_of(z)[None])
    with torch.no_grad(): return np.mean([torch.sigmoid(n_((x - m_) / s_))[0].numpy() for n_, m_, s_ in nets], 0)
# --- build_ledgers_20261008.py（A. 検出照合台帳）から流用：pick（局所最大＋lag）・match（貪欲1対1） ---
def pick(prob, thr, md):
    w = int(md * FR); n = len(prob)
    return [(i / FR + a.lag, float(prob[i])) for i in range(1, n - 1) if prob[i] >= thr and prob[i] >= prob[max(0, i - w):min(n, i + w + 1)].max()]
def match(pred, truth, tol):
    """貪欲1対1照合：予測を時刻順に、未使用の正解のうち最も近いもの（±tol）を取る"""
    used = set(); pairs = []
    for pi in sorted(range(len(pred)), key=lambda i: pred[i]):
        best = None
        for k, t in enumerate(truth):
            if k in used or abs(t - pred[pi]) > tol: continue
            if best is None or abs(t - pred[pi]) < abs(truth[best] - pred[pi]): best = k
        if best is not None: used.add(best); pairs.append((pi, best))
    return pairs

# ---------- 4. しきい値（キャッシュ。無ければ dev で ±1s F1 最大・技法別に決めて保存） ----------
thr_file = f"{ROOT}/out/pitchbar_thr_{'+'.join(TAGS)}_lag{a.lag}.json"
if os.path.exists(thr_file):
    best = {c: tuple(v) for c, v in json.load(open(thr_file, encoding="utf-8")).items()}; say(f"- しきい値: キャッシュ {os.path.basename(thr_file)}")
else:
    say(f"- しきい値を dev({a.dev}) で決定中（初回のみ・数分）…"); tt = time.time()
    dev = [d for d in index if d["session"] == a.dev and os.path.exists(f"{ROOT}/feats/{d['video']}.npz")]
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
    json.dump(best, open(thr_file, "w", encoding="utf-8"), ensure_ascii=False, indent=1); say(f"  {time.time() - tt:.0f}s → {thr_file}")
say(f"- しきい値(thr, 最短間隔s): { {JA[c]: best[c] for c in TYPES} }")

# ---------- 5. 予測 → 検出表 ----------
t2 = time.time(); PR = probs_of(Z); T_PRED = time.time() - t2
lo, hi = entry["span"] if entry else [0.0, DUR]
det = []
for c in TYPES:
    for t, p in pick(PR[TYPES.index(c)], *best[c]): det.append(dict(technique=c, technique_ja=JA[c], time_sec=round(t, 3), score=round(p, 3), in_span=int(lo <= t <= hi)))
det.sort(key=lambda e: e["time_sec"])
wcsv(f"{a.out}/{name}_detections.csv", det, ["technique", "technique_ja", "time_sec", "score", "in_span"])
n_in = collections.Counter(e["technique"] for e in det if e["in_span"])
say(f"- 予測: {len(det)}件（採点区間 {lo:.1f}〜{hi:.1f}s 内 {sum(n_in.values())}件）→ {name}_detections.csv")
say("  予測件数（区間内）: " + "／".join(f"{JA[c]} {n_in.get(c, 0)}" for c in TYPES))

# ---------- 6. 正解との照合（build_ledgers_20261008.py と同じ出力列） ----------
agg = {}; tot = (0, 0, 0)
if entry:
    off = float(entry.get("offset", 0.0)); rows = []
    for c in TYPES:
        pred = [(t, p) for t, p in pick(PR[TYPES.index(c)], *best[c]) if lo <= t <= hi]
        truth = [t for cc, t in entry["events"] if cc == c and lo <= t <= hi]
        pt = [t for t, _ in pred]; pairs = match(pt, truth, a.tol); mp = {i for i, _ in pairs}; mt = {k for _, k in pairs}
        ev = []
        for i, k in pairs: ev.append(dict(kind="TP", pred_time_sec=pt[i], truth_time_sec=truth[k], diff_sec=pt[i] - truth[k], score=pred[i][1]))
        for i in range(len(pt)):
            if i not in mp: ev.append(dict(kind="FP", pred_time_sec=pt[i], truth_time_sec="", diff_sec="", score=pred[i][1]))
        for k in range(len(truth)):
            if k not in mt: ev.append(dict(kind="FN", pred_time_sec="", truth_time_sec=truth[k], diff_sec="", score=""))
        for e in ev:
            tv = e["truth_time_sec"]
            rows.append(dict(session=entry["session"], song=name.rsplit("_", 2)[0], video=name, technique=c, technique_ja=JA[c], **{k: r3(v) for k, v in e.items()}, truth_video_time_sec=(round(tv - off, 3) if tv != "" else "")))
        agg[c] = (len(pairs), len(pt) - len(pairs), len(truth) - len(pairs))
    rows.sort(key=lambda e: e["pred_time_sec"] if e["pred_time_sec"] != "" else e["truth_time_sec"])
    wcsv(f"{a.out}/{name}_matching.csv", rows, ["session", "song", "video", "technique", "technique_ja", "kind", "pred_time_sec", "truth_time_sec", "diff_sec", "score", "truth_video_time_sec"])
    tot = tuple(sum(v[i] for v in agg.values()) for i in range(3))
    def srow(c, tp, fp, fn):
        p, r, f = prf(tp, fp, fn); return dict(technique=c, technique_ja=JA.get(c, "全体"), n_truth=tp + fn, n_pred=tp + fp, TP=tp, FP=fp, FN=fn, precision=round(p, 3), recall=round(r, 3), f1=round(f, 3))
    srows = [srow(c, *agg[c]) for c in TYPES] + [srow("ALL", *tot)]
    wcsv(f"{a.out}/{name}_by_technique.csv", srows, list(srows[0].keys()))
    say(f"- 照合（±{a.tol}s・採点区間内）: 一致 {tot[0]}／誤検知 {tot[1]}／見逃し {tot[2]}  適合率 {prf(*tot)[0]:.3f} 再現率 {prf(*tot)[1]:.3f} F1 {prf(*tot)[2]:.3f} → {name}_matching.csv・{name}_by_technique.csv")
    say(""); say("| 技法 | 正解 | 予測 | TP | FP | FN | 適合率 | 再現率 | F1 |"); say("|---|---|---|---|---|---|---|---|---|")
    for s in srows: say(f"| {s['technique_ja']} | {s['n_truth']} | {s['n_pred']} | {s['TP']} | {s['FP']} | {s['FN']} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} |")
    say("")

# ---------- 7. DAM 採点CSV との件数比較 ----------
dam_row = None; drows = []
if a.dam_csv:
    import pandas as pd
    if not os.path.exists(a.dam_csv): say(f"- DAM CSV がありません: {a.dam_csv}")
    else:
        dam = pd.read_csv(a.dam_csv, encoding="utf-8-sig"); dam["_t"] = pd.to_datetime(dam["playDate"], errors="coerce")
        m = re.search(r"_(\d{8})_(\d{6})$", stem) or (re.search(r"_(\d{8})_(\d{6})$", entry["video"]) if entry else None)
        row = None
        if m is not None:
            t_wav = datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            cand = dam.dropna(subset=["_t"]).copy(); cand["_d"] = (cand["_t"] - t_wav).abs().dt.total_seconds()
            cand = cand[cand["_d"] <= 15 * 60].sort_values("_d")
            if len(cand): row = cand.iloc[0]; say(f"- DAM 行: {row['songName']}（playDate {row['playDate']}・WAV名の時刻との差 {row['_d'] / 60:.1f}分・候補 {len(cand)}件）")
        if row is None:
            say("- DAM 行: WAV名／video名の時刻（_YYYYMMDD_HHMMSS）±15分に playDate が無いため時刻で照合不可。songName で探します")
            song = name.rsplit("_", 2)[0]; cand = dam[dam["songName"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip() == song]
            if len(cand): row = cand.sort_values("_t").iloc[-1]; say(f"  songName 一致: {row['songName']}（playDate {row['playDate']}）")
            else: say("  見つからず（DAM 比較なし）")
        if row is not None:
            dam_row = row
            for c in TYPES:
                dc = int(row[DAMCOL[c]]) if DAMCOL[c] in row and str(row[DAMCOL[c]]) not in ("", "nan") else ""
                g1 = (agg[c][0] + agg[c][2]) if entry else ""
                drows.append(dict(technique=c, technique_ja=JA[c], dam_count=dc, pred_count=n_in.get(c, 0), g1_count=g1, pred_minus_dam=(n_in.get(c, 0) - dc if dc != "" else ""), g1_minus_dam=(g1 - dc if (dc != "" and g1 != "") else "")))
            wcsv(f"{a.out}/{name}_dam_compare.csv", drows, list(drows[0].keys()))
            sd = sum(int(r["dam_count"]) for r in drows if r["dam_count"] != ""); sp = sum(r["pred_count"] for r in drows); ad = sum(abs(r["pred_minus_dam"]) for r in drows if r["pred_minus_dam"] != "")
            say(f"- DAM 件数比較 → {name}_dam_compare.csv（DAM合計 {sd}／予測合計 {sp}／件数整合 1−Σ|予測−DAM|/ΣDAM = {1 - ad / sd if sd else 0:.3f}）")
            say(""); say("| 技法 | DAM回数 | 予測件数 | 正解G1件数 | 予測−DAM | G1−DAM |"); say("|---|---|---|---|---|---|")
            for r in drows: say(f"| {r['technique_ja']} | {r['dam_count']} | {r['pred_count']} | {r['g1_count']} | {r['pred_minus_dam']} | {r['g1_minus_dam']} |")
            say("")

# ---------- 8. 音程バー図（render_pitchbar.py） ----------
if not a.no_fig:
    t3 = time.time(); cmd = [sys.executable, f"{ROOT}/scripts/render_pitchbar.py", "--wav", a.wav, "--out", a.out, "--tags", a.tags, "--lag", str(a.lag), "--tol", str(a.tol), "--threads", str(a.threads)]
    if entry: cmd += ["--video", entry["video"]]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    png = f"{a.out}/{name}_pitchbar.png"
    if r.returncode == 0 and os.path.exists(png):
        say(f"- 音程バー図: {png}（{time.time() - t3:.0f}s）")
        fig_line = next((l for l in r.stdout.splitlines() if l.startswith("一致")), "")
        if entry and fig_line:
            ok = fig_line.startswith(f"一致 {tot[0]}／誤検知 {tot[1]}／見逃し {tot[2]}")
            say(f"  図の照合値: {fig_line}  → 本ツールと{'一致' if ok else '不一致（要確認）'}")
    else: say(f"- 音程バー図: 失敗（rc={r.returncode}）\n{r.stderr[-1500:]}")

# ---------- 8.5. 1録音レポート（HTML・--html・録音1本につき1枚）＝録音だけから検出器で得た独自の分析結果 ----------
# 本人指示（10/08 16:4x・16:5x）：DAM の CSV は「形式の手本」にすぎず、入力は録音だけ（DAM の採点が無い録音でも出せる）。音声波形を見せたいので画像を使う。
# 正解（G1）との判定・DAM 件数比較・測れる範囲表・加点の遅れは「補助レポート」の仕事なのでここには入れない（既存の判定 CSV・summary はそのまま）。
# 構成＝①録音の情報 ②DAM 風の集計（技法別の検出回数・ビブラートの合計秒／平均深さ／平均速さ・歌唱音域・有声時間の割合）
#       ③音程バー図（録音全体・検出マークのみ＝render_pitchbar_icons.py と同じ見た目を本スクリプト内で描く・正解レーンは描かない・≤2800px の帯に分割して埋め込み）
#       ④加点候補の一覧（時刻順） ⑤候補ごとの小図（−2.5〜+0.5 s の波形と pYIN 音程・技法の図形アイコン）。画像は全部 base64 埋め込み・スマホ幅対応。
HTML_NOTE = ""
if a.html:
    import base64, io, shutil, html as _h
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import soundfile as sf
    from scipy.signal import find_peaks; from scipy.ndimage import median_filter
    t4 = time.time(); NFIG = [0]; E = _h.escape
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
    # 技法の色と形＝render_pitchbar_icons.py（DAM 画面の技法マークを目視して決めた図形・色）と同じ
    ICOL = {"shakuri": "#d9262c", "kobushi": "#2a5bd7", "vibrato": "#1fb83a", "fall": "#c13ab6", "accent": "#ef8a1c", "hammering": "#15b1c9"}
    ICORE = {c: "white" for c in TYPES}; ICORE["accent"] = "#ffe14d"
    NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    def _bez(P, n=60):
        P = np.asarray(P, float); t = np.linspace(0, 1, n)[:, None]
        if len(P) == 3: return (1 - t) ** 2 * P[0] + 2 * (1 - t) * t * P[1] + t ** 2 * P[2]
        return (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
    def _rpl(V, rc=0.22, n=24):
        V = np.asarray(V, float); pts = [V[0]]
        for i in range(1, len(V) - 1):
            a_, b_, c_ = V[i - 1], V[i], V[i + 1]
            da = (a_ - b_) / np.linalg.norm(a_ - b_); dc = (c_ - b_) / np.linalg.norm(c_ - b_)
            p = b_ + da * min(rc, np.linalg.norm(a_ - b_) / 2); q = b_ + dc * min(rc, np.linalg.norm(c_ - b_) / 2)
            seg = np.linspace(pts[-1], p, n); pts.extend(seg[1:]); pts.extend(_bez([p, b_, q], n)[1:])
        seg = np.linspace(pts[-1], V[-1], n); pts.extend(seg[1:]); return np.array(pts)
    def _spiral(th0=75, th1=-230, r0=0.88, r1=0.55, n=120):
        th = np.radians(np.linspace(th0, th1, n)); r = np.linspace(r0, r1, n); return np.c_[r * np.cos(th), r * np.sin(th)]
    SHAPE = {"shakuri": lambda: _bez([(-0.72, -0.95), (0.95, -0.95), (-0.95, 0.95), (0.72, 0.95)], 80), "kobushi": lambda: _spiral(),
             "vibrato": lambda: _rpl([(-1.0, -0.85), (-0.5, 0.85), (0.0, -0.35), (0.5, 0.85), (1.0, -0.85)], 0.4), "fall": lambda: _bez([(-0.9, 0.72), (0.6, 0.72), (0.5, -0.95)], 80),
             "accent": lambda: _rpl([(-0.95, 0.75), (-0.35, -0.92), (0.05, 0.55), (0.98, 0.55)], 0.36), "hammering": lambda: _rpl([(-1.0, -0.6), (-0.35, -0.6), (0.35, 0.6), (1.0, 0.6)], 0.28)}
    def b64(fig, **kw):
        buf = io.BytesIO(); fig.savefig(buf, format="png", **kw); plt.close(fig); NFIG[0] += 1
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    def icon_uri(c, hpx=48):
        """技法マーク（帯＝技法色・芯＝白）を hpx 角の透過 PNG に"""
        f = plt.figure(figsize=(hpx / 100, hpx / 100), dpi=100); ax = f.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        P = SHAPE[c](); band_pt = 0.26 * hpx / 2.6 * 72 / 100
        ax.add_line(Line2D(P[:, 0], P[:, 1], color=ICOL[c], lw=band_pt, solid_capstyle="round", solid_joinstyle="round"))
        ax.add_line(Line2D(P[:, 0], P[:, 1], color=ICORE[c], lw=band_pt * 0.42, solid_capstyle="round", solid_joinstyle="round"))
        return b64(f, dpi=100, transparent=True)
    ICON = {c: icon_uri(c) for c in TYPES}
    # (1) 録音の情報（index は曲名の参照にだけ使う。正解は使わない）
    info = sf.info(a.wav); DUR_W = float(info.duration)
    song = name.rsplit("_", 2)[0] if entry else "—（index.json に未登録）"
    mdt = re.search(r"_(\d{8})_(\d{6})$", stem)
    rec_dt = f"{mdt.group(1)[:4]}-{mdt.group(1)[4:6]}-{mdt.group(1)[6:]} {mdt.group(2)[:2]}:{mdt.group(2)[2:4]}:{mdt.group(2)[4:]}" if mdt else "—（ファイル名に _YYYYMMDD_HHMMSS が無い）"
    gear = "録音ファイル WAV（T番号＝SSL2 の連番ファイル・README 参照）" if re.match(r"^T\d+_", stem) else "録音ファイル WAV"
    def mmss(t): return f"{int(t // 60)}:{t % 60:04.1f}"
    def nname(s): s = int(round(s)); return f"{NOTE[s % 12]}{s // 12 - 1}"
    def hz(s): return 440.0 * 2 ** ((s - 69) / 12)
    # (2) DAM 風の集計＝検出器の出力から計算（録音全体）
    ST = Z["st"].astype(np.float64); VO = Z["voiced"] > 0.5
    cnt = collections.Counter(e["technique"] for e in det)
    def vib_measure(t_ev, pre=1.5, post=0.3, med=0.3, dmin=15.0, fmin=3.0, fmax=9.0):
        """vibrato_depth_20261008/vibrato_depth.py の measure() と同一定義：窓 [t−1.5, t+0.3] s・有声区間ごとに 0.3 s 移動中央値を引いた残差・
        山谷差 ≥15 cent かつ半周期 3〜9 Hz の対が 3 つ以上続く最長の連鎖＝周期的な揺れ。深さ＝連鎖内の山谷差の中央値・速さ＝残差の FFT ピーク・持続＝連鎖の端から端"""
        hop = float(Z["hop"]); t0 = float(Z["t0"]); st = Z["st"]; v = Z["voiced"] > 0.5; T = len(st)
        i0 = max(0, int(round((t_ev - pre - t0) / hop))); i1 = min(T, int(round((t_ev + post - t0) / hop)))
        cents = st[i0:i1].astype(np.float64) * 100.0; vo = v[i0:i1]; n = len(vo)
        out = dict(depth_cents=0.0, rate_hz=np.nan, duration_s=0.0, n_halfcycles=0)
        if n == 0: return out
        runs = []; i = 0
        while i < n:
            if not vo[i]: i += 1; continue
            j = i
            while j < n and vo[j]: j += 1
            if j - i >= int(med * FR): runs.append((i, j))
            i = j
        if not runs: return out
        kmed = int(med * FR) | 1; resid = np.full(n, np.nan)
        for (i, j) in runs: c = cents[i:j]; resid[i:j] = c - median_filter(c, size=kmed, mode="nearest")
        ext = []
        for (i, j) in runs:
            r = resid[i:j]; pk, _ = find_peaks(r, distance=3); tr, _ = find_peaks(-r, distance=3)
            e = sorted([(p + i, r[p], 1) for p in pk] + [(q + i, r[q], -1) for q in tr]); alt = []
            for x in e:
                if alt and alt[-1][2] == x[2]:
                    if (x[2] == 1 and x[1] > alt[-1][1]) or (x[2] == -1 and x[1] < alt[-1][1]): alt[-1] = x
                else: alt.append(x)
            ext.append(alt)
        diffs_all = [abs(al[k + 1][1] - al[k][1]) for al in ext for k in range(len(al) - 1)]
        depth_all = float(np.median(diffs_all)) if diffs_all else 0.0
        hmin, hmax = 1.0 / (2 * fmax), 1.0 / (2 * fmin); best_ = None
        for al in ext:
            k = 0
            while k < len(al) - 1:
                m = k
                while m < len(al) - 1 and abs(al[m + 1][1] - al[m][1]) >= dmin and hmin <= (al[m + 1][0] - al[m][0]) * hop <= hmax: m += 1
                if m - k >= 3:
                    span = (al[m][0] - al[k][0]) * hop
                    if best_ is None or span > best_[0]: best_ = (span, al[k:m + 1])
                k = max(m, k + 1)
        def fft_rate(seg):
            seg = seg[np.isfinite(seg)]
            if len(seg) < 10: return np.nan
            x = (seg - seg.mean()) * np.hanning(len(seg)); N = 4096; P = np.abs(np.fft.rfft(x, N)) ** 2; f = np.fft.rfftfreq(N, hop)
            band = (f >= fmin) & (f <= fmax)
            return float(f[band][np.argmax(P[band])]) if band.any() else np.nan
        if best_ is not None:
            chain = best_[1]; pp = [abs(chain[k + 1][1] - chain[k][1]) for k in range(len(chain) - 1)]
            out["depth_cents"] = round(float(np.median(pp)), 1); out["duration_s"] = round(float(best_[0]), 2); out["n_halfcycles"] = len(pp)
            seg = resid[chain[0][0]:chain[-1][0] + 1]
            if len(seg) < 30: seg = resid[max(0, chain[0][0] - 10):min(n, chain[-1][0] + 11)]
            r = fft_rate(seg)
        else:
            out["depth_cents"] = round(depth_all, 1); i, j = max(runs, key=lambda r_: r_[1] - r_[0]); r = fft_rate(resid[i:j])
        out["rate_hz"] = round(r, 2) if np.isfinite(r) else np.nan
        return out
    def vib_in_range(m):
        """測れる範囲（vibrato_depth_20261008/calib/報告_測定の信頼性_2026-10-08.md §4）の中の値だけ返す。範囲外は None（表示「—」）。
        深さ: 連鎖あり・速さ 4〜5 Hz・深さ ≥40 cent・持続 ≥0.6 s／速さ: 連鎖あり・4〜6 Hz／持続: 連鎖あり・速さ ≤6 Hz・持続 ≥0.6 s"""
        chain = m["n_halfcycles"] >= 3; r = m["rate_hz"]; rf = bool(np.isfinite(r))
        dep = m["depth_cents"] if (chain and rf and 4.0 <= r <= 5.0 and m["depth_cents"] >= 40 and m["duration_s"] >= 0.6) else None
        rate = r if (chain and rf and 4.0 <= r <= 6.0) else None
        dur = m["duration_s"] if (chain and rf and r <= 6.0 and m["duration_s"] >= 0.6) else None
        return dep, rate, dur
    cands = [dict(i=i + 1, c=e["technique"], t=float(e["time_sec"]), score=float(e["score"])) for i, e in enumerate(det)]   # det は時刻順
    for k in cands: k["vib"] = vib_in_range(vib_measure(k["t"])) if k["c"] == "vibrato" else None
    vibs = [k["vib"] for k in cands if k["vib"] is not None]
    v_dep = [v[0] for v in vibs if v[0] is not None]; v_rate = [v[1] for v in vibs if v[1] is not None]; v_dur = [v[2] for v in vibs if v[2] is not None]
    voiced_frac = float(VO.mean()); voiced_sec = float(VO.sum()) / FR
    # (3) 音程バー図（録音全体・検出マークのみ）＝render_pitchbar_icons.py と同じ見た目（持続音の規則・半音吸着・図形マーク）を本スクリプト内で描く。
    #     正解レーン・一致線・誤検知／見逃しの帯は描かない（index に無い WAV として外部スクリプトを呼ぶと全予測が「誤検知」の赤帯になるため自前で描く）
    from matplotlib.patches import Rectangle
    st_f = ST.copy(); st_f[~VO] = np.nan; idx_v = np.flatnonzero(VO)
    if len(idx_v) > 10: st_f[idx_v] = median_filter(st_f[idx_v], size=5, mode="nearest")
    segs = []; i = 0; n_ = len(st_f)
    while i < n_:                                                   # song_metrics.py と同じ規則：有声ラン ≥0.15 s・跳び >0.8 半音で分割 → 半音に吸着
        if not VO[i]: i += 1; continue
        j = i
        while j + 1 < n_ and VO[j + 1] and abs(np.nanmedian(st_f[max(i, j - 4):j + 1]) - np.nanmedian(st_f[j + 1:j + 6])) <= 0.8: j += 1
        if (j - i + 1) / FR >= 0.15: segs.append((i / FR, (j + 1) / FR, int(round(float(np.nanmedian(st_f[i:j + 1])))), float(np.median(Z["conf"][i:j + 1]))))
        i = j + 1
    notes = np.array([s_[2] for s_ in segs]) if segs else np.array([60])
    ymin, ymax = int(np.percentile(notes, 1)) - 1, int(np.percentile(notes, 99)) + 1
    if ymax - ymin < 12: ymin, ymax = (ymin + ymax) // 2 - 6, (ymin + ymax) // 2 + 6
    # 歌唱音域＝持続音（上の規則・pYIN 確信度の中央値 ≥0.5）の最低〜最高（1〜2 フレームの跳び・確信度の低い区間＝オクターブ誤りや息の音を除く）
    good = [s_[2] for s_ in segs if s_[3] >= 0.5]
    rng_s = f"{nname(min(good))}（{hz(min(good)):.0f} Hz）〜 {nname(max(good))}（{hz(max(good)):.0f} Hz）・幅 {max(good) - min(good)} 半音（持続音 {len(good)} 個）" if good else "—（持続音なし）"
    big_uri = []; big_note = ""
    try:
        ROWS = 10.0; nrow = int(np.ceil(DUR / ROWS)); DPI = 100; W_IN = 15.0; HEAD = 1.3; IFRAC = 0.2
        RNG = ymax - ymin; M = (2.2 + 1.13 * IFRAC * (RNG + 1.5)) / (1 - 1.13 * IFRAC); M = float(np.ceil(M * 10) / 10)   # 上レーンの余白（マークが収まる幅）
        YLO = ymin - 1.5; YHI = ymax + M; ICON_SEMI = IFRAC * (YHI - YLO); LANE_P = ymax + 1.7 + 0.565 * ICON_SEMI
        RH = 0.55 + 0.065 * (YHI - YLO); HSP = 0.3; FIG_H = HEAD + RH * nrow
        AX_H_IN = (RH * nrow - 0.012 * FIG_H) / (nrow + HSP * (nrow - 1)); ICON_PX = IFRAC * AX_H_IN * DPI
        AX_W_IN = W_IN * (0.905 - 0.055); PX_PER_SEC = AX_W_IN * DPI / ROWS
        SX = (ICON_PX / 2) / PX_PER_SEC; SY = ICON_SEMI / 2; band_pt = 0.13 * ICON_PX * 72 / DPI
        fig = plt.figure(figsize=(W_IN, FIG_H), dpi=DPI); fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(nrow, 1, left=0.055, right=0.905, top=1 - HEAD / FIG_H, bottom=0.012, hspace=HSP)
        for r_ in range(nrow):
            ax = fig.add_subplot(gs[r_, 0]); x0, x1 = r_ * ROWS, (r_ + 1) * ROWS
            ax.set_xlim(x0, x1); ax.set_ylim(YLO, YHI); ax.set_facecolor("#fafafa")
            for sp in ax.spines.values(): sp.set_color("#cccccc")
            for q in range(ymin, ymax + 1): ax.axhline(q, color="#e4e4e4" if q % 12 else "#bdbdbd", lw=0.5, zorder=0)
            ticks = list(range(ymin, ymax + 1)); dense = (ymax - ymin) > 18
            ax.set_yticks(ticks); ax.set_yticklabels([(f"{NOTE[q % 12]}{q // 12 - 1}" if (q % 12 in (0, 4, 7) if dense else "#" not in NOTE[q % 12]) else "") for q in ticks], fontsize=6.5, color="#555")
            ax.tick_params(axis="y", length=0); ax.set_xticks(np.arange(x0, x1 + 0.1, 2.0)); ax.tick_params(axis="x", labelsize=7, colors="#555")
            ax.axhline(ymax + 1.2, color="#dddddd", lw=0.6); ax.text(x0 + 0.08, LANE_P, "検出", fontsize=7, color="#777", va="center", ha="left", zorder=1)
            for s0, s1, nt_, _ in segs:
                if s1 < x0 or s0 > x1: continue
                ax.add_patch(Rectangle((s0, nt_ - 0.42), s1 - s0, 0.84, facecolor="#44506b", edgecolor="white", lw=0.4, zorder=3))
            nrow_c = 0
            for k in cands:
                if x0 - 0.5 <= k["t"] <= x1 + 0.5:
                    P = SHAPE[k["c"]](); al = 0.3 + 0.7 * k["score"]; xs = k["t"] + P[:, 0] * SX; ys = LANE_P + P[:, 1] * SY
                    ax.add_line(Line2D(xs, ys, color=ICOL[k["c"]], lw=band_pt, alpha=al, solid_capstyle="round", solid_joinstyle="round", zorder=5))
                    ax.add_line(Line2D(xs, ys, color=ICORE[k["c"]], lw=band_pt * 0.42, alpha=al, solid_capstyle="round", solid_joinstyle="round", zorder=5.1))
                if x0 <= k["t"] < x1: nrow_c += 1
            ax.text(1.008, 0.5, f"候補 {nrow_c}", transform=ax.transAxes, fontsize=8.5, va="center", ha="left", color="#333")
            ax.text(0.0, 1.01, f"{int(x0 // 60)}:{int(x0 % 60):02d} 〜 {int(x1 // 60)}:{int(x1 % 60):02d}", transform=ax.transAxes, fontsize=7.5, color="#777", va="bottom")
        fig.text(0.055, 1 - 0.12 / FIG_H, f"{stem}　{song if entry else ''}", fontsize=15, fontweight="bold", va="top", ha="left")
        fig.text(0.055, 1 - 0.42 / FIG_H, f"検出器 {'+'.join(TAGS)}（確率平均・lag +{a.lag}s）　候補 {len(cands)} 件＝" + "／".join(f"{JA[c]} {cnt.get(c, 0)}" for c in TYPES) + "　上レーン＝検出マーク（濃さ＝確率）・横バー＝歌った音程（半音に吸着）", fontsize=8.5, color="#555", va="top", ha="left")
        axl = fig.add_axes([0.055, 1 - 1.15 / FIG_H, 0.85, 0.5 / FIG_H]); axl.axis("off"); axl.set_xlim(0, 1); axl.set_ylim(-1, 1); xx = 0.0
        lpt = 0.13 * 0.45 * 72; sxl = 0.45 / (0.85 * W_IN) / 2                  # 凡例マーク＝高さ 0.45 in（0.9 単位×0.25 in）に合わせた正方形
        for c in TYPES:
            P = SHAPE[c]()
            axl.add_line(Line2D(xx + sxl + P[:, 0] * sxl, P[:, 1] * 0.9, color=ICOL[c], lw=lpt, solid_capstyle="round", solid_joinstyle="round"))
            axl.add_line(Line2D(xx + sxl + P[:, 0] * sxl, P[:, 1] * 0.9, color=ICORE[c], lw=lpt * 0.42, solid_capstyle="round", solid_joinstyle="round"))
            axl.text(xx + 2.4 * sxl, 0, JA[c], fontsize=9, color=ICOL[c], va="center", ha="left"); xx += 2.4 * sxl + 0.012 * len(JA[c]) + 0.03
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=DPI, facecolor="white"); plt.close(fig); NFIG[0] += 1
        from PIL import Image
        im = Image.open(io.BytesIO(buf.getvalue())); Wpx, Hpx = im.size; STRIP = 2800; strips = []
        for y0 in range(0, Hpx, STRIP):                               # 縦に長い PNG はスマホで復号できないことがあるので ≤2800px の帯に分けて縦に並べる
            b_ = io.BytesIO(); im.crop((0, y0, Wpx, min(Hpx, y0 + STRIP))).save(b_, format="PNG", optimize=True); strips.append("data:image/png;base64," + base64.b64encode(b_.getvalue()).decode("ascii"))
        big_uri = strips; big_note = f"10 秒×{nrow} 段・{Wpx}×{Hpx} px・{len(buf.getvalue()) / 1e6:.1f} MB・{len(strips)} 分割"
    except Exception as ex: big_note = f"生成に失敗（{type(ex).__name__}: {ex}）"
    # (4) 候補ごとの小図＝録音波形（上）と pYIN 音程（下）・−2.5〜+0.5 s・候補時刻＝技法色の実線
    y_w, sr_w = sf.read(a.wav, dtype="float32", always_2d=True); y_w = y_w.mean(1)
    def card_uri(k):
        tc = k["t"]; t0, t1 = tc - 2.5, tc + 0.5; col = ICOL[k["c"]]
        fig = plt.figure(figsize=(6.0, 2.9), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.12, left=0.105, right=0.985, top=0.9, bottom=0.165)
        ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax1)
        i0, i1 = max(0, int(t0 * sr_w)), min(len(y_w), int(t1 * sr_w)); am = 1e-3
        if i1 > i0:
            seg = y_w[i0:i1]; am = max(float(np.abs(seg).max()), 1e-3); nb = 900; m_ = len(seg) // nb
            if m_ >= 2:
                s2 = seg[:m_ * nb].reshape(nb, m_); tt = t0 + (np.arange(nb) + 0.5) * m_ / sr_w; ax1.fill_between(tt, s2.min(1), s2.max(1), color="#44506b", lw=0)
            else: ax1.plot(t0 + np.arange(len(seg)) / sr_w, seg, color="#44506b", lw=0.6)
        ax1.set_ylim(-am * 1.08, am * 1.08); ax1.set_yticks([]); ax1.set_ylabel("波形", fontsize=8, color="#555"); ax1.tick_params(labelbottom=False)
        f0_, f1_ = max(0, int(round(t0 * FR))), min(len(ST), int(round(t1 * FR))); tt = np.arange(f0_, f1_) / FR; s = np.where(VO[f0_:f1_], ST[f0_:f1_], np.nan)
        ax2.plot(tt, s, color=col, lw=1.7, solid_capstyle="round")
        if np.isfinite(s).any():
            lo_, hi_ = float(np.nanmin(s)), float(np.nanmax(s)); c_ = (lo_ + hi_) / 2; half = max(3.5, (hi_ - lo_) / 2 + 1.5); ax2.set_ylim(c_ - half, c_ + half)
            ticks = list(range(int(np.floor(c_ - half)), int(np.ceil(c_ + half)) + 1)); nt = len(ticks)
            show = (lambda q: True) if nt <= 9 else ((lambda q: q % 12 in (0, 2, 4, 5, 7, 9, 11)) if nt <= 14 else ((lambda q: q % 12 in (0, 4, 7)) if nt <= 26 else (lambda q: q % 12 == 0)))
            ax2.set_yticks(ticks); ax2.set_yticklabels([(f"{NOTE[q % 12]}{q // 12 - 1}" if show(q) else "") for q in ticks], fontsize=7, color="#555")
            for q in ticks: ax2.axhline(q, color="#e4e4e4" if q % 12 else "#bdbdbd", lw=0.5, zorder=0)
        else:
            ax2.set_yticks([]); ax2.text(0.5, 0.5, "有声なし（pYIN）", transform=ax2.transAxes, ha="center", va="center", fontsize=9, color="#888")
        ax2.set_ylabel("音程（音名）", fontsize=8, color="#555"); ax2.set_xlabel("録音の秒", fontsize=8, color="#555"); ax2.tick_params(axis="x", labelsize=7.5, colors="#555"); ax2.tick_params(axis="y", length=0)
        for ax in (ax1, ax2):
            ax.set_facecolor("#fafafa")
            for sp in ax.spines.values(): sp.set_color("#cccccc")
            ax.axvline(tc, color=col, lw=1.5, zorder=4)
        ax1.set_title(f"候補 #{k['i']}　{JA[k['c']]}　{tc:.2f} s（確率 {k['score']:.2f}）", fontsize=8.5, color="#333", loc="left", pad=3)
        ax1.set_xlim(t0, t1)
        return b64(fig, dpi=100, facecolor="white")
    for k in cands: k["img"] = card_uri(k)
    # (5) HTML
    def vib_s(v, sep="／"):
        if v is None: return "—"
        dep, rate, dur = v
        return sep.join([("深さ %.0f cent" % dep) if dep is not None else "深さ —", ("速さ %.1f Hz" % rate) if rate is not None else "速さ —", ("持続 %.2f s" % dur) if dur is not None else "持続 —"])
    css = """*{box-sizing:border-box}body{margin:0;padding:0 16px 40px;max-width:1100px;margin-inline:auto;font-family:"Yu Gothic UI","Meiryo","Hiragino Sans",sans-serif;color:#222;background:#fff;line-height:1.5}
h1{font-size:1.5rem;margin:20px 0 4px}h2{font-size:1.1rem;border-left:5px solid #44506b;padding-left:8px;margin:28px 0 10px}.meta,.foot{color:#555;font-size:.88rem}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}th{background:#f2f2f2}
table.kv th{width:38%}td.big{font-size:1.3rem;font-weight:bold}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.ic{display:inline-block;width:22px;height:22px;background-size:contain;background-repeat:no-repeat;vertical-align:middle;margin-right:4px}
.sum{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:10px}.tile{border:1px solid #ddd;border-radius:8px;padding:8px 10px;border-top:4px solid #999}.tile .n{font-size:1.5rem;font-weight:bold;line-height:1.2}.tile .l{font-size:.85rem;color:#555}
.bigw{overflow-x:auto}.bigin{min-width:720px;border:1px solid #ddd}img.big{width:100%;height:auto;display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}.card{border:1px solid #ddd;border-radius:8px;padding:8px;background:#fff;border-top:4px solid #999}
.card img{width:100%;height:auto;display:block;margin-top:6px}.ch{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:.9rem}.ch .ic{width:28px;height:28px;margin:0}
.tn{font-weight:bold}.t{font-variant-numeric:tabular-nums}.pr{color:#555;font-size:.85rem}.vb{font-size:.85rem;color:#333;margin-top:4px}
details{margin-top:14px}summary{cursor:pointer;font-weight:bold;padding:8px;background:#f6f6f6;border-radius:6px}a{color:#2a5bd7;text-decoration:none}
""" + "".join(f".ic-{c}{{background-image:url({ICON[c]})}}.b-{c}{{border-top-color:{ICOL[c]}}}" for c in TYPES)
    def card_html(k):
        vb = f'<div class="vb">{E(vib_s(k["vib"]))}</div>' if k["vib"] is not None else ""
        return (f'<div class="card b-{k["c"]}" id="c{k["i"]}"><div class="ch"><span class="ic ic-{k["c"]}"></span><span class="tn" style="color:{ICOL[k["c"]]}">{JA[k["c"]]}</span>'
                f'<span class="t">#{k["i"]}　{mmss(k["t"])}（{k["t"]:.2f} s）</span><span class="pr">確率 {k["score"]:.2f}</span></div>{vb}<img src="{k["img"]}" alt="{JA[k["c"]]} {k["t"]:.2f}s の波形と音程" loading="lazy"></div>')
    H = [f'<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>1録音レポート {E(stem)}</title><style>{css}</style></head><body>']
    H.append(f'<h1>1録音レポート：{E(stem)}</h1><div class="meta">録音 1 本から検出器で得た独自の分析結果（録音 1 本につき 1 枚）。入力は録音だけ・DAM の採点や正解は使わない。</div>')
    H.append('<h2>1. 録音の情報</h2><div class="tw"><table class="kv">')
    for k_, v_ in [("録音ID", stem), ("曲名（index.json で分かれば）", song), ("録音日時（ファイル名から）", rec_dt), ("長さ", f"{DUR_W:.1f} s（{mmss(DUR_W)}）"),
                   ("サンプリング", f"{info.samplerate} Hz・{info.channels} ch・{info.subtype}"), ("機材", gear), ("検出器", f"学習済み CRNN {'+'.join(TAGS)}（確率平均・lag +{a.lag} s）・技法別しきい値 " + "／".join(f"{JA[c]} {best[c][0]}" for c in TYPES)),
                   ("生成日時", time.strftime("%Y-%m-%d %H:%M") + "　evaluate_recording.py --html")]:
        H.append(f'<tr><th>{E(k_)}</th><td>{E(v_)}</td></tr>')
    H.append('</table></div>')
    H.append('<h2>2. DAM 風の集計（検出器の出力から計算・録音全体）</h2><div class="sum">' + "".join(f'<div class="tile b-{c}"><div class="n"><span class="ic ic-{c}"></span>{cnt.get(c, 0)}</div><div class="l">{JA[c]}の検出回数</div></div>' for c in TYPES) + '</div>')
    H.append('<div class="tw"><table class="kv">')
    nv = cnt.get("vibrato", 0)
    for k_, v_ in [("ビブラートの合計秒", (f"{sum(v_dur):.1f} s（持続が測れた {len(v_dur)} 件の和・範囲外 {nv - len(v_dur)} 件は件数のみ）" if nv else "—（検出なし）")),
                   ("ビブラートの平均深さ", (f"{np.mean(v_dep):.0f} cent（測れた {len(v_dep)} 件の平均・範囲外 {nv - len(v_dep)} 件は件数のみ）" if v_dep else f"—（測れる範囲内の件数 0／{nv}）")),
                   ("ビブラートの平均速さ", (f"{np.mean(v_rate):.1f} Hz（測れた {len(v_rate)} 件の平均・範囲外 {nv - len(v_rate)} 件は件数のみ）" if v_rate else f"—（測れる範囲内の件数 0／{nv}）")),
                   ("歌唱音域（持続音＝pYIN 有声 0.15 s 以上・確信度 0.5 以上 の最低〜最高）", rng_s), ("有声時間の割合", f"{voiced_frac * 100:.1f}%（{voiced_sec:.0f} s／{DUR:.0f} s）")]:
        H.append(f'<tr><th>{E(k_)}</th><td>{E(v_)}</td></tr>')
    H.append('</table></div><div class="foot">ビブラートの中身＝候補時刻の窓 [−1.5, +0.3] s で pYIN 音程の揺れを測る（vibrato_depth.py と同一定義）。「測れる範囲」＝較正で信頼できた帯＝深さは速さ 4〜5 Hz・40 cent 以上・持続 0.6 s 以上／速さは周期的な揺れがあり 4〜6 Hz／持続は速さ 6 Hz 以下・0.6 s 以上。範囲外は「—」。</div>')
    H.append(f'<h2>3. 音程バー図（録音全体・検出マークのみ・{E(big_note)}）</h2>' + (('<div class="bigw"><div class="bigin">' + "".join(f'<img class="big" src="{u}" alt="音程バー図 {i_ + 1}">' for i_, u in enumerate(big_uri)) + '</div></div>') if big_uri else '<p>生成失敗</p>') + '<div class="foot">太い横バー＝歌った音程（半音に吸着・持続音 0.15 s 以上）・上側レーンの図形マーク＝検出した加点候補（濃さ＝確率）。正解レーンは描いていない。スマホでは横にスクロール。</div>')
    H.append(f'<h2>4. 加点候補の一覧（{len(cands)} 件・時刻順）</h2><div class="tw"><table><tr><th>#</th><th>技法</th><th>時刻</th><th>確率</th><th>中身</th></tr>')
    for k in cands: H.append(f'<tr><td class="num"><a href="#c{k["i"]}">{k["i"]}</a></td><td><span class="ic ic-{k["c"]}"></span><span style="color:{ICOL[k["c"]]};font-weight:bold">{JA[k["c"]]}</span></td><td class="num">{mmss(k["t"])}（{k["t"]:.2f} s）</td><td class="num">{k["score"]:.2f}</td><td>{E(vib_s(k["vib"]))}</td></tr>')
    H.append('</table></div><div class="foot">中身＝ビブラートは深さ／速さ／持続。他の技法は現時点では「—」。# を押すと小図へ飛ぶ。</div>')
    NTOP = 30
    H.append(f'<h2>5. 候補ごとの小図（{len(cands)} 件）</h2><div class="foot">各カード＝技法アイコン・時刻・確率・ビブラートは中身・小図＝候補前後 −2.5〜+0.5 s の録音波形（上）と pYIN 音程（下）。縦線＝候補時刻。</div>')
    if len(cands) > 50:
        top = sorted(cands, key=lambda k: -k["score"])[:NTOP]; top_ids = {k["i"] for k in top}; rest = [k for k in cands if k["i"] not in top_ids]
        H.append(f'<p class="meta">確率の高い順に上位 {NTOP} 件を展開。残り {len(rest)} 件（時刻順）は折りたたみ。</p><div class="cards">' + "".join(card_html(k) for k in top) + '</div>')
        H.append(f'<details><summary>残り {len(rest)} 件を開く（時刻順）</summary><div class="cards" style="margin-top:10px">' + "".join(card_html(k) for k in rest) + '</div></details>')
    else: H.append('<div class="cards">' + "".join(card_html(k) for k in cands) + '</div>')
    H.append(f'<p class="foot">生成 {time.strftime("%Y-%m-%d %H:%M")}　evaluate_recording.py --html（1録音レポート＝録音 1 本につき 1 枚）　図 {NFIG[0]} 枚（音程バー 1・技法アイコン {len(TYPES)}・候補の小図 {len(cands)}）。正解との判定・DAM 件数比較は補助レポート（同フォルダの CSV・summary）の仕事。</p></body></html>')
    hpath = f"{a.out}/report_{stem}.html"
    open(hpath, "w", encoding="utf-8").write("".join(H))
    HTML_NOTE = f"- 1録音レポート: {hpath}（{os.path.getsize(hpath) / 1e6:.1f} MB・候補カード {len(cands)} 件・図 {NFIG[0]} 枚・{time.time() - t4:.0f}s・正解と DAM は不使用）"
    say(HTML_NOTE)

# ---------- 9. summary ----------
el = time.time() - T0
say(f"- 所要: 合計 {el:.0f}s（特徴量 {T_FEAT:.0f}s・推論 {T_PRED:.1f}s）  モデル {'+'.join(TAGS)}（確率平均・lag +{a.lag}s）  WAV長 {DUR:.0f}s")
say(f"- 生成: evaluate_recording.py（{time.strftime('%Y-%m-%d %H:%M')}）")
open(f"{a.out}/{name}_summary.md", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
print(f"→ {a.out}/{name}_summary.md")
