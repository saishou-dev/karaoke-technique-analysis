"""vib_100_20261109.py — ビブラートの加点 100件で A＝tc−te（カウンターの時刻−揺れの終わり）を数える（2026-11-09・Opus）

使い方（どれも優先度を下げ・スレッド2で動く）:
  python vib_100_20261109.py f0check            # 6/28 の f0 の時刻ずれ（収録全体の時刻→先頭 0）の検算＝声のある所と wav の音量の相互相関
  python vib_100_20261109.py eyeview            # 基準合わせの 12件の拡大図（区間なし＝目で揺れの始まりと終わりを付ける用）
  python vib_100_20261109.py check <版>         # 12件で取り出し規則の区間と目の区間を比べる（図と表）
  python vib_100_20261109.py run <版>           # 100件（曲ごと最大2件・random_state=0）で A・C・偶然・図・表

入力（読むだけ）:
  g3v2_20260928/out/c2_shift3_global_20261105/labels/index_shift3_global_20261105.json（events＝tc−1.6722。+1.6722 で tc＝orig に戻す）
  g3v2_20260928/index.json（tc の照合だけ）
  dataset/pitch_pyin の f0（index の pitch・8/7 は video 名の時刻で対応）。先頭を 0 に直す（6/28 の f0 は収録全体の時刻）
  marks_20260928/out/mark_pos_time_20261105/D_times_20261105.csv（12件の抜き方だけに使う・前回と同じ）
規則の出発点＝vib_endpoint2_20261105.py の v2（包絡＋その場の帯の割合＋150セントの段の除外）。
"""
import os, sys
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "2"
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass
import json, glob
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic UI", "Yu Gothic", "MS Gothic", "sans-serif"]
ROOT = r"G:\研究アーカイブ\成果物\動画解析"
IDX = os.path.join(ROOT, r"g3v2_20260928\index.json")
IDX_SH = os.path.join(ROOT, r"g3v2_20260928\out\c2_shift3_global_20261105\labels\index_shift3_global_20261105.json")
PDIR = os.path.join(ROOT, r"dataset\pitch_pyin")
DCSV = os.path.join(ROOT, r"marks_20260928\out\mark_pos_time_20261105\D_times_20261105.csv")
OUT = os.path.join(ROOT, r"g3v2_20260928\out\vib_100_20261109")
FIG = r"G:\本人確認待ち\#61_ビブラート100件_2026-11-09"
LAG = 1.6722
PRE, POST = 10.0, 1.0      # 枠＝tc−10秒〜tc+1秒（前回と同じ）
RS = 0
N_MAIN = 100
PER_SONG = 2
TOL = 0.15                 # 目の区間と端の差がこれ以内なら合う
os.makedirs(OUT, exist_ok=True)


# ---------------- f0 の読み込み ----------------
def runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1))


def fill_gaps(c, maxgap=3):
    c = c.copy()
    for i, j in runs(np.isnan(c)):
        if j - i + 1 <= maxgap and i > 0 and j < len(c) - 1:
            c[i:j + 1] = np.linspace(c[i - 1], c[j + 1], j - i + 3)[1:-1]
    return c


def load_f0(path):
    """f0 の時刻は先頭を 0 に直す（6/28 の f0 は収録全体の時刻・ほかの収録は先頭 0 のまま変わらない）。
    lulu 6/28 だけは元の時刻のまま（f0check＝元の時刻で wav の音量と合う・先頭 0 に直すと合わない）"""
    f = pd.read_csv(path, encoding="utf-8-sig")
    t = f.audio_time_sec.values.astype(float)
    if not any(k in os.path.basename(path) for k in T0_KEEP):
        t = t - t[0]
    hz = f.f0_hz.values.astype(float)
    ok = (f.voiced.values > 0) & (hz > 0)
    c = np.full(len(hz), np.nan)
    c[ok] = 1200 * np.log2(hz[ok] / 55.0)
    return t, c


# f0check の結果で決める＝相互相関の山が 0 から外れる曲（f0 と wav が合わない）は使わない
F0_BAD = ["ナハトムジーク_20260628_120104"]   # 先頭 0 でも元の時刻でも山が 0 に来ない（−0.79秒・r 0.17）
T0_KEEP = ["lulu._20260628_094340"]           # 元の時刻のままで山が −0.01秒（r 0.29）


def songs():
    sh = json.load(open(IDX_SH, encoding="utf-8"))
    idx = json.load(open(IDX, encoding="utf-8"))
    fs87 = glob.glob(os.path.join(PDIR, "*_20260807_*_f0.csv"))
    out = []
    for si, (s, s0) in enumerate(zip(sh, idx)):
        assert s["video"] == s0["video"]
        p = s.get("pitch"); src = "index"
        if not p or not os.path.exists(p):
            p = None; src = ""
            if s["session"] == "20260807":
                tm = s["video"].split("_")[-1]
                m = [f for f in fs87 if os.path.basename(f).endswith(f"_20260807_{tm}_f0.csv")]
                if len(m) == 1:
                    p, src = m[0], "8/7対応"
        if s["video"] in F0_BAD:
            p, src = None, "f0とwavが合わない"
        ev = []
        for (c, t1), (c0, t0) in zip(s["events"], s0["events"]):
            assert c == c0
            tc = round(float(t1) + LAG, 4)
            assert abs(tc - float(t0)) < 0.002, (s["video"], tc, t0)
            ev.append((c, tc))
        out.append(dict(si=si, video=s["video"], session=s["session"], offset=float(s["offset"]), pitch=p, src=src, events=ev, wav=s["wav"]))
    return out


# ---------------- 揺れの区間（版） ----------------
HOP = 0.01
VER = dict(
    # v2＝vib_endpoint2 の v2 そのまま
    v2=dict(med=51, band=(3.5, 8.0), core=20.0, ext=10.0, loc_core=0.5, loc_ext=0.3, step=150.0, win=21, gap=0.15, minlen=0.3),
    # v3＝包絡をやめ、山谷の続きで拾う（目の基準と同じ形）
    v3=dict(mode="zz", sm=3, h=15, pp=30, ppmax=250, half=(0.06, 0.30), nmin=4, cv=0.5, tail=0.10),
    # v4＝遅い動き（1.5Hz 以下）を引いてから山谷・差 150セント超（音の移り）で切る・山谷 5個以上（2周期）・間隔 0.4秒まで（遅い揺れ）
    v4=dict(mode="zz", sm=3, h=15, lp=1.5, pp=30, ppmax=150, half=(0.06, 0.40), nmin=5, cv=0.5, tail=0.10),
    # v5＝目の基準（2回目）をそのまま規則に＝f0−0.5秒移動中央値の上で山谷（戻り 30セント以上）・間隔 0.06〜0.30秒・山谷 4個以上・差 200セント超（音の移り）で切る
    v5=dict(mode="zz", sm=3, h=30, res=51, pp=30, ppmax=200, half=(0.06, 0.30), nmin=4, cv=9.0, tail=0.0),
    # v6＝v5 に「f0 そのものも同じ向きに 20セント以上動く」と「声の終わりの確定していない山谷は数えない」を足す
    v6=dict(mode="zz", sm=3, h=30, res=51, pp=30, ppmax=200, half=(0.06, 0.30), nmin=4, cv=9.0, tail=0.0, f0rev=20, last=False),
    # v7＝v6 に目の基準の残り2つを足す＝f0 の差 150セント超（音の移り）で切る・声の終わりの最後の山谷も数える
    v7=dict(mode="zz", sm=3, h=30, res=51, pp=30, ppmax=200, half=(0.06, 0.30), nmin=4, cv=9.0, tail=0.0, f0rev=20, f0max=150, last=True),
)


def zigzag(x, h, last=True):
    """山谷を交互に拾う（戻りが h セント以上で向きが変わったとみなす）。返す＝[(添字, +1 山 / −1 谷)]。最後の山谷（確定前）も入れる"""
    out = []
    if len(x) < 3:
        return out
    d = 0; imax = imin = 0
    for i in range(1, len(x)):
        if d == 0:
            if x[i] > x[imax]: imax = i
            if x[i] < x[imin]: imin = i
            if x[i] > x[imin] + h:
                out.append((imin, -1)); d = 1; imax = i
            elif x[i] < x[imax] - h:
                out.append((imax, 1)); d = -1; imin = i
        elif d == 1:
            if x[i] > x[imax]: imax = i
            elif x[i] < x[imax] - h:
                out.append((imax, 1)); d = -1; imin = i
        else:
            if x[i] < x[imin]: imin = i
            elif x[i] > x[imin] + h:
                out.append((imin, -1)); d = 1; imax = i
    if last:
        if d == 1: out.append((imax, 1))
        elif d == -1: out.append((imin, -1))
    return out


def zz_segments(t, c, P):
    """山谷の続き＝隣の山谷の差 pp〜ppmax セント・間隔 half 秒・山谷 nmin 個以上・間隔のばらつき（標準偏差÷平均）cv 以下。
    区間＝最初の山谷〜最後の山谷（声の切れ目までが tail 秒以内なら声の切れ目まで）"""
    cf = fill_gaps(c)
    xs = pd.Series(cf).rolling(P.get("sm", 3), center=True, min_periods=1).mean().to_numpy(copy=True)
    xs[np.isnan(cf)] = np.nan
    if P.get("res"):
        med = pd.Series(cf).rolling(P["res"], center=True, min_periods=P["res"] // 2).median().values
        xs = xs - med
    if P.get("lp"):
        # 声の続きごとに、遅い動き（lp Hz 以下・往復のバターワース 2次）を引く＝上がり下がりしながらの揺れも中心の周りの揺れにする
        sos = butter(2, P["lp"], btype="low", fs=1 / HOP, output="sos")
        for i, j in runs(~np.isnan(xs)):
            seg = xs[i:j + 1]
            if len(seg) > 12:
                xs[i:j + 1] = seg - sosfiltfilt(sos, seg, padlen=min(len(seg) - 1, 50))
            else:
                xs[i:j + 1] = seg - np.mean(seg)
    segs = []
    f0s = pd.Series(cf).rolling(P.get("sm", 3), center=True, min_periods=1).mean().values
    for i, j in runs(~np.isnan(xs)):
        if (j - i + 1) * HOP < 0.25:
            continue
        x = xs[i:j + 1]; tt = t[i:j + 1]; fx = f0s[i:j + 1]
        ex = zigzag(x, P["h"], last=P.get("last", True))
        if len(ex) < P["nmin"]:
            continue
        ch = [ex[0]]
        def flush(ch):
            if len(ch) >= P["nmin"]:
                iv = np.diff([tt[k] for k, _ in ch])
                if iv.std() / iv.mean() <= P["cv"]:
                    a = tt[ch[0][0]]; b = tt[ch[-1][0]]
                    if tt[-1] - b <= P.get("tail", 0):
                        b = tt[-1]
                    if P.get("pad"):
                        a -= P["pad"] * iv[0]; b += P["pad"] * iv[-1]
                    segs.append((max(a, tt[0]), min(b, tt[-1])))
        for k in range(1, len(ex)):
            (k0, _), (k1, _) = ex[k - 1], ex[k]
            sw = abs(x[k1] - x[k0]); dt = tt[k1] - tt[k0]
            okf = True
            if P.get("f0rev"):
                # f0 そのものも同じ向きに f0rev セント以上動く（中央値を引いた残りだけの見かけの山谷＝音の段の前後を外す）
                df_ = fx[k1] - fx[k0]
                okf = np.sign(df_) == np.sign(x[k1] - x[k0]) and abs(df_) >= P["f0rev"] and abs(df_) <= P.get("f0max", 1e9)
            if P["pp"] <= sw <= P["ppmax"] and P["half"][0] <= dt <= P["half"][1] and okf:
                ch.append(ex[k])
            else:
                flush(ch); ch = [ex[k]]
        flush(ch)
    return segs


def vib_segments(t, c, ver):
    P = VER[ver]
    if P.get("mode") == "zz":
        return zz_segments(t, c, P), {}
    cf = fill_gaps(c)
    med = pd.Series(cf).rolling(P["med"], center=True, min_periods=P["med"] // 2).median().values
    r = cf - med
    voiced = ~np.isnan(r)
    r0 = np.where(voiced, r, 0.0)
    sos = butter(4, list(P["band"]), btype="band", fs=1 / HOP, output="sos")
    bp = sosfiltfilt(sos, r0)
    env = pd.Series(np.abs(hilbert(bp))).rolling(5, center=True, min_periods=1).mean().to_numpy(copy=True)
    env[~voiced] = 0.0
    w = P["win"]
    num = pd.Series(bp ** 2).rolling(w, center=True, min_periods=1).mean().values
    den = pd.Series(r0 ** 2).rolling(w, center=True, min_periods=1).mean().values
    loc = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    big = pd.Series(np.abs(r0) > P["step"]).rolling(w, center=True, min_periods=1).max().values > 0
    core = (env >= P["core"]) & (loc >= P["loc_core"]) & ~big
    ext = (env >= P["ext"]) & (loc >= P["loc_ext"]) & ~big
    segs = []
    for i, j in runs(ext):
        if core[i:j + 1].any():
            segs.append([t[i], t[j]])
    mg = []
    for a, b in segs:
        if mg and a - mg[-1][1] <= P["gap"] + 1e-9:
            mg[-1][1] = b
        else:
            mg.append([a, b])
    segs = [(a, b) for a, b in mg if b - a >= P["minlen"]]
    if P.get("post"):
        segs = P["post"](t, cf, r0, segs, P)
    return segs, dict(r0=r0, bp=bp, env=env)


def match(tc, segs):
    """前回と同じ：枠 tc−10〜tc+1 に重なる区間のうち、tc 以前に終わり te が tc に最も近い物（始まりは tc−10 で切る）"""
    win = [(a, b) for a, b in segs if b >= tc - PRE and a <= tc + POST]
    contain = sum(1 for a, b in win if a <= tc < b)
    cand = [(a, b) for a, b in win if b <= tc]
    if not cand:
        return None, len(win), contain
    a, b = max(cand, key=lambda x: x[1])
    return (max(a, tc - PRE), b), len(win), contain


# ---------------- 12件（前回と同じ抜き方） ----------------
def cal_cases():
    """vib_endpoint2 の check_cases と同じ＝vibrato の行を曲の順・events の順に並べ、D を付け、
    D がある行から a_counter_minus_D 2.5秒超 6件・以下 6件（pandas sample random_state=0）"""
    S = songs()
    rows = []
    for s in S:
        vib = [tc for c, tc in s["events"] if c == "vibrato"]
        for k, tc in enumerate(vib):
            rows.append(dict(song=s["si"], video=s["video"], session=s["session"], k=k, tc=tc, offset=s["offset"]))
    res = pd.DataFrame(rows)
    d = pd.read_csv(DCSV, encoding="utf-8-sig")
    d = d[d.counter == "vibrato"].copy(); d["session"] = d.session.astype(str)
    res["D"] = np.nan; res["a_cmD"] = np.nan
    for _, e in d.iterrows():
        sel = res[(res.session == e.session) & (res.video == e.video)]
        if len(sel) == 0:
            continue
        tw = float(e.video_time_sec) + float(sel.offset.iloc[0])
        j = (sel.tc - tw).abs()
        if j.min() > 0.01:
            continue
        if pd.notna(e.D):
            res.loc[j.idxmin(), "D"] = float(e.D) + float(sel.offset.iloc[0])
            res.loc[j.idxmin(), "a_cmD"] = float(e.a_counter_minus_D)
    dd = res[res.D.notna()]
    a = dd[dd.a_cmD > 2.5].sample(6, random_state=RS)
    b = dd[dd.a_cmD <= 2.5].sample(6, random_state=RS)
    cs = pd.concat([a.assign(grp="2.5秒超"), b.assign(grp="2.5秒以下")]).reset_index(drop=True)
    cs.index = cs.index + 1
    return cs, S, len(dd)


# 目の判定 1回目（Opus・eye/ の図＝1段 2.8秒で見た）。縦の幅が広く浅い揺れの読みが揃わなかったため使わない（記録だけ）
EYE1 = {1: [(275.95, 277.85), (278.00, 279.68)], 2: [(241.95, 242.70)], 3: [], 4: [(162.15, 163.05), (164.80, 165.25)],
        5: [(208.85, 209.50)], 6: [(208.85, 209.50)], 7: [(178.95, 179.75)], 8: [(190.33, 191.10), (192.50, 193.25)],
        9: [(46.60, 47.10)], 10: [], 11: [(172.40, 172.85)], 12: [(80.95, 81.85)]}
# 目の判定 2回目（Opus・eye2/ の図＝1段 1.87秒・上 f0・下 f0−0.5秒移動中央値で見た）。基準を先に文で決めた：
#   揺れ＝山と谷が交互に 4個以上（戻り 30セント以上を山谷に数える）・隣の山谷の間隔 0.06〜0.30秒・音の移り（f0 の段）は揺れに数えない・
#   上がり下がりしながらの上下も揺れに入れる。始まり＝最初の山谷・終わり＝最後の山谷。
#   tc に最も近い物（tc 以前に終わる）だけ使うため、tc から前へ見て最初に当たった揺れを付けた（[ ] の後ろの物は参考）。
EYE = {1: [(276.07, 277.72), (277.95, 279.68)],
       2: [(242.05, 242.62)],
       3: [],
       4: [(164.85, 165.10), (167.50, 167.90)],
       5: [(209.00, 209.50), (213.15, 213.60)],
       6: [(209.00, 209.50)],
       7: [(178.92, 179.75)],                       # 下がりながらの上下
       8: [(190.52, 191.07)],
       9: [(46.60, 47.05)],                         # 上がりながらの上下
       10: [],
       11: [],
       12: []}


def eye_match(n, tc):
    segs = EYE.get(n, [])
    return match(tc, segs)[0]


def judge(m, me):
    if m is None and me is None:
        return "合う（どちらも区間なし）"
    if m is None:
        return "合わない＝規則が取れず"
    if me is None:
        return "合わない＝目は揺れなし"
    ds, de = m[0] - me[0], m[1] - me[1]
    if abs(ds) <= TOL + 1e-6 and abs(de) <= TOL + 1e-6:
        return "合う"
    return f"合わない＝始まり{ds:+.2f}・終わり{de:+.2f}"


# ---------------- f0 の時刻の検算 ----------------
def wav_db(path, H=0.01):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = int(len(x) / sr / H); L = int(round(H * sr))
    x = x[:n * L].reshape(n, L)
    return 10 * np.log10((x ** 2).mean(axis=1) + 1e-10), H


def f0check():
    sh = json.load(open(IDX_SH, encoding="utf-8"))
    rows = []
    keep = {}
    for s in sh:
        if s["session"] != "20260628":
            continue
        f = pd.read_csv(s["pitch"], encoding="utf-8-sig")
        traw = f.audio_time_sec.values.astype(float)
        t = traw - traw[0]
        v = ((f.voiced.values > 0) & (f.f0_hz.values > 0)).astype(float)
        db, H = wav_db(s["wav"])
        n = len(db)
        vg = np.zeros(n); ii = np.round(t / H).astype(int); ok = (ii >= 0) & (ii < n); vg[ii[ok]] = v[ok]
        A = (vg - vg.mean()) / vg.std(); B = (db - db.mean()) / db.std()
        cc = {}
        for L in range(-1500, 1501):   # f0 を L フレーム後ろへずらした時の相関
            cc[L] = np.mean(A[:n - L] * B[L:]) if L >= 0 else np.mean(A[-L:] * B[:n + L])
        Lb = max(cc, key=cc.get)
        # 直す前（収録全体の時刻のまま）の相関
        vr = np.zeros(n); ir = np.round(traw / H).astype(int); okr = (ir >= 0) & (ir < n); vr[ir[okr]] = v[okr]
        r_raw = np.corrcoef(vr, db)[0, 1] if vr.std() > 0 else np.nan
        nv = sum(1 for e in s["events"] if e[0] == "vibrato")
        rows.append(dict(video=s["video"], f0_first=traw[0], f0_dur=t[-1], wav_dur=n * H, best_shift_sec=Lb * H, r_best=cc[Lb], r_at0=cc[0],
                         r_raw_time=r_raw, n_vibrato=nv, use=s["video"] not in F0_BAD))
        keep[s["video"]] = (t, v, traw, db, H)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "f0check_20261109.csv"), index=False, encoding="utf-8-sig")
    print(df.round(3).to_string())
    # 2曲の図（直した f0 の声のある所・直す前・wav の音量）
    pick = [v for v in df.video if v.startswith("ANSWER")] + [v for v in df.video if "114948" in v]
    fig, axs = plt.subplots(len(pick), 1, figsize=(18, 4.2 * len(pick)), facecolor="white", squeeze=False)
    for ax, vname in zip(axs[:, 0], pick):
        t, v, traw, db, H = keep[vname]
        tw = np.arange(len(db)) * H
        lo, hi = 30, 75
        sel = (tw >= lo) & (tw <= hi)
        ax.plot(tw[sel], db[sel], color="#6c757d", lw=0.8, label="wav の音量（dB）")
        y0, y1 = np.nanmin(db[sel]), np.nanmax(db[sel])
        s1 = (t >= lo) & (t <= hi)
        ax.fill_between(t[s1], y1 + 3, y1 + 3 + 6 * v[s1], color="#2a9d8f", step="mid", label="声あり（直した f0・先頭 0）")
        s2 = (traw >= lo) & (traw <= hi)
        ax.fill_between(traw[s2], y1 + 11, y1 + 11 + 6 * v[s2], color="#e76f51", step="mid", label="声あり（直す前の時刻）")
        r = df[df.video == vname].iloc[0]
        ax.set_title(f"{vname}  f0 の先頭 {r.f0_first:.2f}秒→0  相互相関の山 {r.best_shift_sec:+.2f}秒（r={r.r_best:.2f}）", fontsize=15)
        ax.set_xlim(lo, hi); ax.legend(fontsize=12, loc="lower right"); ax.grid(alpha=.3); ax.tick_params(labelsize=12)
        ax.set_xlabel("録音の時刻（秒）", fontsize=13)
    fig.tight_layout()
    fp = os.path.join(OUT, "f0check_20261109.png"); fig.savefig(fp, dpi=70, facecolor="white"); plt.close(fig); print(fp)


# ---------------- 目で見る用の図 ----------------
def eyeview(ver=None, tag="eye"):
    cs, S, nD = cal_cases()
    d = os.path.join(OUT, "check", tag); os.makedirs(d, exist_ok=True)
    for n, e in cs.iterrows():
        s = S[int(e.song)]
        t, c = load_f0(s["pitch"])
        segs = vib_segments(t, c, ver)[0] if ver else []
        cf = fill_gaps(c)
        med = pd.Series(cf).rolling(51, center=True, min_periods=25).median().values
        r = cf - med
        lo0, hi0 = e.tc - PRE - 0.2, e.tc + POST
        parts = 4; L = (hi0 - lo0) / parts
        fig = plt.figure(figsize=(20, 5.2 * parts), facecolor="white")
        gs = fig.add_gridspec(2 * parts, 1, height_ratios=[2.2, 1] * parts, hspace=0.35)
        for p in range(parts):
            ax = fig.add_subplot(gs[2 * p]); ax2 = fig.add_subplot(gs[2 * p + 1], sharex=ax)
            lo, hi = lo0 + p * L, lo0 + (p + 1) * L
            sel = (t >= lo) & (t <= hi)
            ax.plot(t[sel], c[sel], color="black", lw=1.0, marker=".", ms=3)
            ax2.plot(t[sel], np.clip(r[sel], -120, 120), color="black", lw=1.0, marker=".", ms=2)
            ax2.axhline(0, color="gray", lw=.6); ax2.set_ylim(-120, 120); ax2.set_yticks([-100, -50, 0, 50, 100])
            if np.isfinite(c[sel]).any():
                ax.set_ylim(np.nanmin(c[sel]) - 40, np.nanmax(c[sel]) + 40)
            for a, b in segs:
                if b >= lo and a <= hi:
                    ax.axvspan(a, b, color="#8ecae6", alpha=.45, lw=0); ax2.axvspan(a, b, color="#8ecae6", alpha=.45, lw=0)
            for a, b in EYE.get(n, []):
                if b >= lo and a <= hi:
                    ax2.plot([a, b], [-105] * 2, color="#e85d04", lw=6)
            for x_ in (ax, ax2):
                if lo <= e.tc <= hi:
                    x_.axvline(e.tc, color="#d62728", lw=2)
                x_.set_xlim(lo, hi)
                x_.set_xticks(np.arange(np.ceil(lo * 2) / 2, hi, 0.5))
                x_.set_xticks(np.arange(np.ceil(lo * 10) / 10, hi, 0.1), minor=True)
                x_.grid(which="major", alpha=.5); x_.grid(which="minor", alpha=.2)
                x_.tick_params(labelsize=12)
        fig.axes[0].set_title(f"例{n} {e.grp} {e.video} tc={e.tc:.2f}（赤）  下段＝f0−0.5秒移動中央値（セント）" + (f"  水色＝規則 {ver}" if ver else "") + ("  橙＝目" if EYE.get(n) else ""), fontsize=15)
        fp = os.path.join(d, f"{n:02d}.png"); fig.savefig(fp, dpi=58, facecolor="white", bbox_inches="tight"); plt.close(fig)
    cs.to_csv(os.path.join(OUT, "check", "cases12_20261109.csv"), encoding="utf-8-sig")
    print(cs[["grp", "video", "tc", "D", "a_cmD"]].to_string()); print("D あり", nD)


def check(ver):
    cs, S, nD = cal_cases()
    rows = []
    for n, e in cs.iterrows():
        s = S[int(e.song)]
        t, c = load_f0(s["pitch"])
        segs = vib_segments(t, c, ver)[0]
        m = match(e.tc, segs)[0]
        me = eye_match(n, e.tc)
        rows.append(dict(no=n, grp=e.grp, video=e.video, tc=e.tc, rule_ts=None if m is None else round(m[0], 2), rule_te=None if m is None else round(m[1], 2),
                         eye_ts=None if me is None else round(me[0], 2), eye_te=None if me is None else round(me[1], 2), judge=judge(m, me),
                         segs=[(round(a, 2), round(b, 2)) for a, b in segs if b >= e.tc - PRE and a <= e.tc + POST]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "check", f"check_{ver}_20261109.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(df.to_string())
    print(ver, "合う", int(df.judge.str.startswith("合う").sum()), "/ 12")
    eyeview(ver, tag=f"cmp_{ver}")
    return df


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "f0check":
        f0check()
    elif mode == "eyeview":
        eyeview()
    elif mode == "check":
        check(sys.argv[2])


# ---------------- 12件のまとめ（全版） ----------------
VERS = ["v2", "v3", "v4", "v5", "v6", "v7"]


def check_all():
    cs, S, nD = cal_cases()
    rows = []
    for n, e in cs.iterrows():
        t, c = load_f0(S[int(e.song)]["pitch"])
        me = eye_match(n, e.tc)
        r = dict(no=n, grp=e.grp, video=e.video, tc=e.tc, eye_ts=None if me is None else me[0], eye_te=None if me is None else me[1],
                 A_eye=None if me is None else e.tc - me[1], C_eye=None if me is None else e.tc - me[0])
        for v in VERS:
            m = match(e.tc, vib_segments(t, c, v)[0])[0]
            r[f"{v}_ts"] = None if m is None else m[0]; r[f"{v}_te"] = None if m is None else m[1]
            r[f"{v}_judge"] = judge(m, me)
            # 参考の緩い合い方＝終わりの差 0.3秒以内（どちらも区間なしも含む）
            r[f"{v}_loose"] = int((m is None and me is None) or (m is not None and me is not None and abs(m[1] - me[1]) <= 0.3))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "check", "check_all_20261109.csv"), index=False, encoding="utf-8-sig")
    summ = {v: dict(strict=int(df[f"{v}_judge"].str.startswith("合う").sum()), loose=int(df[f"{v}_loose"].sum())) for v in VERS}
    print(summ)
    return df, summ


# ---------------- 100件 ----------------
def population():
    S = songs()
    rows = []
    for s in S:
        vib = [tc for c, tc in s["events"] if c == "vibrato"]
        for k, tc in enumerate(vib):
            rows.append(dict(song=s["si"], video=s["video"], session=s["session"], k=k, tc=tc, pitch=s["pitch"], src=s["src"]))
    return pd.DataFrame(rows), S


def sample100(pop):
    use = pop[pop.pitch.notna()]
    per = pd.concat([g.sample(min(PER_SONG, len(g)), random_state=RS) for _, g in use.groupby("song")])
    return per.sample(N_MAIN, random_state=RS).sort_values(["song", "tc"]), len(use), len(per)


def stat(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return dict(n=0)
    q1, q2, q3 = x.quantile([.25, .5, .75])
    return dict(n=len(x), med=q2, q1=q1, q3=q3, iqr=q3 - q1, le05=float((x <= 0.5).mean()), inmed05=float(((x - q2).abs() <= 0.5).mean()))


def run(ver, nrand=10):
    pop, S = population()
    main, n_use, n_per = sample100(pop)
    cache = {}
    rng = np.random.default_rng(RS)
    rows, base = [], []
    for _, e in main.iterrows():
        if e.song not in cache:
            t, c = load_f0(e.pitch)
            cache[e.song] = (t, vib_segments(t, c, ver)[0])
        t, segs = cache[e.song]
        m, nwin, ncon = match(e.tc, segs)
        r = dict(e); r.pop("pitch")
        r.update(n_win=nwin, tc_in_seg=ncon)
        if e.tc - PRE < t[0] - 1e-6 or e.tc > t[-1] + 1e-6:
            r["why"] = "f0 の外"
        elif m is None:
            r["why"] = "枠に区間なし" if nwin == 0 else "tc以前に終わる区間なし"
        else:
            r.update(why="", ts=m[0], te=m[1])
        rows.append(r)
        for _ in range(nrand):
            tr = rng.uniform(t[0] + PRE, t[-1] - POST)
            mb = match(tr, segs)[0]
            base.append(dict(song=e.song, tr=tr, A=np.nan if mb is None else tr - mb[1], C=np.nan if mb is None else tr - mb[0]))
    res = pd.DataFrame(rows)
    for col in ("ts", "te"):
        if col not in res: res[col] = np.nan
    res["A"] = res.tc - res.te; res["C"] = res.tc - res.ts; res["L"] = res.te - res.ts
    bl = pd.DataFrame(base)
    res.to_csv(os.path.join(OUT, f"vib100_{ver}_20261109.csv"), index=False, encoding="utf-8-sig")
    bl.to_csv(os.path.join(OUT, f"vib100_rand_{ver}_20261109.csv"), index=False, encoding="utf-8-sig")
    out = dict(ver=ver, n_pop=len(pop), n_use=n_use, n_per=n_per, n_main=len(res), n_songs=res.song.nunique(),
               n_0628=int((res.session == "20260628").sum()), n_0807=int((res.session == "20260807").sum()),
               nopick=int(res.te.isna().sum()), why=res.why.value_counts().to_dict(), tc_in=int((res.tc_in_seg > 0).sum()),
               A=stat(res.A), C=stat(res.C), L=stat(res.L), rA=stat(bl.A), rC=stat(bl.C), r_nopick=float(bl.A.isna().mean()), n_rand=len(bl))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    json.dump(out, open(os.path.join(OUT, f"summary_{ver}_20261109.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    return res, bl, out


if __name__ == "__main__" and sys.argv[1] in ("checkall", "run"):
    if sys.argv[1] == "checkall":
        check_all()
    else:
        run(sys.argv[2])


# ---------------- 図（#61） ----------------
def figs(ver="v7"):
    os.makedirs(FIG, exist_ok=True)
    res = pd.read_csv(os.path.join(OUT, f"vib100_{ver}_20261109.csv"), encoding="utf-8-sig")
    bl = pd.read_csv(os.path.join(OUT, f"vib100_rand_{ver}_20261109.csv"), encoding="utf-8-sig")
    A = res.A.dropna(); R = bl.A.dropna()
    bins = np.arange(0, 10.01, 0.25)
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")
    ax.hist(A, bins=bins, color="#2a9d8f", alpha=.85, label=f"加点の時刻（{len(A)}件・取れず {res.A.isna().sum()}件）")
    h, _ = np.histogram(R, bins=bins)
    ax.step(bins[:-1], h * len(A) / len(R), where="post", color="#6c757d", lw=2, label=f"偶然＝同じ曲の一様乱数の時刻（{len(R)}点・{len(A)}件に縮めた）")
    ax.axvline(0.5, color="#e76f51", ls="--", lw=1.5, label="0.5秒")
    ax.axvline(LAG, color="#7b2cbf", ls=":", lw=2, label="今の遅れ 1.67秒")
    ax.axvline(A.median(), color="black", lw=2, label=f"中央 {A.median():.2f}秒")
    ax.set_xlabel("A＝カウンターの時刻 − 揺れの終わり（秒）", fontsize=14); ax.set_ylabel("件数（0.25秒ごと）", fontsize=14)
    ax.set_title(f"ビブラート 100件の A の分布（規則 {ver}・基準合わせは 12件中 2件＝未通過・参考）", fontsize=15)
    ax.legend(fontsize=12); ax.grid(alpha=.3); ax.tick_params(labelsize=12); ax.set_xlim(0, 10)
    fig.tight_layout(); fp = os.path.join(FIG, "#61-1_Aの分布_100件.png"); fig.savefig(fp, dpi=80, facecolor="white"); plt.close(fig); print(fp)
    # 例（基準合わせの 12件から）
    cs, S, _ = cal_cases()
    ex = [(1, "合う例1", "#61-2_合う例1.png"), (4, "合う例2", "#61-3_合う例2.png"),
          (3, "合わない例1（規則は揺れを拾う・目は揺れなし）", "#61-4_合わない例1.png"),
          (7, "合わない例2（下がりながらの揺れを規則が取れず・手前の別の揺れに付く）", "#61-5_合わない例2.png")]
    for n, lab, nm in ex:
        e = cs.loc[n]
        t, c = load_f0(S[int(e.song)]["pitch"])
        segs = vib_segments(t, c, ver)[0]
        m = match(e.tc, segs)[0]; me = eye_match(n, e.tc)
        cf = fill_gaps(c); med = pd.Series(cf).rolling(51, center=True, min_periods=25).median().values; r = cf - med
        lo = min([e.tc - 4.0] + [x[0] - 0.6 for x in (m, me) if x is not None]); hi = e.tc + 0.6
        sel = (t >= lo) & (t <= hi)
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.1]), facecolor="white")
        ax.plot(t[sel], c[sel], color="black", lw=1.1, marker=".", ms=3)
        ax2.plot(t[sel], np.clip(r[sel], -150, 150), color="black", lw=.9, marker=".", ms=2); ax2.set_ylim(-150, 150); ax2.axhline(0, color="gray", lw=.6)
        for a, b in segs:
            if b >= lo and a <= hi:
                for x_ in (ax, ax2): x_.axvspan(a, b, color="#8ecae6", alpha=.45, lw=0)
        yl = ax.get_ylim()
        if me is not None:
            ax.plot([me[0], me[1]], [yl[0] + 0.03 * (yl[1] - yl[0])] * 2, color="#e85d04", lw=8, solid_capstyle="butt", label="目の揺れ（tc に最も近い物）")
        if m is not None:
            for x_ in (m[0], m[1]):
                ax.axvline(x_, color="#023e8a", ls="--", lw=1.5)
            ax.plot([], [], color="#023e8a", ls="--", label="規則の区間の端（tc に最も近い物）")
        ax.axvline(e.tc, color="#d62728", lw=2.2, label="カウンターの時刻 tc")
        ax.axvline(e.tc - LAG, color="#7b2cbf", ls=":", lw=2, label="tc−1.67秒")
        ax.fill_between([], [], color="#8ecae6", alpha=.45, label=f"規則 {ver} の揺れの区間")
        ax.set_ylabel("声の高さ（セント）", fontsize=13); ax2.set_ylabel("f0−0.5秒中央値", fontsize=12)
        ax2.set_xlabel("録音の時刻（秒）", fontsize=13)
        for x_ in (ax, ax2):
            x_.set_xticks(np.arange(np.ceil(lo * 10) / 10, hi, 0.1), minor=True); x_.grid(which="major", alpha=.4); x_.grid(which="minor", alpha=.12); x_.tick_params(labelsize=12)
        Ae = "—" if me is None else f"{e.tc - me[1]:.2f}"; Ar = "—" if m is None else f"{e.tc - m[1]:.2f}"
        ax.set_title(f"{lab}  基準合わせ 例{n}  {e.video}  tc={e.tc:.2f}秒\n目の A＝{Ae}秒  規則の A＝{Ar}秒  判定＝{judge(m, me)}", fontsize=14)
        ax.legend(fontsize=11, loc="best", ncol=2, framealpha=.9)
        fig.tight_layout(); fp = os.path.join(FIG, nm); fig.savefig(fp, dpi=75, facecolor="white"); plt.close(fig); print(fp)


if __name__ == "__main__" and sys.argv[1] == "figs":
    figs(sys.argv[2] if len(sys.argv) > 2 else "v7")
