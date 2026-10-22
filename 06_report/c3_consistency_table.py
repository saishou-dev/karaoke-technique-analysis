# -*- coding: utf-8 -*-
r"""C3（曲単位の整合性検証）の整合表 v1 を作る（2026-10-23）。

主張 C3 ＝「1 本の録音から測った音響特徴量を録音単位でまとめ直すと、同じ歌唱に対する
DAM 精密採点の曲単位の集計値（回数・合計秒・音域）と対応する」。
証拠の形 ＝ 同じ意味を持つ 3 系列を 1 行 1 録音で並べ、相関と差の分布を出した表。
  (1) DAM 側  … 精密採点 Ai Hearts の API 出力（曲単位の集計値）
  (2) 測定側  … 録音 WAV の基本周波数（F0・100 fps）から測った値
  (3) 正解側  … 画面の加点回数表示から作った技法イベント正解データ（G1）の件数

入力:
  DAM      G:/研究アーカイブ/成果物/dam_api/dam_ai-hearts_full_2026-10-21T10-50-11.csv
  ピッチ    g3v2_20260928/feats/<video>.npz（〜2026-10-14）／c3_consistency_20261023/feats_<session>/<wav>.npz（10-21 以降）
  収録一覧  g3v2_20260928/index.json（video・span）／G1_events_<session>_wao.csv（10-21 以降）
  正解      dataset/G1_events_all.csv（〜2026-10-14）／dataset/G1_events_20261021_wao.csv
  参照      song_metrics の 歌声分析_<session>.csv（既存の音響特徴量・突合用／任意）

出力（--out 既定 = G:/研究アーカイブ/成果物/動画解析/c3_consistency_20261023）:
  c3_consistency_songs.csv   1 行 1 録音の全列
  c3_consistency_items.csv   項目ごとの n・相関・差の中央値・外れ録音
  c3_consistency_stats.json  上記の素の値
  c3_consistency.md          表と数値の本文

ビブラートの測定の枠組み（vibrato_depth.py / c3_fullspan.py と同一）:
  セント列 = F0 を MIDI 半音×100 に直し、有声区間ごとに 0.3 s の移動中央値（旋律の土台）を引いた残差。
  極値列   = 残差の山と谷を交互に取る。
  連鎖     = 隣り合う山谷の差 >= dmin cent かつ 半周期 ∈ [1/(2*fmax), 1/(2*fmin)] s を満たす対が
             min_pairs 組以上続く、途切れない最長の並び。
  1 連鎖の長さ = 最初の極値から最後の極値まで ＋ 半周期 1 つ（0.5/速さ）。
  録音単位のビブラート合計秒 = 採点区間内で長さが下限以上の全連鎖の長さの和。

主系列の判定条件（2026-10-23 v1.2 で既定を切り替え）:
  既定 = 山谷差 15 cent・3〜9 Hz・対 4 組以上・長さの下限 0 s
         （格子探索 144 通り＝c3_vibrato_grid.csv で「比の中央値 0.9〜1.1 のうち r 最大」だった条件）
  旧既定 = 山谷差 15 cent・3〜9 Hz・対 3 組以上・長さの下限 0.6 s
         （列 vib_sec_ge06 等として残す。--vib-dmin 15 --vib-band 3,9 --vib-pairs 3 --vib-minlen 0.6 で主系列も戻せる）
  音域の主系列 = 最高音は持続音の最高値・最低音は高信頼な持続音の最低値（パーセンタイル版は【旧】として残す）

使い方: python c3_consistency_table.py [--sessions 20260924,20261014,20261021] [--out ...]
        python c3_consistency_table.py --grid          # 判定条件の格子探索
        python c3_consistency_table.py --range-probe    # 音域の切り分け（基準メロディ 対 歌唱）
"""
import argparse, json, os, re, sys, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path("G:/研究アーカイブ/成果物/動画解析")
G3 = BASE / "g3v2_20260928"
DS = BASE / "dataset"
SM = BASE / "song_metrics"
DAM_CSV = Path("G:/研究アーカイブ/成果物/dam_api/dam_ai-hearts_full_2026-10-21T10-50-11.csv")

FR = 100.0            # ピッチ曲線の時間解像度（fps）
MED = 0.3             # 移動中央値の窓[s]
DMIN = 15.0           # 山谷差の下限[cent]
FMIN, FMAX = 3.0, 9.0
HMIN, HMAX = 1.0 / (2 * FMAX), 1.0 / (2 * FMIN)
SPAN_PAD = (-5.0, 2.0)  # index.json の span と同じ作り方（最初の加点−5 s 〜 最後の加点＋2 s）

# 主系列のビブラート判定条件（2026-10-23 v1.2 で既定を切り替え）。
# 既定 ＝ 格子探索 144 通り（c3_vibrato_grid.csv）で「比の中央値 0.9〜1.1 のうち r 最大」だった条件。
# 旧既定（山谷差 15 cent・3〜9 Hz・対 3 組以上・長さの下限 0.6 s）は列 vib_sec_ge06 等として残す。
# --vib-dmin / --vib-band / --vib-pairs / --vib-minlen で主系列を旧既定に戻せる。
PRI = dict(dmin=15.0, fmin=3.0, fmax=9.0, pairs=4, minlen=0.0)

COUNTERS = ["vibrato", "shakuri", "kobushi", "accent", "hammering", "fall"]
DAM_COUNT = {"vibrato": "vibratoCount", "shakuri": "shakuriCount", "kobushi": "kobushiCount",
             "accent": "accentCount", "hammering": "hammeringOnCount", "fall": "fallCount"}
JA = {"vibrato": "ビブラート", "shakuri": "しゃくり", "kobushi": "こぶし",
      "accent": "アクセント", "hammering": "ハンマリング", "fall": "フォール"}


# ---------------------------------------------------------------- 曲名の正規化と DAM 行の突合
def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\[生音\]|[（(]生音[)）]", "", s)
    s = re.sub(r"[（(\[].*?[)）\]]", "", s)
    return re.sub(r"\s+", "", s).lower()


def hhmmss(s):
    m = re.search(r"(\d{2}):(\d{2}):(\d{2})", str(s))
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    m = re.search(r"_(\d{8})_(\d{2})(\d{2})(\d{2})", str(s))
    if m:
        return int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
    return None


# ---------------------------------------------------------------- 極値列の抽出（判定条件の格子探索 --grid 用）
def extrema_in(z, ta, tb):
    """区間 [ta,tb] の有声区間ごとに「山谷を交互に並べた列」を返す。
    返り値 ＝ [(時刻の配列[s], 残差の配列[cent]), ...]。
    移動中央値の窓 MED だけに依存し、山谷差の下限・速さの範囲・対の数には依存しないので、
    判定条件を振る探索ではこれを一度だけ作って再利用できる。"""
    hop = float(z["hop"]); t0 = float(z["t0"]); st = z["st"]; v = z["voiced"] > 0.5; T = len(st)
    i0 = max(0, int(round((ta - t0) / hop))); i1 = min(T, int(round((tb - t0) / hop)))
    if i1 - i0 < int(MED * FR):
        return []
    cents = st[i0:i1].astype(np.float64) * 100.0
    vo = v[i0:i1]; n = len(vo)
    runs = []; i = 0
    while i < n:
        if not vo[i]:
            i += 1; continue
        j = i
        while j < n and vo[j]:
            j += 1
        if j - i >= int(MED * FR):
            runs.append((i, j))
        i = j
    kmed = int(MED * FR) | 1
    out = []
    for (i, j) in runs:
        c = cents[i:j]
        r = c - median_filter(c, size=kmed, mode="nearest")
        pk, _ = find_peaks(r, distance=3); tr, _ = find_peaks(-r, distance=3)
        e = sorted([(p + i, r[p], 1) for p in pk] + [(q + i, r[q], -1) for q in tr])
        al = []
        for x in e:
            if al and al[-1][2] == x[2]:
                if (x[2] == 1 and x[1] > al[-1][1]) or (x[2] == -1 and x[1] < al[-1][1]):
                    al[-1] = x
            else:
                al.append(x)
        if len(al) >= 2:
            out.append((np.array([t0 + (i0 + a[0]) * hop for a in al]),
                        np.array([a[1] for a in al])))
    return out


def chain_durs(ex, dmin, fmin, fmax, min_pairs):
    """極値列から連鎖を取り、1 本ごとの長さ[s]（最初の極値から最後の極値まで ＋ 半周期 1 つ）を返す。
    連鎖 ＝ 隣り合う山谷の差 >= dmin cent かつ 半周期 ∈ [1/(2*fmax), 1/(2*fmin)] s を満たす対が
    min_pairs 組以上続く、途切れない最長の並び（chains_in と同じ規則を配列で解いたもの）。"""
    hmin, hmax = 1.0 / (2 * fmax), 1.0 / (2 * fmin)
    durs = []
    for (t, v) in ex:
        if len(t) < min_pairs + 1:
            continue
        dv = np.abs(np.diff(v)); dt = np.diff(t)
        ok = (dv >= dmin) & (dt >= hmin) & (dt <= hmax)
        if not ok.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
        for s, e in zip(idx[0::2], idx[1::2]):      # ok[s:e] が連続 True ＝ 対が e-s 組
            if e - s < min_pairs:
                continue
            rate = 1.0 / (2 * float(np.median(dt[s:e])))
            durs.append(t[e] - t[s] + 0.5 / rate)
    return np.array(durs)


def chains_param(ex, dmin, fmin, fmax, min_pairs):
    """chain_durs と同じ連鎖を取り、1 本ごとの (長さ[s], 深さ[cent], 速さ[Hz]) を返す。"""
    hmin, hmax = 1.0 / (2 * fmax), 1.0 / (2 * fmin)
    durs, deps, rates = [], [], []
    for (t, v) in ex:
        if len(t) < min_pairs + 1:
            continue
        dv = np.abs(np.diff(v)); dt = np.diff(t)
        ok = (dv >= dmin) & (dt >= hmin) & (dt <= hmax)
        if not ok.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
        for s, e in zip(idx[0::2], idx[1::2]):
            if e - s < min_pairs:
                continue
            rate = 1.0 / (2 * float(np.median(dt[s:e])))
            durs.append(t[e] - t[s] + 0.5 / rate)
            deps.append(float(np.median(dv[s:e])))
            rates.append(rate)
    return np.array(durs), np.array(deps), np.array(rates)


# ---------------------------------------------------------------- ビブラート連鎖（定義は docstring 参照）
def chains_in(z, ta, tb):
    hop = float(z["hop"]); t0 = float(z["t0"]); st = z["st"]; v = z["voiced"] > 0.5; T = len(st)
    i0 = max(0, int(round((ta - t0) / hop))); i1 = min(T, int(round((tb - t0) / hop)))
    if i1 - i0 < int(MED * FR):
        return []
    cents = st[i0:i1].astype(np.float64) * 100.0
    vo = v[i0:i1]; n = len(vo)
    runs = []; i = 0
    while i < n:
        if not vo[i]:
            i += 1; continue
        j = i
        while j < n and vo[j]:
            j += 1
        if j - i >= int(MED * FR):
            runs.append((i, j))
        i = j
    kmed = int(MED * FR) | 1
    out = []
    for (i, j) in runs:
        c = cents[i:j]
        r = c - median_filter(c, size=kmed, mode="nearest")
        pk, _ = find_peaks(r, distance=3); tr, _ = find_peaks(-r, distance=3)
        e = sorted([(p + i, r[p], 1) for p in pk] + [(q + i, r[q], -1) for q in tr])
        al = []
        for x in e:
            if al and al[-1][2] == x[2]:
                if (x[2] == 1 and x[1] > al[-1][1]) or (x[2] == -1 and x[1] < al[-1][1]):
                    al[-1] = x
            else:
                al.append(x)
        k = 0
        while k < len(al) - 1:
            m = k
            while m < len(al) - 1 and abs(al[m + 1][1] - al[m][1]) >= DMIN and HMIN <= (al[m + 1][0] - al[m][0]) * hop <= HMAX:
                m += 1
            if m - k >= 3:
                ch = al[k:m + 1]
                pp = [abs(ch[q + 1][1] - ch[q][1]) for q in range(len(ch) - 1)]
                half = [(ch[q + 1][0] - ch[q][0]) * hop for q in range(len(ch) - 1)]
                rate = 1.0 / (2 * float(np.median(half)))
                out.append(dict(t_start=t0 + (i0 + ch[0][0]) * hop, t_end=t0 + (i0 + ch[-1][0]) * hop,
                                depth=float(np.median(pp)), rate=rate, n_pairs=len(pp)))
            k = max(m, k + 1)
    return out


def measure(npz_path, span):
    z = np.load(npz_path)
    hop = float(z["hop"]); t0 = float(z["t0"]); st = z["st"]; v = z["voiced"] > 0.5
    cf = z["conf"] if "conf" in z.files else np.ones(len(st), np.float32)
    lo, hi = span
    i0 = max(0, int(round((lo - t0) / hop))); i1 = min(len(st), int(round((hi - t0) / hop)))
    sv = st[i0:i1]; vv = v[i0:i1]; cv = cf[i0:i1]
    d = dict(span_sec=round((i1 - i0) * hop, 1), wav_sec=round(len(st) * hop, 1),
             voiced_ratio=round(float(vv.mean()), 3) if len(vv) else np.nan)
    pit = sv[vv].astype(np.float64)
    if len(pit) > 100:
        d.update(pitch_p5=round(float(np.percentile(pit, 5)), 1), pitch_p95=round(float(np.percentile(pit, 95)), 1),
                 pitch_p1=round(float(np.percentile(pit, 1)), 1), pitch_p99=round(float(np.percentile(pit, 99)), 1),
                 pitch_min=round(float(pit.min()), 1), pitch_max=round(float(pit.max()), 1))
    else:
        for k in ("pitch_p5", "pitch_p95", "pitch_p1", "pitch_p99", "pitch_min", "pitch_max"):
            d[k] = np.nan
    # 音符単位の音域（フレーム単位の最小値は F0 探索下限 70 Hz に張り付くため、持続音の代表値で測り直す）
    # 持続音 ＝ 有声が 0.15 s 以上続き、前後で音高が 0.8 半音以上跳ばない区間（song_metrics.py と同じ作り方）
    notes = []; notes_hc = []   # hc ＝ 高信頼（pYIN の有声確率の中央値 >= 0.6・長さ >= 0.3 s）。オクターブ下への推定誤りを除くため
    sm = sv.astype(np.float64).copy()
    ii = np.where(vv)[0]
    if len(ii) > 10:
        sm[ii] = median_filter(sm[ii], size=5, mode="nearest")
    n = len(vv); i = 0
    while i < n:
        if not vv[i]:
            i += 1; continue
        j = i
        while j + 1 < n and vv[j + 1] and abs(np.median(sm[max(i, j - 4):j + 1]) - np.median(sm[j + 1:j + 6])) <= 0.8:
            j += 1
        if (j - i + 1) * hop >= 0.15:
            med = float(np.median(sm[i:j + 1]))
            notes.append(med)
            if (j - i + 1) * hop >= 0.3 and float(np.median(cv[i:j + 1])) >= 0.6:
                notes_hc.append(med)
        i = j + 1
    d["n_notes"] = len(notes); d["n_notes_hc"] = len(notes_hc)
    for na, tag in ((np.array(notes), "note"), (np.array(notes_hc), "note_hc")):
        if len(na) >= 20:
            d.update({f"{tag}_min": round(float(na.min()), 1), f"{tag}_p5": round(float(np.percentile(na, 5)), 1),
                      f"{tag}_p95": round(float(np.percentile(na, 95)), 1), f"{tag}_max": round(float(na.max()), 1)})
        else:
            for k in ("min", "p5", "p95", "max"):
                d[f"{tag}_{k}"] = np.nan
    ch = chains_in(z, lo, hi)
    dur = np.array([c["t_end"] - c["t_start"] + 0.5 / c["rate"] for c in ch]) if ch else np.array([])
    dep = np.array([c["depth"] for c in ch]) if ch else np.array([])
    rat = np.array([c["rate"] for c in ch]) if ch else np.array([])
    for th, tag in ((0.0, "all"), (0.6, "ge06"), (1.0, "ge10")):
        m = dur >= th
        d[f"vib_sec_{tag}"] = round(float(dur[m].sum()), 1)
        d[f"vib_n_{tag}"] = int(m.sum())
    # 採点区間（span）は「最初の加点−5 s 〜 最後の加点＋2 s」なので曲の前後が欠ける。
    # DAM の合計秒は曲全体の採点なので、録音全体を窓にした版も併記して窓の取り方の影響を見る。
    chf = chains_in(z, t0, t0 + len(st) * hop)
    durf = np.array([c["t_end"] - c["t_start"] + 0.5 / c["rate"] for c in chf]) if chf else np.array([])
    for th, tag in ((0.0, "all"), (0.6, "ge06"), (1.0, "ge10")):
        mf = durf >= th
        d[f"vib_sec_full_{tag}"] = round(float(durf[mf].sum()), 1)
        d[f"vib_n_full_{tag}"] = int(mf.sum())
    m = dur >= 0.6
    d["vib_depth_med_old"] = round(float(np.median(dep[m])), 1) if m.sum() else np.nan
    d["vib_rate_med_old"] = round(float(np.median(rat[m])), 2) if m.sum() else np.nan
    d["vib_len_med_old"] = round(float(np.median(dur[m])), 2) if m.sum() else np.nan
    # ---- 主系列（2026-10-23 v1.2 で既定を切り替え。格子探索 c3_vibrato_grid.csv の最良条件）
    ex = extrema_in(z, lo, hi)
    pd_, pdep, prat = chains_param(ex, PRI["dmin"], PRI["fmin"], PRI["fmax"], PRI["pairs"])
    mp = pd_ >= PRI["minlen"]
    d["vib_sec"] = round(float(pd_[mp].sum()), 1); d["vib_n"] = int(mp.sum())
    d["vib_depth_med"] = round(float(np.median(pdep[mp])), 1) if mp.sum() else np.nan
    d["vib_rate_med"] = round(float(np.median(prat[mp])), 2) if mp.sum() else np.nan
    d["vib_len_med"] = round(float(np.median(pd_[mp])), 2) if mp.sum() else np.nan
    exf = extrema_in(z, t0, t0 + len(st) * hop)
    fd_, _, _ = chains_param(exf, PRI["dmin"], PRI["fmin"], PRI["fmax"], PRI["pairs"])
    d["vib_sec_full"] = round(float(fd_[fd_ >= PRI["minlen"]].sum()), 1)
    return d


# ---------------------------------------------------------------- 収録一覧（録音ごと）
def recordings(session, out):
    """[{session, video, song, t_sec, npz, span, g1}] を返す。"""
    idx = json.load(open(G3 / "index.json", encoding="utf-8"))
    rows = []
    insess = [d for d in idx if d["session"] == session]
    if insess:
        for d in insess:
            npz = G3 / "feats" / f"{d['video']}.npz"
            if not npz.exists():
                continue
            rows.append(dict(session=session, video=d["video"], song=d["video"].split(f"_{session}_")[0],
                             t_sec=hhmmss(d["video"]), npz=str(npz), span=tuple(d["span"])))
        return rows
    # index.json に無いセッション（10/21 以降）＝ G1 の技法イベントから収録一覧と採点区間を作る
    g1p = DS / f"G1_events_{session}_wao.csv"
    if not g1p.exists():
        return rows
    g = pd.read_csv(g1p, encoding="utf-8-sig")
    fd = out / f"feats_{session}"
    for video, sub in g.groupby("video"):
        wav = str(sub["wav"].dropna().iloc[0]) if sub["wav"].notna().any() else ""
        npz = fd / (os.path.splitext(wav)[0] + ".npz")
        if not wav or not npz.exists():
            continue
        t = sub["audio_time_sec"].astype(float)
        rows.append(dict(session=session, video=video, song=str(video).split(f"_{session}_")[0],
                         t_sec=hhmmss(wav), npz=str(npz),
                         span=(float(t.min()) + SPAN_PAD[0], float(t.max()) + SPAN_PAD[1])))
    return rows


def g1_counts(session):
    """{video: {counter: (行数, deltaの和, 最終値)}}"""
    p = DS / f"G1_events_{session}_wao.csv"
    if p.exists():
        g = pd.read_csv(p, encoding="utf-8-sig")
    else:
        a = pd.read_csv(DS / "G1_events_all.csv", encoding="utf-8-sig")
        g = a[a["session"].astype(str) == session].copy()
    res = {}
    for (video, counter), sub in g.groupby(["video", "counter"]):
        res.setdefault(video, {})[counter] = (len(sub), int(sub["delta"].sum()), int(sub["count_value"].max()))
    return res


def dam_rows(session):
    d = pd.read_csv(DAM_CSV, encoding="utf-8-sig")
    day = f"{session[:4]}-{session[4:6]}-{session[6:]}"
    d = d[d["playDate"].astype(str).str.startswith(day)].copy()
    d["key"] = d["songName"].map(norm)
    d["t_sec"] = d["playDate"].map(hhmmss)
    return d.reset_index(drop=True)


def match_dam(recs, dam, tol=420):
    """曲名の正規化一致＋時刻の最近傍（±tol 秒）で 1 対 1 に結ぶ。"""
    cand = []
    for i, r in enumerate(recs):
        k = norm(r["song"])
        for j, dr in dam.iterrows():
            if dr["key"] != k or r["t_sec"] is None or dr["t_sec"] is None:
                continue
            diff = abs(r["t_sec"] - dr["t_sec"])
            if diff <= tol:
                cand.append((diff, i, j))
    used_r, used_d, pair = set(), set(), {}
    for diff, i, j in sorted(cand):
        if i in used_r or j in used_d:
            continue
        pair[i] = (j, diff); used_r.add(i); used_d.add(j)
    return pair


# ---------------------------------------------------------------- 項目の統計
def item_stats(df, dam_col, meas_col, label, unit, scope):
    s = df[[dam_col, meas_col, "session", "song", "video"]].dropna(subset=[dam_col, meas_col])
    x = s[dam_col].astype(float).to_numpy(); y = s[meas_col].astype(float).to_numpy()
    out = dict(scope=scope, item=label, unit=unit, dam_col=dam_col, meas_col=meas_col, n=int(len(s)))
    if len(s) >= 4 and np.std(x) > 0 and np.std(y) > 0:
        out["pearson_r"] = round(float(stats.pearsonr(x, y)[0]), 3)
        rho, p = stats.spearmanr(x, y)
        out["spearman_rho"] = round(float(rho), 3); out["spearman_p"] = float(f"{p:.3g}")
    else:
        out["pearson_r"] = out["spearman_rho"] = out["spearman_p"] = None
    diff = y - x
    out["median_diff"] = round(float(np.median(diff)), 2) if len(diff) else None
    out["median_abs_diff"] = round(float(np.median(np.abs(diff))), 2) if len(diff) else None
    out["mad_diff"] = round(float(np.median(np.abs(diff - np.median(diff)))), 2) if len(diff) else None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(x != 0, y / np.where(x == 0, np.nan, x), np.nan)
    rr = ratio[np.isfinite(ratio)]
    out["median_ratio"] = round(float(np.median(rr)), 3) if len(rr) else None
    out["exact_match_rate"] = round(float(np.mean(np.abs(diff) < 0.5)), 3) if len(diff) else None
    if len(diff):
        k = np.argsort(-np.abs(diff - np.median(diff)))[:3]
        out["outliers"] = [dict(session=str(s["session"].iloc[t]), song=str(s["song"].iloc[t]),
                                dam=float(x[t]), meas=float(y[t]), diff=round(float(diff[t]), 2)) for t in k]
    else:
        out["outliers"] = []
    return out


def paired(sessions, out):
    """[(session, video, song, npz, span, DAM のビブラート合計秒)] を返す（音を読まない軽い前段）。"""
    res = []
    for ses in sessions:
        recs = recordings(ses, out)
        if not recs:
            continue
        dam = dam_rows(ses); pair = match_dam(recs, dam)
        for i, r in enumerate(recs):
            if i not in pair:
                continue
            dr = dam.iloc[pair[i][0]]
            res.append((ses, r["video"], r["song"], r["npz"], r["span"],
                        float(dr["vibratoTotalSecond"]) if pd.notna(dr["vibratoTotalSecond"]) else np.nan))
    return res


def stats_of(x, y):
    """DAM 側 x と測定側 y の対応の強さ。n・r・ρ・p・差の中央値・差の絶対値の中央値・比の中央値。"""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = dict(n=int(len(x)))
    if len(x) >= 4 and np.std(x) > 0 and np.std(y) > 0:
        d["pearson_r"] = round(float(stats.pearsonr(x, y)[0]), 3)
        rho, p = stats.spearmanr(x, y)
        d["spearman_rho"] = round(float(rho), 3); d["spearman_p"] = float(f"{p:.3g}")
    else:
        d["pearson_r"] = d["spearman_rho"] = d["spearman_p"] = None
    diff = y - x
    d["median_diff"] = round(float(np.median(diff)), 2) if len(diff) else None
    d["median_abs_diff"] = round(float(np.median(np.abs(diff))), 2) if len(diff) else None
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(x != 0, y / np.where(x == 0, np.nan, x), np.nan)
    rr = rr[np.isfinite(rr)]
    d["median_ratio"] = round(float(np.median(rr)), 3) if len(rr) else None
    d["within_05"] = round(float(np.mean(np.abs(diff) < 0.5)), 3) if len(diff) else None
    d["within_15"] = round(float(np.mean(np.abs(diff) < 1.5)), 3) if len(diff) else None
    return d


# 格子（指示 2026-10-23・候補は全部回す）
G_DMIN = [10.0, 15.0, 20.0, 30.0]
G_BAND = [(3.0, 9.0), (4.0, 8.0), (4.5, 8.0)]
G_PAIRS = [2, 3, 4]
G_MINLEN = [0.0, 0.4, 0.6, 0.8]


def run_grid(sessions, out):
    """ビブラートの判定条件を格子で振り、DAM のビブラート合計秒との対応を表にする。"""
    pr = paired(sessions, out)
    print(f"格子探索 対象 {len(pr)} 本・条件 {len(G_DMIN)*len(G_BAND)*len(G_PAIRS)*len(G_MINLEN)} 通り", flush=True)
    ex = {}
    for k, (ses, video, song, npz, span, dsec) in enumerate(pr):
        ex[(ses, video)] = extrema_in(np.load(npz), span[0], span[1])
        if (k + 1) % 20 == 0:
            print(f"  極値列 {k+1}/{len(pr)}", flush=True)
    rows = []
    for dmin in G_DMIN:
        for (fmin, fmax) in G_BAND:
            for np_ in G_PAIRS:
                durs = {(p[0], p[1]): chain_durs(ex[(p[0], p[1])], dmin, fmin, fmax, np_) for p in pr}
                for mlen in G_MINLEN:
                    sec = np.array([float(durs[(p[0], p[1])][durs[(p[0], p[1])] >= mlen].sum()) for p in pr])
                    dam = np.array([p[5] for p in pr])
                    ses = np.array([p[0] for p in pr])
                    base = dict(dmin_cent=dmin, f_lo_hz=fmin, f_hi_hz=fmax, min_pairs=np_, min_len_sec=mlen)
                    r = dict(base); r["scope"] = "全体"; r.update(stats_of(dam, sec))
                    r["median_meas_sec"] = round(float(np.median(sec)), 2); rows.append(r)
                    for s in sorted(set(ses)):
                        m = ses == s
                        r = dict(base); r["scope"] = s; r.update(stats_of(dam[m], sec[m]))
                        r["median_meas_sec"] = round(float(np.median(sec[m])), 2); rows.append(r)
    G = pd.DataFrame(rows)
    cols = ["scope", "dmin_cent", "f_lo_hz", "f_hi_hz", "min_pairs", "min_len_sec", "n",
            "pearson_r", "spearman_rho", "spearman_p", "median_diff", "median_abs_diff", "median_ratio", "median_meas_sec"]
    G = G[cols].sort_values(["scope", "dmin_cent", "f_lo_hz", "min_pairs", "min_len_sec"])
    G.to_csv(out / "c3_vibrato_grid.csv", index=False, encoding="utf-8-sig")
    print(f"→ {out/'c3_vibrato_grid.csv'}（{len(G)} 行）", flush=True)
    for s in ["全体"] + sorted(set(G.scope) - {"全体"}):
        g = G[G.scope == s].dropna(subset=["pearson_r"])
        un = g[(g.median_ratio >= 0.9) & (g.median_ratio <= 1.1)]
        pick = (un.loc[un.pearson_r.idxmax()] if len(un) else g.loc[g.pearson_r.idxmax()])
        print(f"  [{s}] 最良（比 0.9〜1.1 のうち r 最大）= dmin {pick.dmin_cent:g} / {pick.f_lo_hz:g}-{pick.f_hi_hz:g} Hz / 対 {pick.min_pairs:g} 以上 / 下限 {pick.min_len_sec:g} s"
              f" → r={pick.pearson_r} ρ={pick.spearman_rho} 差={pick.median_diff} 比={pick.median_ratio}"
              f"{'' if len(un) else '（比 0.9〜1.1 に該当なし＝r 最大で代替）'}", flush=True)
        b = g.loc[g.pearson_r.idxmax()]
        print(f"       r 最大 = dmin {b.dmin_cent:g} / {b.f_lo_hz:g}-{b.f_hi_hz:g} Hz / 対 {b.min_pairs:g} / 下限 {b.min_len_sec:g} s → r={b.pearson_r} 差={b.median_diff} 比={b.median_ratio}", flush=True)
    cur = G[(G.scope == "全体") & (G.dmin_cent == 15) & (G.f_lo_hz == 3) & (G.min_pairs == 3) & (G.min_len_sec == 0.6)]
    if len(cur):
        c = cur.iloc[0]
        print(f"  [現行条件] dmin 15 / 3-9 Hz / 対 3 / 下限 0.6 s → r={c.pearson_r} ρ={c.spearman_rho} 差={c.median_diff} 比={c.median_ratio}", flush=True)
    return 0


REF = BASE / "pitchbar_read_20261008/out"


def run_range_probe(sessions, out):
    """DAM の音域が「歌唱の音域」か「基準メロディの音域」かを切り分ける。
    基準メロディ ＝ 音程ガイド表示の読み取り結果（`pitchbar_read_20261008/out/<video>_ref.csv` の ref_midi）。"""
    T = pd.read_csv(out / "c3_consistency_songs.csv", encoding="utf-8-sig")
    rows = []
    for _, r in T.iterrows():
        p = REF / f"{r['video']}_ref.csv"
        d = dict(session=r["session"], video=r["video"], song=r["song"],
                 dam_low=r["dam_singingRangeLowest"], dam_high=r["dam_singingRangeHighest"],
                 sing_hc_min=r.get("meas_note_hc_min"), sing_hc_p5=r.get("meas_note_hc_p5"),
                 sing_hc_max=r.get("meas_note_hc_max"), sing_hc_p95=r.get("meas_note_hc_p95"))
        if p.exists():
            f = pd.read_csv(p, encoding="utf-8-sig")
            f = f[(f["audio_time_sec"] >= r["span_lo"]) & (f["audio_time_sec"] <= r["span_hi"])]
            g = f[f["ref_midi"].notna()]
            gh = g[g["conf"] >= 0.5]
            for tag, h in (("ref", g), ("ref_hc", gh)):
                if len(h) >= 100:
                    d[f"{tag}_min"] = round(float(h["ref_midi"].min()), 1)
                    d[f"{tag}_p5"] = round(float(np.percentile(h["ref_midi"], 5)), 1)
                    d[f"{tag}_p95"] = round(float(np.percentile(h["ref_midi"], 95)), 1)
                    d[f"{tag}_max"] = round(float(h["ref_midi"].max()), 1)
                    d[f"{tag}_n"] = len(h)
                else:
                    for k in ("min", "p5", "p95", "max", "n"):
                        d[f"{tag}_{k}"] = np.nan
        else:
            for tag in ("ref", "ref_hc"):
                for k in ("min", "p5", "p95", "max", "n"):
                    d[f"{tag}_{k}"] = np.nan
        rows.append(d)
    R = pd.DataFrame(rows)
    R.to_csv(out / "c3_range_probe.csv", index=False, encoding="utf-8-sig")
    comp = [("最低音 対 基準メロディの最低値", "dam_low", "ref_min"),
            ("最低音 対 基準メロディの 5 パーセンタイル", "dam_low", "ref_p5"),
            ("最低音 対 歌唱の高信頼 5 パーセンタイル", "dam_low", "sing_hc_p5"),
            ("最低音 対 歌唱の高信頼な最小値", "dam_low", "sing_hc_min"),
            ("最高音 対 基準メロディの最高値", "dam_high", "ref_max"),
            ("最高音 対 基準メロディの 95 パーセンタイル", "dam_high", "ref_p95"),
            ("最高音 対 歌唱の高信頼 95 パーセンタイル", "dam_high", "sing_hc_p95"),
            ("最高音 対 歌唱の高信頼な最大値", "dam_high", "sing_hc_max")]
    srows = []
    for label, dc, mc in comp:
        sub = R.dropna(subset=[dc, mc])
        # 比較は同じ録音集合で行う＝基準メロディが読めている録音（9/24 の 28 本）に揃えた列を必ず出す
        for scope in ["基準メロディあり"] + sorted(set(sub["session"].astype(str))):
            s = sub[sub["ref_min"].notna()] if scope.startswith("基準") else sub[sub["session"].astype(str) == scope]
            if not len(s):
                continue
            d = dict(item=label, scope=scope, dam_col=dc, meas_col=mc)
            d.update(stats_of(s[dc].to_numpy(float), s[mc].to_numpy(float)))
            srows.append(d)
    S = pd.DataFrame(srows)
    S.to_csv(out / "c3_range_probe_stats.csv", index=False, encoding="utf-8-sig")
    print(f"→ {out/'c3_range_probe.csv'} / c3_range_probe_stats.csv", flush=True)
    for _, r in S.iterrows():
        print(f"  [{r['scope']}] {r['item']}: n={r['n']} r={r['pearson_r']} ρ={r['spearman_rho']} 差={r['median_diff']} |差|={r['median_abs_diff']} ±0.5内={r['within_05']} ±1.5内={r['within_15']}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="20260924,20261014,20261021")
    ap.add_argument("--out", default=str(BASE / "c3_consistency_20261023"))
    ap.add_argument("--vib-dmin", type=float, default=PRI["dmin"], help="主系列の山谷差の下限[cent]（旧既定 15）")
    ap.add_argument("--vib-band", default=f"{PRI['fmin']:g},{PRI['fmax']:g}", help="主系列の揺れの速さの範囲[Hz]（旧既定 3,9）")
    ap.add_argument("--vib-pairs", type=int, default=PRI["pairs"], help="主系列の山谷の対の数の下限（旧既定 3）")
    ap.add_argument("--vib-minlen", type=float, default=PRI["minlen"], help="主系列の 1 本の長さの下限[s]（旧既定 0.6）")
    ap.add_argument("--grid", action="store_true", help="ビブラートの判定条件を格子で振る（整合表は作り直さない）")
    ap.add_argument("--range-probe", action="store_true", help="DAM の音域と基準メロディ・歌唱の音域を並べる（整合表は作り直さない）")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sessions = [s for s in a.sessions.split(",") if s]
    fl, fh = (float(x) for x in a.vib_band.split(","))
    PRI.update(dmin=a.vib_dmin, fmin=fl, fmax=fh, pairs=a.vib_pairs, minlen=a.vib_minlen)
    print(f"主系列の判定条件＝山谷差 {PRI['dmin']:g} cent・{PRI['fmin']:g}〜{PRI['fmax']:g} Hz・対 {PRI['pairs']} 組以上・長さの下限 {PRI['minlen']:g} s", flush=True)
    if a.grid or a.range_probe:
        rc = 0
        if a.grid:
            rc |= run_grid(sessions, out)
        if a.range_probe:
            rc |= run_range_probe(sessions, out)
        return rc

    rows, skipped = [], []
    for ses in sessions:
        recs = recordings(ses, out)
        if not recs:
            skipped.append(dict(session=ses, reason="収録一覧（index.json / G1）またはピッチ曲線 npz が無い")); continue
        dam = dam_rows(ses)
        pair = match_dam(recs, dam)
        g1 = g1_counts(ses)
        sm = None
        p = SM / f"歌声分析_{ses}.csv"
        if p.exists():
            sm = pd.read_csv(p, encoding="utf-8-sig")
            sm["key"] = sm["song"].map(norm)
        for i, r in enumerate(recs):
            if i not in pair:
                skipped.append(dict(session=ses, video=r["video"], reason="DAM 採点行と突合できなかった（曲名または時刻が一致せず）")); continue
            j, diff = pair[i]
            dr = dam.iloc[j]
            row = dict(session=ses, video=r["video"], song=r["song"], artist=dr.get("artistName", ""),
                       play_date=dr["playDate"], match_diff_sec=diff,
                       dam_total_score=float(dr["totalScore"]) if pd.notna(dr["totalScore"]) else np.nan)
            for c in COUNTERS:
                row[f"dam_{c}_count"] = float(dr[DAM_COUNT[c]]) if pd.notna(dr[DAM_COUNT[c]]) else np.nan
            for k in ("vibratoTotalSecond", "vibratoType", "vibratoSkill", "longtoneCount", "longtoneSkill",
                      "singingRangeHighest", "singingRangeLowest", "vocalRangeHighest", "vocalRangeLowest",
                      "radarChartVibratoLongtone"):
                row["dam_" + k] = float(dr[k]) if pd.notna(dr[k]) else np.nan
            gg = g1.get(r["video"], {})
            for c in COUNTERS:
                n_rows, n_delta, last = gg.get(c, (0, 0, 0))
                row[f"g1_{c}_rows"] = n_rows; row[f"g1_{c}_delta"] = n_delta; row[f"g1_{c}_last"] = last
            row["span_lo"] = round(r["span"][0], 2); row["span_hi"] = round(r["span"][1], 2)
            row.update({("meas_" + k): v for k, v in measure(r["npz"], r["span"]).items()})
            if sm is not None:
                h = sm[sm["key"] == norm(r["song"])]
                if len(h) > 1 and "time" in h.columns and r["t_sec"] is not None:
                    def tdiff(s):
                        s = str(s).zfill(6)
                        return abs(int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) - r["t_sec"])
                    h = h.iloc[np.argsort(h["time"].map(tdiff).to_numpy())]
                if len(h):
                    hh = h.iloc[0]
                    for k in ("voiced", "range_lo", "range_hi", "vib_n", "vib_rate", "vib_depth"):
                        row["sm_" + k] = float(hh[k]) if pd.notna(hh.get(k)) else np.nan
            rows.append(row)
    T = pd.DataFrame(rows).sort_values(["session", "play_date"]).reset_index(drop=True)
    if len(T):   # 派生列（幅）も CSV に入れる
        T["dam_range_width"] = T["dam_singingRangeHighest"] - T["dam_singingRangeLowest"]
        T["meas_pitch_width"] = T["meas_pitch_p95"] - T["meas_pitch_p5"]
        T["meas_note_width"] = T["meas_note_max"] - T["meas_note_min"]
        T["meas_note_width_pri"] = T["meas_note_max"] - T["meas_note_hc_min"]
    T.to_csv(out / "c3_consistency_songs.csv", index=False, encoding="utf-8-sig")

    # ---- 項目の定義（DAM 列 ↔ 比べる列）
    PT = (f"山谷差 {PRI['dmin']:g} cent・{PRI['fmin']:g}〜{PRI['fmax']:g} Hz・対 {PRI['pairs']} 組以上・下限 {PRI['minlen']:g} s")
    pairs = [(f"【主】ビブラート合計秒（DAM）対 連鎖合計秒（測定・{PT}）", "dam_vibratoTotalSecond", "meas_vib_sec", "秒"),
             (f"【主】ビブラート回数（DAM）対 連鎖本数（測定・{PT}）", "dam_vibrato_count", "meas_vib_n", "回"),
             ("【主】ビブラート回数（DAM）対 技法イベントの最終値（正解データ）", "dam_vibrato_count", "g1_vibrato_last", "回"),
             ("【主】最高音（DAM singingRangeHighest）対 持続音の最高値（測定）", "dam_singingRangeHighest", "meas_note_max", "MIDI 半音"),
             ("【主】最低音（DAM singingRangeLowest）対 高信頼な持続音の最低値（測定）", "dam_singingRangeLowest", "meas_note_hc_min", "MIDI 半音"),
             ("【主】音域の幅（DAM singingRange）対 持続音の最高値−高信頼な持続音の最低値（測定）", "dam_range_width", "meas_note_width_pri", "半音"),
             (f"ビブラート合計秒（DAM）対 連鎖合計秒（測定・録音全体・{PT}）", "dam_vibratoTotalSecond", "meas_vib_sec_full", "秒"),
             ("【旧】ビブラート合計秒（DAM）対 連鎖合計秒（測定・下限 0.6 s）", "dam_vibratoTotalSecond", "meas_vib_sec_ge06", "秒"),
             ("【旧】ビブラート合計秒（DAM）対 連鎖合計秒（測定・下限なし）", "dam_vibratoTotalSecond", "meas_vib_sec_all", "秒"),
             ("【旧】ビブラート合計秒（DAM）対 連鎖合計秒（測定・下限 1.0 s）", "dam_vibratoTotalSecond", "meas_vib_sec_ge10", "秒"),
             ("【旧】ビブラート合計秒（DAM）対 連鎖合計秒（測定・録音全体・下限 0.6 s）", "dam_vibratoTotalSecond", "meas_vib_sec_full_ge06", "秒"),
             ("【旧】ビブラート合計秒（DAM）対 連鎖合計秒（測定・録音全体・下限 1.0 s）", "dam_vibratoTotalSecond", "meas_vib_sec_full_ge10", "秒"),
             ("【旧】ビブラート回数（DAM）対 連鎖本数（測定・下限 1.0 s）", "dam_vibrato_count", "meas_vib_n_ge10", "回"),
             ("【旧】ビブラート回数（DAM）対 技法イベント件数（正解データ）", "dam_vibrato_count", "g1_vibrato_last", "回"),
             ("【旧】最高音（DAM singingRangeHighest）対 F0 の 95 パーセンタイル（測定）", "dam_singingRangeHighest", "meas_pitch_p95", "MIDI 半音"),
             ("【旧】最低音（DAM singingRangeLowest）対 F0 の 5 パーセンタイル（測定）", "dam_singingRangeLowest", "meas_pitch_p5", "MIDI 半音"),
             ("【旧】最高音（DAM singingRangeHighest）対 F0 の最大値（測定）", "dam_singingRangeHighest", "meas_pitch_max", "MIDI 半音"),
             ("【旧】最低音（DAM singingRangeLowest）対 F0 の最小値（測定）", "dam_singingRangeLowest", "meas_pitch_min", "MIDI 半音"),
             ("【旧】最高音（DAM singingRangeHighest）対 持続音の最高値（測定・主系列と同一）", "dam_singingRangeHighest", "meas_note_max", "MIDI 半音"),
             ("【旧】最低音（DAM singingRangeLowest）対 持続音の最低値（測定）", "dam_singingRangeLowest", "meas_note_min", "MIDI 半音"),
             ("【旧】最高音（DAM singingRangeHighest）対 持続音の 95 パーセンタイル（測定）", "dam_singingRangeHighest", "meas_note_p95", "MIDI 半音"),
             ("【旧】最低音（DAM singingRangeLowest）対 持続音の 5 パーセンタイル（測定）", "dam_singingRangeLowest", "meas_note_p5", "MIDI 半音"),
             ("【旧】最低音（DAM singingRangeLowest）対 高信頼な持続音の最低値（測定・主系列と同一）", "dam_singingRangeLowest", "meas_note_hc_min", "MIDI 半音"),
             ("【旧】最低音（DAM singingRangeLowest）対 高信頼な持続音の 5 パーセンタイル（測定）", "dam_singingRangeLowest", "meas_note_hc_p5", "MIDI 半音"),
             ("【旧】最高音（DAM singingRangeHighest）対 高信頼な持続音の最高値（測定）", "dam_singingRangeHighest", "meas_note_hc_max", "MIDI 半音"),
             ("【旧】最高音（DAM vocalRangeHighest）対 持続音の最高値（測定）", "dam_vocalRangeHighest", "meas_note_max", "MIDI 半音"),
             ("【旧】最低音（DAM vocalRangeLowest）対 持続音の最低値（測定）", "dam_vocalRangeLowest", "meas_note_min", "MIDI 半音"),
             ("【旧】音域の幅（DAM singingRange）対 F0 の 5〜95 パーセンタイル幅（測定）", "dam_range_width", "meas_pitch_width", "半音"),
             ("【旧】音域の幅（DAM singingRange）対 持続音の最高−最低（測定）", "dam_range_width", "meas_note_width", "半音")]
    for c in COUNTERS:
        pairs.append((f"{JA[c]}の回数（DAM）対 技法イベントの最終値（正解データ）", f"dam_{c}_count", f"g1_{c}_last", "回"))
        pairs.append((f"{JA[c]}の回数（DAM）対 技法イベントの行数（正解データ）", f"dam_{c}_count", f"g1_{c}_rows", "回"))
    items = []
    for label, dc, mc, unit in pairs:
        if dc not in T.columns or mc not in T.columns:
            continue
        items.append(item_stats(T, dc, mc, label, unit, "全体"))
        for ses in sorted(T["session"].unique()):
            items.append(item_stats(T[T["session"] == ses], dc, mc, label, unit, ses))
    I = pd.DataFrame([{k: v for k, v in it.items() if k != "outliers"} for it in items])
    I.to_csv(out / "c3_consistency_items.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(generated="2026-10-23", sessions=sessions, n_recordings=int(len(T)),
                   skipped=skipped, items=items,
                   not_measured=[
                       dict(item="ロングトーンの回数（DAM longtoneCount）",
                            reason="DAM がロングトーンと判定する長さ・安定度の基準が公開されていないため、同じ意味の測定値を定義できない。未測定。"),
                       dict(item="ビブラートの型（DAM vibratoType）",
                            reason="型の判定規則が公開されておらず、測定側に対応する量を定義できない。未測定（型ごとの測定値の中央値のみ別途参考）。"),
                       dict(item="有声割合（測定 meas_voiced_ratio）",
                            reason="DAM 側に対応する公開項目が無い。測定値のみ記録し、整合の検定は行わない。"),
                       dict(item="抑揚・リズム（DAM intonation / timing）",
                            reason="算出式が公開されておらず、同じ意味の測定値を定義できない。未測定。")]),
              open(out / "c3_consistency_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- Markdown
    L = ["# C3（曲単位の整合性検証）整合表 v1（2026-10-23 生成）", "",
         f"対象 ＝ 収録セッション {', '.join(sessions)}。",
         f"突合できた録音 ＝ {len(T)} 本。", ""]
    if skipped:
        L += ["## 突合できなかったもの", ""] + [f"- {s.get('session','')} {s.get('video','')}：{s['reason']}" for s in skipped] + [""]
    L += ["## 項目ごとの整合（全体）", "",
          "| 項目 | 単位 | n | ピアソン r | スピアマン ρ | p | 差の中央値（測定−DAM） | 差の絶対値の中央値 | 比の中央値 |",
          "|---|---|---|---|---|---|---|---|---|"]
    def g(v, d=3):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{d}f}" if isinstance(v, float) else str(v))

    def gp(v):
        return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2g}"
    for it in items:
        if it["scope"] != "全体":
            continue
        L.append(f"| {it['item']} | {it['unit']} | {it['n']} | {g(it['pearson_r'])} | {g(it['spearman_rho'])} | "
                 f"{gp(it['spearman_p'])} | {g(it['median_diff'],2)} | {g(it['median_abs_diff'],2)} | {g(it['median_ratio'])} |")
    L += ["", "## セッション別（主要項目）", "",
          "| 項目 | 収録セッション | n | ピアソン r | スピアマン ρ | 差の中央値 | ±0.5 以内の割合 |", "|---|---|---|---|---|---|---|"]
    PRIMARY = ("【主】", "技法イベントの最終値")
    for it in items:
        if it["scope"] == "全体" or not any(k in it["item"] for k in PRIMARY):
            continue
        L.append(f"| {it['item']} | {it['scope']} | {it['n']} | {g(it['pearson_r'])} | {g(it['spearman_rho'])} | "
                 f"{g(it['median_diff'],2)} | {g(it['exact_match_rate'])} |")
    L += ["", "## 外れ録音（差が中央値から最も離れた 3 本）", ""]
    for it in items:
        if it["scope"] != "全体" or not it["outliers"]:
            continue
        L.append(f"- {it['item']}：" + "／".join(f"{o['song']}（{o['session']}・DAM {o['dam']:g}・測定 {o['meas']:g}・差 {o['diff']:+g}）" for o in it["outliers"]))
    L += ["", "## 1 行 1 録音の表（主要列）", ""]
    cols = ["session", "song", "dam_total_score", "dam_vibratoTotalSecond", "meas_vib_sec", "meas_vib_sec_ge06",
            "dam_vibrato_count", "meas_vib_n", "g1_vibrato_last",
            "dam_singingRangeHighest", "meas_note_max", "dam_singingRangeLowest", "meas_note_hc_min",
            "meas_voiced_ratio", "meas_vib_depth_med", "meas_vib_rate_med"]
    cols = [c for c in cols if c in T.columns]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for _, r in T.iterrows():
        L.append("| " + " | ".join("—" if pd.isna(r[c]) else (f"{r[c]:g}" if isinstance(r[c], (int, float, np.floating)) else str(r[c])) for c in cols) + " |")
    L += ["", "## 技法別の回数（DAM 対 正解データ）", "",
          "| 技法 | n | 完全一致率 | 差の中央値 | スピアマン ρ |", "|---|---|---|---|---|"]
    for it in items:
        if it["scope"] == "全体" and "技法イベントの最終値" in it["item"]:
            L.append(f"| {it['item'].split('の回数')[0]} | {it['n']} | {g(it['exact_match_rate'])} | {g(it['median_diff'],2)} | {g(it['spearman_rho'])} |")
    L += ["", "## 未測定の項目と理由", ""]
    for nm in json.load(open(out / "c3_consistency_stats.json", encoding="utf-8"))["not_measured"]:
        L.append(f"- {nm['item']}：{nm['reason']}")
    (out / "c3_consistency.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"録音 {len(T)} 本・項目 {len([i for i in items if i['scope']=='全体'])} 件 → {out}", flush=True)
    for it in items:
        if it["scope"] == "全体":
            print(f"  {it['item']}: n={it['n']} r={it['pearson_r']} rho={it['spearman_rho']} p={it['spearman_p']} med_diff={it['median_diff']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
