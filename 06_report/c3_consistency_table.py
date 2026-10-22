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

ビブラートの測定定義（vibrato_depth.py / c3_fullspan.py と同一・変更しない）:
  セント列 = F0 を MIDI 半音×100 に直し、有声区間ごとに 0.3 s の移動中央値（旋律の土台）を引いた残差。
  極値列   = 残差の山と谷を交互に取る。
  連鎖     = 隣り合う山谷の差 >= 15 cent かつ 半周期 ∈ [1/18, 1/6] s（3〜9 Hz）を満たす最長の並びで、山谷対が 3 組以上。
  1 連鎖の長さ = 最初の極値から最後の極値まで ＋ 半周期 1 つ（0.5/速さ）。
  録音単位のビブラート合計秒 = 採点区間内の全連鎖の長さの和（長さの下限 0 / 0.6 / 1.0 s の 3 通り）。

使い方: python c3_consistency_table.py [--sessions 20260924,20261014,20261021] [--out ...]
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
    d["vib_depth_med"] = round(float(np.median(dep[m])), 1) if m.sum() else np.nan
    d["vib_rate_med"] = round(float(np.median(rat[m])), 2) if m.sum() else np.nan
    d["vib_len_med"] = round(float(np.median(dur[m])), 2) if m.sum() else np.nan
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="20260924,20261014,20261021")
    ap.add_argument("--out", default=str(BASE / "c3_consistency_20261023"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sessions = [s for s in a.sessions.split(",") if s]

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
    T.to_csv(out / "c3_consistency_songs.csv", index=False, encoding="utf-8-sig")

    # ---- 項目の定義（DAM 列 ↔ 比べる列）
    pairs = [("ビブラート合計秒（DAM）対 連鎖合計秒（測定・下限 0.6 s）", "dam_vibratoTotalSecond", "meas_vib_sec_ge06", "秒"),
             ("ビブラート合計秒（DAM）対 連鎖合計秒（測定・下限なし）", "dam_vibratoTotalSecond", "meas_vib_sec_all", "秒"),
             ("ビブラート合計秒（DAM）対 連鎖合計秒（測定・下限 1.0 s）", "dam_vibratoTotalSecond", "meas_vib_sec_ge10", "秒"),
             ("ビブラート合計秒（DAM）対 連鎖合計秒（測定・録音全体・下限 0.6 s）", "dam_vibratoTotalSecond", "meas_vib_sec_full_ge06", "秒"),
             ("ビブラート合計秒（DAM）対 連鎖合計秒（測定・録音全体・下限 1.0 s）", "dam_vibratoTotalSecond", "meas_vib_sec_full_ge10", "秒"),
             ("ビブラート回数（DAM）対 連鎖本数（測定・下限 1.0 s）", "dam_vibrato_count", "meas_vib_n_ge10", "回"),
             ("ビブラート回数（DAM）対 技法イベント件数（正解データ）", "dam_vibrato_count", "g1_vibrato_last", "回"),
             ("最高音（DAM singingRangeHighest）対 F0 の 95 パーセンタイル（測定）", "dam_singingRangeHighest", "meas_pitch_p95", "MIDI 半音"),
             ("最低音（DAM singingRangeLowest）対 F0 の 5 パーセンタイル（測定）", "dam_singingRangeLowest", "meas_pitch_p5", "MIDI 半音"),
             ("最高音（DAM singingRangeHighest）対 F0 の最大値（測定）", "dam_singingRangeHighest", "meas_pitch_max", "MIDI 半音"),
             ("最低音（DAM singingRangeLowest）対 F0 の最小値（測定）", "dam_singingRangeLowest", "meas_pitch_min", "MIDI 半音"),
             ("最高音（DAM singingRangeHighest）対 持続音の最高値（測定）", "dam_singingRangeHighest", "meas_note_max", "MIDI 半音"),
             ("最低音（DAM singingRangeLowest）対 持続音の最低値（測定）", "dam_singingRangeLowest", "meas_note_min", "MIDI 半音"),
             ("最高音（DAM singingRangeHighest）対 持続音の 95 パーセンタイル（測定）", "dam_singingRangeHighest", "meas_note_p95", "MIDI 半音"),
             ("最低音（DAM singingRangeLowest）対 持続音の 5 パーセンタイル（測定）", "dam_singingRangeLowest", "meas_note_p5", "MIDI 半音"),
             ("最低音（DAM singingRangeLowest）対 高信頼な持続音の最低値（測定）", "dam_singingRangeLowest", "meas_note_hc_min", "MIDI 半音"),
             ("最低音（DAM singingRangeLowest）対 高信頼な持続音の 5 パーセンタイル（測定）", "dam_singingRangeLowest", "meas_note_hc_p5", "MIDI 半音"),
             ("最高音（DAM singingRangeHighest）対 高信頼な持続音の最高値（測定）", "dam_singingRangeHighest", "meas_note_hc_max", "MIDI 半音"),
             ("最高音（DAM vocalRangeHighest）対 持続音の最高値（測定）", "dam_vocalRangeHighest", "meas_note_max", "MIDI 半音"),
             ("最低音（DAM vocalRangeLowest）対 持続音の最低値（測定）", "dam_vocalRangeLowest", "meas_note_min", "MIDI 半音"),
             ("音域の幅（DAM singingRange）対 F0 の 5〜95 パーセンタイル幅（測定）", "dam_range_width", "meas_pitch_width", "半音"),
             ("音域の幅（DAM singingRange）対 持続音の最高−最低（測定）", "dam_range_width", "meas_note_width", "半音")]
    for c in COUNTERS:
        pairs.append((f"{JA[c]}の回数（DAM）対 技法イベントの最終値（正解データ）", f"dam_{c}_count", f"g1_{c}_last", "回"))
        pairs.append((f"{JA[c]}の回数（DAM）対 技法イベントの行数（正解データ）", f"dam_{c}_count", f"g1_{c}_rows", "回"))
    if len(T):
        T["dam_range_width"] = T["dam_singingRangeHighest"] - T["dam_singingRangeLowest"]
        T["meas_pitch_width"] = T["meas_pitch_p95"] - T["meas_pitch_p5"]
        T["meas_note_width"] = T["meas_note_max"] - T["meas_note_min"]
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
    PRIMARY = ("下限 0.6 s）", "持続音の最高値", "持続音の最低値", "高信頼な持続音の最低値", "技法イベントの最終値")
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
    cols = ["session", "song", "dam_total_score", "dam_vibratoTotalSecond", "meas_vib_sec_ge06", "meas_vib_sec_all",
            "dam_vibrato_count", "meas_vib_n_ge10", "g1_vibrato_last",
            "dam_singingRangeHighest", "meas_pitch_p95", "dam_singingRangeLowest", "meas_pitch_p5",
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
