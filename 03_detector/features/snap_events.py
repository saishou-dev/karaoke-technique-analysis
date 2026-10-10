# -*- coding: utf-8 -*-
"""G1イベント（カウンター増分時刻）を「直前の音（ノート）の終わり」に吸着させ、index_snapped.json とノート表を作る。
   ノート＝pYIN有声ランを音程跳び(0.8半音)で分割（classic_detectors と同じ規則）。
   吸着規則＝イベント時刻 t の [t-2.5, t-0.10] にノート終端があれば、その中で最も遅い終端へ。なければ吸着不可（unsnapped）。"""
import sys, json, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--min_len", type=int, default=8); ap.add_argument("--jump", type=float, default=0.8); ap.add_argument("--merge_gap", type=int, default=0, help="この長さ以下の無声ギャップは埋めて音符をつなぐ（フレーム）"); ap.add_argument("--suffix", default="")
A = ap.parse_args()
from scipy.ndimage import median_filter
sys.stdout.reconfigure(encoding='utf-8')
ROOT = "G:/研究アーカイブ/成果物/動画解析/g3v2_20260928"; FR = 100
TYPES = ["shakuri", "kobushi", "vibrato", "fall", "accent", "hammering"]
index = json.load(open(f"{ROOT}/index.json", encoding="utf-8"))
def segments(st, v, min_len=8, jump=0.8):
    sm = st.copy(); ok = ~np.isnan(sm); idx = np.where(ok)[0]
    if len(idx) < 10: return [], sm
    sm[idx] = median_filter(sm[idx], size=5, mode="nearest"); segs = []; i = 0; n = len(sm)
    while i < n:
        if not ok[i]: i += 1; continue
        j = i
        while j + 1 < n and ok[j + 1]:
            a_ = np.nanmedian(sm[max(i, j - 4):j + 1]); b_ = np.nanmedian(sm[j + 1:j + 6])
            if abs(b_ - a_) > jump and (j - i) >= min_len: break
            j += 1
        if (j - i + 1) >= min_len: segs.append((i, j))
        i = j + 1
    return segs, sm
LO, HI = 0.10, 2.5
out = []; lagstat = {c: [] for c in TYPES}; unsn = {c: 0 for c in TYPES}; tot = {c: 0 for c in TYPES}
notes_all = {}
for d in index:
    z = np.load(f"{ROOT}/feats/{d['video']}.npz"); st = z["st"].astype(float); v = z["voiced"] > 0; st[~v] = np.nan
    if A.merge_gap > 0:  # 短い無声ギャップを線形補間で埋める
        idx = np.where(v)[0]
        if len(idx) > 10:
            gaps = np.diff(idx); g_i = np.where((gaps > 1) & (gaps <= A.merge_gap + 1))[0]
            for gi in g_i:
                a0, b0 = idx[gi], idx[gi + 1]; st[a0 + 1:b0] = np.linspace(st[a0], st[b0], b0 - a0 + 1)[1:-1]; v[a0 + 1:b0] = True
    segs, sm = segments(st, v, min_len=A.min_len, jump=A.jump); ends = np.array([(j + 1) / FR for i, j in segs]); starts = np.array([i / FR for i, j in segs])
    notes_all[d["video"]] = dict(start=starts.tolist(), end=ends.tolist())
    ev2 = []; labels = [[0] * len(TYPES) for _ in segs]
    for c, t in d["events"]:
        tot[c] += 1
        cand = np.where((ends >= t - HI) & (ends <= t - LO))[0]
        if len(cand) == 0: unsn[c] += 1; ev2.append((c, t, -1)); continue
        k = int(cand[np.argmax(ends[cand])]); ev2.append((c, float(ends[k]), k)); lagstat[c].append(t - ends[k])
        if c in TYPES: labels[k][TYPES.index(c)] += 1
    d2 = dict(d); d2["events"] = [(c, t) for c, t, k in ev2 if k >= 0]; d2["events_raw"] = d["events"]; d2["note_idx"] = [k for _, _, k in ev2]
    out.append(d2); notes_all[d["video"]]["labels"] = labels
json.dump(out, open(f"{ROOT}/index_snapped{A.suffix}.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(notes_all, open(f"{ROOT}/notes{A.suffix}.json", "w", encoding="utf-8"), ensure_ascii=False)
print("| 技法 | 件数 | 吸着不可 | 遅れ中央値(s) | p10 | p90 |"); print("|---|---|---|---|---|---|")
for c in TYPES:
    a = np.array(lagstat[c]); print(f"| {c} | {tot[c]} | {unsn[c]} | {np.median(a):.2f} | {np.percentile(a,10):.2f} | {np.percentile(a,90):.2f} |")
n_notes = sum(len(v["start"]) for v in notes_all.values()); n_pos = sum(sum(1 for l in v["labels"] if any(l)) for v in notes_all.values())
print(f"ノート総数 {n_notes}・加点付きノート {n_pos}")
