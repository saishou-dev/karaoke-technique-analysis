# -*- coding: utf-8 -*-
"""C3 補助：持続を窓で切らずに測る（DAM の vibratoTotalSecond と比べるための候補）。
(a) 加点ごと＝窓 [t-6.0, t+0.5] で連鎖を全部拾い、主窓 [t-2.5, t-1.0] と重なる連鎖の全長（極値間＋半周期）
(b) 曲全体＝採点区間の全有声区間から連鎖を全部拾い、長さ閾値ごとに合計秒と本数（加点に紐づけない）
測定の定義は vibrato_depth.py の measure() と同一（移動中央値 0.3 s・山谷差 >= 15 cent・半周期 1/18〜1/6 s・対 3 つ以上）。
出力: c3_fullspan_events.csv・c3_fullspan_songs.csv
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent; VD = HERE.parent; ROOT = VD.parent
FR = 100; MED = 0.3; DMIN = 15.0; FMIN, FMAX = 3.0, 9.0
HMIN, HMAX = 1.0/(2*FMAX), 1.0/(2*FMIN)
MAIN = (-2.5, -1.0)

index = json.load(open(ROOT/"index.json", encoding="utf-8"))
test = {d["video"]: d for d in index if d["session"]=="20260924"}
ev = pd.read_csv(VD/"vibrato_depth_events_win-2.5_-1.0.csv", encoding="utf-8-sig")
ev = ev[ev.kind=="truth"].copy()

def chains_in(z, ta, tb):
    """区間 [ta,tb] の有声区間ごとに残差→山谷→連鎖を全部返す。各連鎖=(t_start,t_end,depth_med,rate_chain,n_pairs)"""
    hop = float(z["hop"]); t0 = float(z["t0"]); st = z["st"]; v = z["voiced"]>0.5; T = len(st)
    i0 = max(0, int(round((ta-t0)/hop))); i1 = min(T, int(round((tb-t0)/hop)))
    cents = st[i0:i1].astype(np.float64)*100.0; vo = v[i0:i1]; n = len(vo)
    runs=[]; i=0
    while i<n:
        if not vo[i]: i+=1; continue
        j=i
        while j<n and vo[j]: j+=1
        if j-i >= int(MED*FR): runs.append((i,j))
        i=j
    kmed = int(MED*FR)|1
    out=[]
    for (i,j) in runs:
        c = cents[i:j]; r = c - median_filter(c, size=kmed, mode="nearest")
        pk,_ = find_peaks(r, distance=3); tr,_ = find_peaks(-r, distance=3)
        e = sorted([(p+i, r[p], 1) for p in pk] + [(q+i, r[q], -1) for q in tr])
        al=[]
        for x in e:
            if al and al[-1][2]==x[2]:
                if (x[2]==1 and x[1]>al[-1][1]) or (x[2]==-1 and x[1]<al[-1][1]): al[-1]=x
            else: al.append(x)
        k=0
        while k < len(al)-1:
            m=k
            while m < len(al)-1 and abs(al[m+1][1]-al[m][1])>=DMIN and HMIN <= (al[m+1][0]-al[m][0])*hop <= HMAX: m+=1
            if m-k >= 3:
                ch = al[k:m+1]
                pp = [abs(ch[q+1][1]-ch[q][1]) for q in range(len(ch)-1)]
                half = [(ch[q+1][0]-ch[q][0])*hop for q in range(len(ch)-1)]
                rate = 1.0/(2*float(np.median(half)))
                out.append(dict(t_start=t0+(i0+ch[0][0])*hop, t_end=t0+(i0+ch[-1][0])*hop, depth=float(np.median(pp)), rate_chain=rate, n_pairs=len(pp)))
            k = max(m, k+1)
    return out

FE = {}
def feats(v):
    if v not in FE: FE[v] = np.load(ROOT/"feats"/f"{v}.npz")
    return FE[v]

# (a) 加点ごと
rows=[]
for r in ev.itertuples():
    z = feats(r.video); t = r.t_event
    ch = chains_in(z, t-6.0, t+0.5)
    a, b = t+MAIN[0], t+MAIN[1]
    ov = [(min(c["t_end"],b)-max(c["t_start"],a), c) for c in ch]
    ov = [(o,c) for o,c in ov if o>0]
    if ov:
        o,c = max(ov, key=lambda x: x[0]); how="overlap"
    else:
        cand = [c for c in ch if c["t_end"]>=t-4.0 and c["t_start"]<=t]
        if cand: c = max(cand, key=lambda c: c["t_end"]-c["t_start"]); how="nearest"
        else: c=None; how="none"
    d = dict(video=r.video, t_event=t, sub=r.sub, win_duration=r.duration_s, win_depth=r.depth_cents, how=how)
    if c:
        span = c["t_end"]-c["t_start"]
        d.update(full_start=round(c["t_start"]-t,2), full_end=round(c["t_end"]-t,2), full_duration=round(span,2), full_duration_corr=round(span+0.5/c["rate_chain"],2),
                 full_depth=round(c["depth"],1), full_rate_chain=round(c["rate_chain"],2), full_n_pairs=c["n_pairs"])
    else:
        d.update(full_start=np.nan, full_end=np.nan, full_duration=0.0, full_duration_corr=0.0, full_depth=np.nan, full_rate_chain=np.nan, full_n_pairs=0)
    rows.append(d)
E = pd.DataFrame(rows)
E.to_csv(HERE/"c3_fullspan_events.csv", index=False, encoding="utf-8-sig")

# (b) 曲全体
srows=[]
for v,d in test.items():
    z = feats(v); lo,hi = d["span"]
    ch = chains_in(z, lo, hi)
    dur = np.array([c["t_end"]-c["t_start"]+0.5/c["rate_chain"] for c in ch]) if ch else np.array([])
    row = dict(video=v, n_chains_all=len(ch), song_chain_sec_all=round(float(dur.sum()),2))
    for th in (0.3, 0.6, 1.0, 1.5, 2.0):
        m = dur>=th
        row[f"n_chains_ge{th}"] = int(m.sum()); row[f"song_chain_sec_ge{th}"] = round(float(dur[m].sum()),2)
    e = E[E.video==v]
    row["full_sum"] = round(float(e.full_duration.sum()),2); row["full_sum_corr"] = round(float(e.full_duration_corr.sum()),2)
    row["full_mean_corr"] = round(float(e.full_duration_corr.mean()),2); row["n_none"] = int((e.how=="none").sum())
    srows.append(row)
S = pd.DataFrame(srows)
S.to_csv(HERE/"c3_fullspan_songs.csv", index=False, encoding="utf-8-sig")
print(E.describe().round(2).to_string())
print(E.how.value_counts())
print(S.to_string())
