# -*- coding: utf-8 -*-
"""9/11 G1生成（pts補正・同期オフセット付与）と、ブラインド予測との照合（2026-09-12）。
   G1行: video,counter,count_value,delta,frame,video_time_sec(pts),stable_read_time_sec(pts),wav,audio_time_sec,multi_increment
   照合: 予測（dataset/blind_20260911/<T>_pred.csv・WAV時間軸）と G1（audio_time_sec）を非対称窓で突合 → dataset/blind_20260911/match_20260911.csv と summary
"""
import os, csv, pandas as pd, numpy as np
ROOT=r"G:\研究アーカイブ\成果物\動画解析"; RES=os.path.join(ROOT,"results_wao_20260911"); BL=os.path.join(ROOT,"dataset","blind_20260911")
sync=pd.read_csv(os.path.join(ROOT,"dataset","sync","sync_offsets_20260911.csv"),encoding="utf-8-sig")
pts={}
for fn in os.listdir(os.path.join(ROOT,"dataset","pts")):
    if "_20260911_" in fn: pts[fn[:-4]]=[float(r["pts_sec"]) for r in csv.DictReader(open(os.path.join(ROOT,"dataset","pts",fn),encoding="utf-8"))]
g1=[]
for name in sorted(os.listdir(RES)):
    ce=os.path.join(RES,name,"counter_events.csv")
    if not os.path.exists(ce) or name not in pts: continue
    ev=pd.read_csv(ce); ev=ev[ev["delta"]>0]; p=pts[name]
    srow=sync[sync["video"]==name]
    for _,e in ev.iterrows():
        fi=min(int(e["event_frame_est"]),len(p)-1); si=min(int(e["frame"]),len(p)-1); vt=p[fi]; st=p[si]
        for _,s in (srow.iterrows() if len(srow) else [(0,None)]):
            g1.append(dict(video=name,counter=e["counter"],count_value=int(e["value"]),delta=int(e["delta"]),frame=int(e["frame"]),event_frame_est=int(e["event_frame_est"]),video_time_sec=round(vt,3),stable_read_time_sec=round(st,3),
                           wav=(s["wav"] if s is not None else ""),audio_time_sec=(round(vt+float(s["offset_sec"]),3) if s is not None else None),multi_increment=int(e["delta"]>1)))
G=pd.DataFrame(g1); G.to_csv(os.path.join(ROOT,"dataset","G1_events_20260911.csv"),index=False,encoding="utf-8-sig")
print("G1 rows",len(G),G["counter"].value_counts().to_dict())
WIN={"shakuri":(-0.3,0.9),"kobushi":(-0.3,0.9),"fall":(-0.3,0.9),"vibrato":(-0.2,1.0)}
out=[]
for wav in sorted(set(G["wav"])):
    if not wav: continue
    stem=wav[:-4]; pf=os.path.join(BL,stem+"_pred.csv")
    if not os.path.exists(pf): continue
    P=pd.read_csv(pf,encoding="utf-8-sig"); T=G[(G["wav"]==wav)&(G["multi_increment"]==0)]
    for typ,(lo,hi) in WIN.items():
        for tol in (1.0,0.5):
            ex=tol-0.5; Pt=P[P["type"]==typ]["time_sec"].values; Tt=T[T["counter"]==typ]["audio_time_sec"].values
            hit_t=sum(1 for t in Tt if any(lo-ex<=t-p<=hi+ex for p in Pt)); hit_p=sum(1 for p in Pt if any(lo-ex<=t-p<=hi+ex for t in Tt))
            rec=hit_t/len(Tt) if len(Tt) else np.nan; prec=hit_p/len(Pt) if len(Pt) else np.nan
            out.append(dict(wav=stem,type=typ,tol=tol,true=len(Tt),pred=len(Pt),hit_true=hit_t,hit_pred=hit_p,recall=rec,precision=prec))
M=pd.DataFrame(out); M.to_csv(os.path.join(BL,"match_20260911.csv"),index=False,encoding="utf-8-sig")
S=M.groupby(["type","tol"]).agg(true=("true","sum"),pred=("pred","sum"),hit_true=("hit_true","sum"),hit_pred=("hit_pred","sum")).reset_index()
S["recall"]=(S["hit_true"]/S["true"]).round(3); S["precision"]=(S["hit_pred"]/S["pred"]).round(3); S["f1"]=(2*S["recall"]*S["precision"]/(S["recall"]+S["precision"])).round(3)
S.to_csv(os.path.join(BL,"match_summary_20260911.csv"),index=False,encoding="utf-8-sig"); print(S.to_string(index=False))
