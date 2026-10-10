# -*- coding: utf-8 -*-
"""目視で「増えている」と確認した候補について、増加フレームを絞り込む（2026-09-15・WSL cv2）。
   montage で「前の値」が見えた offset（before）と「後の値」が見えた offset（after）の間を1フレームずつ走査し、
   ROI（Laplacian）の NCC が after 基準 > before 基準 になった最初のフレーム（3フレーム連続）を event_frame とする。
   出力: dataset/rescan_20260915/found_refined.csv・evidence/refined_<video>_<counter>_<value>_f<frame>.png（±3フレームの拡大タイル）"""
import os, csv, json, sys, cv2, numpy as np
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; RD=os.path.join(ROOT,"dataset/rescan_20260915")
# (video, counter, value, f0, before_off, after_off)
C=[("ニュー・マイ・ノーマル_20260915_111844","hammering",6,14041,-30,-8),
   ("天国_20260915_113749","hammering",22,13651,-90,-30),
   ("天国_20260915_113749","hammering",23,13651,-2,10),
   ("地球儀 with Vaundy_20260915_120243","shakuri",9,11821,-30,-8),
   ("おもかげ (produced by Vaundy)_20260915_120810","shakuri",3,4207,-30,-8),
   ("おもかげ (produced by Vaundy)_20260915_120810","shakuri",4,5086,-30,-8),
   ("おもかげ (produced by Vaundy)_20260915_120810","shakuri",5,5462,-30,-8),
   ("Mela!(生音)_20260915_125548","hammering",6,12855,-90,-30),
   ("Mela!(生音)_20260915_131253","shakuri",11,11972,-90,-30),
   ("Mela!(生音)_20260915_131253","accent",10,11555,-90,-30)]
def prep(t): g=t.max(axis=2).astype(np.float32); return cv2.Laplacian(cv2.GaussianBlur(g,(3,3),0),cv2.CV_32F,ksize=3)
def ncc(a,b): a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/max(1e-6,np.linalg.norm(a)*np.linalg.norm(b)))
def grab(cap,f,roi):
    cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img=cap.read(); x1,y1,x2,y2=roi; return img[y1:y2,x1:x2] if ok else None
out=[]
for name,c,v,f0,bo,ao in C:
    roi=json.load(open(os.path.join(RD,"roi",name+".json")))[c]; cap=cv2.VideoCapture(os.path.join(ROOT,"bands_wao",name+".mp4"))
    B=prep(grab(cap,f0+bo,roi)); A=prep(grab(cap,f0+ao,roi)); cap.set(cv2.CAP_PROP_POS_FRAMES,f0+bo); run=0; ev=None; trace=[]
    for f in range(f0+bo,f0+ao+1):
        ok,img=cap.read()
        if not ok: break
        x1,y1,x2,y2=roi; P=prep(img[y1:y2,x1:x2]); sa=ncc(P,A); sb=ncc(P,B); trace.append((f,round(sa,2),round(sb,2)))
        run=run+1 if sa>sb else 0
        if run>=3 and ev is None: ev=f-2
    if ev is None: ev=f0+ao
    tiles=[]
    for o in (-3,-1,0,1,3):
        t=grab(cap,ev+o,[max(0,roi[0]-12),max(0,roi[1]-12),roi[2]+12,roi[3]+12]); t=cv2.resize(t,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
        t=cv2.copyMakeBorder(t,28,0,0,4,cv2.BORDER_CONSTANT,value=(40,40,40)); cv2.putText(t,f"{o:+d}",(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1); tiles.append(t)
    h=max(t.shape[0] for t in tiles); m=np.hstack([cv2.copyMakeBorder(t,0,h-t.shape[0],0,0,cv2.BORDER_CONSTANT) for t in tiles])
    m=cv2.copyMakeBorder(m,30,0,0,0,cv2.BORDER_CONSTANT,value=(40,40,40)); cv2.putText(m,f"{name[:28]} {c} value={v} refined f{ev} (cand f{f0})",(4,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    cv2.imwrite(os.path.join(RD,"evidence",f"refined_{name}_{c}_{v}_f{ev}.png"),m); cap.release()
    out.append(dict(video=name,counter=c,value=v,cand_frame=f0,event_frame=ev,before_off=bo,after_off=ao)); print(name[:20],c,v,"cand",f0,"->",ev, trace[max(0,ev-(f0+bo)-2):ev-(f0+bo)+4])
with open(os.path.join(RD,"found_refined.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("DONE")
