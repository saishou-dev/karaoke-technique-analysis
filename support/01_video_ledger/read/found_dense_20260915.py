# -*- coding: utf-8 -*-
"""目視用の密なタイル列（found_refine の自動絞り込みが背景変化で外れた候補の手動絞り込み用・2026-09-15・WSL cv2）。
   引数: video counter value start_frame end_frame step  → evidence/dense_<video>_<counter>_<value>_f<start>-<end>.png"""
import os, sys, json, cv2, numpy as np
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; RD=os.path.join(ROOT,"dataset/rescan_20260915")
name,c,v,a,b,st=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])
roi=json.load(open(os.path.join(RD,"roi",name+".json")))[c]; x1,y1,x2,y2=roi; cap=cv2.VideoCapture(os.path.join(ROOT,"bands_wao",name+".mp4")); tiles=[]
for f in range(a,b+1,st):
    cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img=cap.read()
    if not ok: break
    t=img[max(0,y1-6):y2+6, max(0,x1-6):x2+6]; t=cv2.resize(t,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
    t=cv2.copyMakeBorder(t,26,0,0,3,cv2.BORDER_CONSTANT,value=(40,40,40)); cv2.putText(t,f"f{f}",(2,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); tiles.append(t)
h=max(t.shape[0] for t in tiles); m=np.hstack([cv2.copyMakeBorder(t,0,h-t.shape[0],0,0,cv2.BORDER_CONSTANT) for t in tiles])
m=cv2.copyMakeBorder(m,28,0,0,0,cv2.BORDER_CONSTANT,value=(40,40,40)); cv2.putText(m,f"{name[:28]} {c} value={v} f{a}-{b} step{st}",(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
out=os.path.join(RD,"evidence",f"dense_{name}_{c}_{v}_f{a}-{b}.png"); cv2.imwrite(out,m); print(out)
