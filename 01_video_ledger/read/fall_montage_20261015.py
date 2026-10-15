#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""found_montage_20260915.py の型。引数: <band.mp4> <roi.json> <counter> <frame> <outpng>
   前後 [-90,-30,-8,-2,+3,+10,+30,+90] の ROI を4倍拡大して横に並べる。"""
import sys, os, json, cv2, numpy as np
band,roij,c,f0,out=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5]
OFF=[-90,-30,-8,-2,3,10,30,90]; K=4
roi=json.load(open(roij)); x1,y1,x2,y2=roi[c]
cap=cv2.VideoCapture(band); fps=cap.get(cv2.CAP_PROP_FPS); tiles=[]
for o in OFF:
    f=max(0,f0+o); cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img=cap.read()
    if not ok: continue
    t=img[max(0,y1-12):y2+12, max(0,x1-12):x2+12]
    t=cv2.resize(t,None,fx=K,fy=K,interpolation=cv2.INTER_CUBIC)
    t=cv2.copyMakeBorder(t,28,0,0,4,cv2.BORDER_CONSTANT,value=(40,40,40))
    cv2.putText(t,"%+d f%d"%(o,f),(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    tiles.append(t)
cap.release()
h=max(t.shape[0] for t in tiles); tiles=[cv2.copyMakeBorder(t,0,h-t.shape[0],0,0,cv2.BORDER_CONSTANT) for t in tiles]
m=np.hstack(tiles); m=cv2.copyMakeBorder(m,30,0,0,0,cv2.BORDER_CONSTANT,value=(40,40,40))
cv2.putText(m,"%s %s f%d t=%.2fs"%(os.path.basename(band)[:34],c,f0,f0/fps),(4,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
os.makedirs(os.path.dirname(out),exist_ok=True); cv2.imwrite(out,m); print("wrote",out)
