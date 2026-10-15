#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""found_dense の型＋glyph_boxes の検出枠を重ねる。引数: band roi.json counter f_start f_end step out.png"""
import sys,os,json,cv2,numpy as np
band,roij,c,f0,f1,step,out=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),sys.argv[7]
THRESHOLDS=(215,200,185,228,170,150,130)
def glyph_boxes(rg):
    best1=None;best2=None;bt1=bt2=None
    for T in THRESHOLDS:
        m=(rg>T).astype(np.uint8); n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
        boxes=[];junk=0
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if 28<=h<=58 and 3<=w<=44 and area>=50: boxes.append((x,y,w,h,area))
            elif area>30: junk+=area
        if not(1<=len(boxes)<=2 and junk<=1200): continue
        boxes.sort(key=lambda b:b[0])
        if len(boxes)==2:
            gap=boxes[1][0]-(boxes[0][0]+boxes[0][2]);hd=abs(boxes[0][3]-boxes[1][3]);yd=abs(boxes[0][1]-boxes[1][1])
            if gap>14 or gap<-4 or hd>8 or yd>6: continue
            if best2 is None: best2=boxes;bt2=T
        elif best1 is None: best1=boxes;bt1=T
    return (best2,bt2) if best2 is not None else (best1,bt1)
roi=json.load(open(roij));x1,y1,x2,y2=roi[c]
cap=cv2.VideoCapture(band);fps=cap.get(cv2.CAP_PROP_FPS);K=6;tiles=[]
for f in range(f0,f1+1,step):
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read()
    if not ok: continue
    rg=img.max(axis=2)[y1:y2,x1:x2]
    bx,T=glyph_boxes(rg)
    t=cv2.resize(img[y1:y2,x1:x2],None,fx=K,fy=K,interpolation=cv2.INTER_NEAREST)
    if bx:
        for i,b in enumerate(bx):
            col=(0,255,0) if i==0 else (0,0,255)
            cv2.rectangle(t,(b[0]*K,b[1]*K),((b[0]+b[2])*K,(b[1]+b[3])*K),col,2)
    t=cv2.copyMakeBorder(t,46,0,0,4,cv2.BORDER_CONSTANT,value=(40,40,40))
    cv2.putText(t,"f%d"%f,(4,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(t,"box=%s T=%s"%(len(bx) if bx else 0,T),(4,38),cv2.FONT_HERSHEY_SIMPLEX,0.45,(120,255,120),1)
    tiles.append(t)
cap.release()
h=max(t.shape[0] for t in tiles);tiles=[cv2.copyMakeBorder(t,0,h-t.shape[0],0,0,cv2.BORDER_CONSTANT) for t in tiles]
m=np.hstack(tiles);m=cv2.copyMakeBorder(m,30,0,0,0,cv2.BORDER_CONSTANT,value=(40,40,40))
cv2.putText(m,"%s %s f%d-%d  GREEN=1st box RED=2nd box"%(os.path.basename(band)[:30],c,f0,f1),(4,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
os.makedirs(os.path.dirname(out),exist_ok=True);cv2.imwrite(out,m);print("wrote",out,m.shape)
