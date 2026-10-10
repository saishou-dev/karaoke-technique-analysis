# -*- coding: utf-8 -*-
"""rescan_missing_20260915.py の FOUND 候補を目視するための比較タイル（2026-09-15・WSL cv2）。
   各候補について帯動画の ROI（較正済み roi/<video>.json）を候補フレームの前後 [-90,-30,-8,-2,+3,+10,+30,+90] で切り出し 4倍に拡大して横に並べる。
   出力: dataset/rescan_20260915/evidence/montage_<video>_<counter>_<value>_f<frame>.png（既存の evidence タイルは触らない）"""
import os, csv, json, cv2, numpy as np
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; RD=os.path.join(ROOT,"dataset/rescan_20260915")
OFF=[-90,-30,-8,-2,3,10,30,90]; K=4
rows=[r for r in csv.DictReader(open(os.path.join(RD,"rescan_missing.csv"),encoding="utf-8-sig")) if r["kind"]=="FOUND"]
print("FOUND rows:",len(rows))
for r in rows:
    name=r["video"]; c=r["counter"]; f0=int(r["frame"]); roi=json.load(open(os.path.join(RD,"roi",name+".json")))
    x1,y1,x2,y2=roi[c]; cap=cv2.VideoCapture(os.path.join(ROOT,"bands_wao",name+".mp4")); tiles=[]
    for o in OFF:
        f=max(0,f0+o); cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img=cap.read()
        if not ok: continue
        t=img[max(0,y1-12):y2+12, max(0,x1-12):x2+12]; t=cv2.resize(t,None,fx=K,fy=K,interpolation=cv2.INTER_CUBIC)
        t=cv2.copyMakeBorder(t,28,0,0,4,cv2.BORDER_CONSTANT,value=(40,40,40)); cv2.putText(t,f"{o:+d}",(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        tiles.append(t)
    cap.release(); h=max(t.shape[0] for t in tiles); tiles=[cv2.copyMakeBorder(t,0,h-t.shape[0],0,0,cv2.BORDER_CONSTANT) for t in tiles]
    m=np.hstack(tiles); m=cv2.copyMakeBorder(m,30,0,0,0,cv2.BORDER_CONSTANT,value=(40,40,40))
    cv2.putText(m,f"{name[:30]} {c} value={r['value']} f{f0} t={r['time']}",(4,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    out=os.path.join(RD,"evidence",f"montage_{name}_{c}_{r['value']}_f{f0}.png"); cv2.imwrite(out,m); print("wrote",out)
