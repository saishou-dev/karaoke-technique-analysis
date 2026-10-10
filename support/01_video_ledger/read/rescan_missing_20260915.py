# -*- coding: utf-8 -*-
"""欠落した増分の部分再走査 9/15版（rescan_missing_20260911.py の複製・ROI＝dataset/rescan_20260915/roi/<video>.json・入力＝results_wao_20260915_rescan・verify_20260915_rescan.csv で DIFF の曲・カウンターについて、
   最後に読めたイベント以降（および増分の飛び＝delta>1 の直前区間）を全フレーム走査し、緩い閾値（NCC 0.60・全閾値）で
   f+1..T の各値の初出（3フレーム以上連続）を探して追記候補を出す。証拠としてROIのタイル画像も保存。
   使い方: python rescan_missing_20260915.py（WSL） 出力: dataset/rescan_20260915/rescan_missing.csv・rescan_20260915/evidence/*.png（候補のみ・G1 へは未反映）"""
import os, csv, json, sys, numpy as np, cv2
ROOT="/mnt/g/研究アーカイブ/成果物/動画解析"; RES=os.path.join(ROOT,"results_wao_20260915_rescan"); RD=os.path.join(ROOT,"dataset/rescan_20260915")
sys.path.insert(0, os.path.expanduser("~/karaoke-vision"))
ROI_DEFAULT=json.load(open(os.path.join(ROOT,"templates/wao_20260816/roi_wao.json")))
LAB={int(k):v for k,v in json.load(open(os.path.join(ROOT,"templates/wao_20260816/cluster_reps_wao_ext_labels.json"))).items()}
reps=np.load(os.path.join(ROOT,"templates/wao_20260816/cluster_reps_wao_ext.npy"))
RF=[]
for r in reps:
    v=r.ravel()-r.mean(); RF.append(v/max(1e-6,np.linalg.norm(v)))
RF=np.stack(RF); NORM=(28,40); THR=(228,215,200,185,170,150,130,110)
def boxes(rg):
    best1=best2=None
    for T in THR:
        m=(rg>T).astype(np.uint8); n,lab,st,_=cv2.connectedComponentsWithStats(m,8); bx=[]; junk=0
        for i in range(1,n):
            x,y,w,h,a=st[i]
            if 26<=h<=60 and 3<=w<=46 and a>=40: bx.append((x,y,w,h))
            elif a>30: junk+=a
        if not(1<=len(bx)<=2 and junk<=1500): continue
        bx.sort(key=lambda b:b[0])
        if len(bx)==2:
            gap=bx[1][0]-(bx[0][0]+bx[0][2]);
            if gap>16 or gap<-4 or abs(bx[0][3]-bx[1][3])>9 or abs(bx[0][1]-bx[1][1])>7: continue
            if best2 is None: best2=bx
        elif best1 is None: best1=bx
    return best2 if best2 is not None else best1
def classify(rg,b,ncc):
    x,y,w,h=b; p=2; g=rg[max(0,y-p):y+h+p, max(0,x-p):x+w+p].astype(np.float32); g=(g-g.min())/max(1.0,g.max()-g.min()); g=cv2.resize(g,NORM)
    v=g.ravel()-g.mean(); n=np.linalg.norm(v)
    if n<1e-6: return None
    sc=RF@(v/n); bi=int(np.argmax(sc))
    return LAB.get(bi) if sc[bi]>=ncc else None
def read_val(rg,ncc=0.60):
    bx=boxes(rg)
    if not bx: return None
    ds=[]
    for b in bx:
        d=classify(rg,b,ncc)
        if d is None: return None
        ds.append(d)
    return ds[0] if len(ds)==1 else ds[0]*10+ds[1]
M={"shakuri":"shakuriCount","kobushi":"kobushiCount","fall":"fallCount","vibrato":"vibratoCount","accent":"accentCount","hammering":"hammeringOnCount"}
ver=list(csv.DictReader(open(os.path.join(ROOT,"dataset/verify_20260915_rescan.csv"),encoding="utf-8-sig")))
os.makedirs(os.path.join(RD,"evidence"),exist_ok=True)
out=[]
for r in ver:
    if r["status"]!="DIFF": continue
    name=r["video"]; ROI=json.load(open(os.path.join(RD,"roi",name+".json"))) if os.path.exists(os.path.join(RD,"roi",name+".json")) else ROI_DEFAULT; ev=list(csv.DictReader(open(os.path.join(RES,name,"counter_events.csv"))))
    cap=cv2.VideoCapture(os.path.join(ROOT,"bands_wao",name+".mp4")); N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    for c in M:
        target=int(r["csv_"+c]); final=int(r["read_"+c])
        if final>=target:
            if final>target: out.append(dict(video=name,counter=c,kind="OVERCOUNT",value="",frame="",time="",note=f"read {final} > csv {target}: 目視で偽イベントを特定"))
            continue
        evs=[e for e in ev if e["counter"]==c and int(e["delta"])>0]
        last_fr=int(evs[-1]["frame"]) if evs else 0
        # 走査区間＝最後のイベント以降（全フレーム）＋ delta>1 の直前2秒
        ranges=[(last_fr, N)]
        for e in evs:
            if int(e["delta"])>1: ranges.append((max(0,int(e["frame"])-int(2*fps)), int(e["frame"])))
        x1,y1,x2,y2=ROI[c]; found={}
        for (a,b) in ranges:
            cap.set(cv2.CAP_PROP_POS_FRAMES,a); run_v=None; run_n=0; f=a
            while f<b:
                ok,img=cap.read()
                if not ok: break
                v=read_val(img.max(axis=2)[y1:y2,x1:x2])
                if v is not None and v==run_v: run_n+=1
                else: run_v=v; run_n=1
                if v is not None and final<v<=target and run_n>=3 and v not in found:
                    found[v]=f-2
                    tile=img[max(0,y1-10):y2+10, max(0,x1-10):x2+10]; cv2.imwrite(os.path.join(RD,"evidence",f"{name}_{c}_{v}_f{f-2}.png"),tile)
                f+=1
        for v in range(final+1,target+1):
            if v in found: out.append(dict(video=name,counter=c,kind="FOUND",value=v,frame=found[v],time=round(found[v]/fps,3),note="rescan ncc0.60 run>=3"))
            else: out.append(dict(video=name,counter=c,kind="NOT_FOUND",value=v,frame="",time="",note="要目視（帯フェード/演出遮蔽の可能性）"))
        print(name,c,final,"->",target,found,flush=True)
    cap.release()
with open(os.path.join(ROOT,"dataset/rescan_20260915/rescan_missing.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=["video","counter","kind","value","frame","time","note"]); w.writeheader(); w.writerows(out)
print("WROTE", len(out))
