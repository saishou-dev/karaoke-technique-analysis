# -*- coding: utf-8 -*-
"""検収クリップ再生成 v2（2026-09-04）：窓＝[vt-1.0, vt-1.0+dur]（ツールの赤線＝1.0s位置に一致）・960x540/60fps・0807はrename_mappingで実ファイル名に変換"""
import csv, os, re, subprocess, time, json, concurrent.futures as cf
ROOT=r"G:\研究アーカイブ\成果物\動画解析"; SRC=r"G:\研究アーカイブ\ビデオ素材"
FF=os.path.join(os.environ["LOCALAPPDATA"],"Microsoft","WinGet","Packages","Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe","ffmpeg-9.0.1-full_build","bin","ffmpeg.exe")
html=open(os.path.join(ROOT,"review_20260902.html"),encoding="utf-8",errors="replace").read()
items=[]
for m in re.finditer(r'\{"id":"([^"]+)","video":"([^"]+)","c":"[^"]+","n":\d+,"d":\d+,"vt":([0-9.]+),"sess":"(\d+)","dur":([0-9.]+)',html):
    items.append(dict(id=m.group(1),video=m.group(2),vt=float(m.group(3)),sess=m.group(4),dur=float(m.group(5))))
ren={}
for s in os.listdir(SRC):
    p=os.path.join(SRC,s,"rename_mapping.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p,encoding="utf-8-sig")):
            o=r.get("orig") or r.get("original"); ren[(s,os.path.splitext(o)[0])]=r["renamed"]
out=os.path.join(ROOT,"clips"); os.makedirs(out,exist_ok=True)
todo=[it for it in items if not os.path.exists(os.path.join(out,it["id"]+".mp4"))]
print(f"items={len(items)} todo={len(todo)}",flush=True)
def cut(it):
    fn=ren.get((it["sess"],it["video"]),it["video"]+".mp4")
    src=os.path.join(SRC,it["sess"],fn)
    if not os.path.exists(src): return (it["id"],"nosrc:"+fn)
    st=max(0.0,it["vt"]-1.0); dur=max(1.0,it["dur"])
    dst=os.path.join(out,it["id"]+".mp4")
    cmd=[FF,"-y","-v","error","-ss",f"{st:.3f}","-i",src,"-t",f"{dur:.3f}","-vf","scale=960:540","-r","60","-c:v","libx264","-preset","veryfast","-crf","22","-c:a","aac","-b:a","64k","-movflags","+faststart",dst]
    p=subprocess.run(cmd,capture_output=True,text=True); return (it["id"],"ok" if p.returncode==0 else "err:"+p.stderr[-120:])
t0=time.time(); done=0; bad=[]
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for i,st in ex.map(cut,todo):
        done+=1
        if st!="ok": bad.append((i,st))
        if done%100==0: print(f"{done}/{len(todo)} elapsed={time.time()-t0:.0f}s bad={len(bad)}",flush=True)
print(f"FINISHED done={done} bad={len(bad)} elapsed={time.time()-t0:.0f}s",flush=True)
for b in bad[:20]: print(b)
