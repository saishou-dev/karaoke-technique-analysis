# -*- coding: utf-8 -*-
"""帯クロップ v1（2026-09-12・Windowsネイティブ ffmpeg・3並列・.part→rename）。入力＝dataset/band_probe_20260911.csv。完成済みはスキップ。
   フィルタ＝crop(余白20/48)→rotate(-angle)→crop→scale=2480:120。fpsは元のまま（ptsの対応を保つ）。"""
import os, csv, math, subprocess, time, concurrent.futures as cf
ROOT=r"G:\研究アーカイブ\成果物\動画解析"; VID=r"G:\研究アーカイブ\ビデオ素材\20260911"
FF=os.path.join(os.environ["LOCALAPPDATA"],"Microsoft","WinGet","Packages","Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe","ffmpeg-9.0.1-full_build","bin","ffmpeg.exe")
rows=list(csv.DictReader(open(os.path.join(ROOT,"dataset","band_probe_20260911.csv"),encoding="utf-8")))
def job(r):
    name=r["video"]; out=os.path.join(ROOT,"bands_wao",name+".mp4")
    if os.path.exists(out): return name,"skip"
    xL,yL,w,h=int(r["xL"]),int(r["yL"]),int(r["w"]),int(r["h"]); ang=float(r["angle_deg"]); mx,my=20,48
    vf=f"crop={w+2*mx}:{h+2*my}:{xL-mx}:{yL-my},rotate={-math.radians(ang):.6f}:c=black,crop={w}:{h}:{mx}:{my},scale=2480:120"
    t0=time.time(); p=subprocess.run([FF,"-y","-v","error","-threads","4","-i",os.path.join(VID,name+".mp4"),"-vf",vf,"-c:v","libx264","-crf","15","-preset","fast","-an",out+".part.mp4"],capture_output=True,text=True)
    if p.returncode==0: os.replace(out+".part.mp4",out); return name,f"ok {time.time()-t0:.0f}s"
    return name,"err "+p.stderr[-200:]
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    for n,s in ex.map(job,rows): print(n,s,flush=True)
print("ALL DONE",flush=True)
