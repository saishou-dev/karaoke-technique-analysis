# -*- coding: utf-8 -*-
"""検収クリップ v3（2026-09-09）：窓を1秒延長し、許容窓の外（t>=dur）に赤帯を焼き込む。
   窓＝[vt-1.0, vt-1.0+dur+1.0]（既定 dur=2.0 → 3秒クリップ・最後の1秒が赤帯＝許容窓外）。
   基準は不変：赤帯が出る前に数字上昇が見えたら○。出力 clips3/<id>.mp4（既存 clips/ は残す）。
   使い方: python gen_clips_v3_redband.py review_sample_20260807_821.html review_sample_20260628_561.html
"""
import csv, os, re, subprocess, sys, time, concurrent.futures as cf
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; SRC = r"G:\研究アーカイブ\ビデオ素材"
FF = os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "WinGet", "Packages",
                  "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe", "ffmpeg-9.0.1-full_build", "bin", "ffmpeg.exe")
EXT = 1.0
pages = sys.argv[1:] or ["review_sample_20260807_821.html", "review_sample_20260628_561.html"]
items = {}
for pg in pages:
    html = open(os.path.join(ROOT, pg), encoding="utf-8", errors="replace").read()
    for m in re.finditer(r'\{"id":"([^"]+)","video":"([^"]+)","c":"[^"]+","n":\d+,"d":\d+,"vt":([0-9.]+),"sess":"(\d+)","dur":([0-9.]+)', html):
        items[m.group(1)] = dict(id=m.group(1), video=m.group(2), vt=float(m.group(3)), sess=m.group(4), dur=float(m.group(5)))
items = list(items.values())
ren = {}
for s in os.listdir(SRC):
    p = os.path.join(SRC, s, "rename_mapping.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            o = r.get("orig") or r.get("original"); ren[(s, os.path.splitext(o)[0])] = r["renamed"]
out = os.path.join(ROOT, "clips3"); os.makedirs(out, exist_ok=True)
todo = [it for it in items if not os.path.exists(os.path.join(out, it["id"] + ".mp4"))]
print(f"items={len(items)} todo={len(todo)}", flush=True)

def cut(it):
    fn = ren.get((it["sess"], it["video"]), it["video"] + ".mp4")
    src = os.path.join(SRC, it["sess"], fn)
    if not os.path.exists(src): return (it["id"], "nosrc:" + fn)
    st = max(0.0, it["vt"] - 1.0); dur = max(1.0, it["dur"]); total = dur + EXT
    dst = os.path.join(out, it["id"] + ".mp4")
    vf = f"scale=960:540,drawbox=x=0:y=ih-12:w=iw:h=12:color=red@0.85:t=fill:enable='gte(t,{dur:.3f})'"
    cmd = [FF, "-y", "-v", "error", "-ss", f"{st:.3f}", "-i", src, "-t", f"{total:.3f}", "-vf", vf, "-r", "60",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-an", "-movflags", "+faststart", dst]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return (it["id"], "ok" if p.returncode == 0 else "err:" + p.stderr[-160:])

t0 = time.time(); done = 0; bad = []
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for i, st in ex.map(cut, todo):
        done += 1
        if st != "ok": bad.append((i, st))
        if done % 100 == 0: print(f"{done}/{len(todo)} elapsed={time.time()-t0:.0f}s bad={len(bad)}", flush=True)
print(f"FINISHED done={done} bad={len(bad)} elapsed={time.time()-t0:.0f}s", flush=True)
for b in bad[:30]: print(b)
