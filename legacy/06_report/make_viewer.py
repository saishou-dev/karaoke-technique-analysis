#!/usr/bin/env python3
"""検証ビューア生成: ピッチ曲線+技法イベント重畳のセルフコンテインドHTML
UI原則: ページ構造不変・近接ポップ・やめるボタンなし(外側タップ)・技法6色
"""
import csv
import json
import os
import sys
import glob

BASE = sys.argv[1]
OUT = os.path.join(BASE, "viewer_20260628.html")

events = {}
with open(os.path.join(BASE, "dataset", "G1_events_20260628.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if not r["audio_time_sec"]:
            continue
        events.setdefault(r["video"], []).append({
            "t": float(r["audio_time_sec"]),
            "c": r["counter"],
            "n": int(r["count_value"]),
            "d": int(r["delta"]),
        })

songs = {}
for p in sorted(glob.glob(os.path.join(BASE, "dataset", "pitch", "*_f0.csv"))):
    video = os.path.basename(p)[:-7]
    pts = []
    with open(p, encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i % 2:
                continue  # 50Hzに間引き
            if r["voiced"] == "1":
                pts.append([round(float(r["audio_time_sec"]), 2), round(float(r["f0_hz"]), 1)])
    songs[video] = {"pitch": pts, "events": events.get(video, [])}

data_js = json.dumps(songs, ensure_ascii=False, separators=(",", ":"))

html = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>カラオケ研究 検証ビューア 6/28</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#14161c;color:#e8e8ee}
#bar{padding:8px 12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select{font-size:15px;padding:4px 8px;background:#22252e;color:#e8e8ee;border:1px solid #444;border-radius:6px}
.lg{display:flex;gap:10px;font-size:12px;align-items:center}
.lg span{display:inline-flex;align-items:center;gap:4px}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
#wrap{position:relative}
canvas{display:block;width:100vw;height:calc(100vh - 60px)}
#pop{position:absolute;display:none;background:#22252e;border:1px solid #555;border-radius:8px;padding:6px 10px;font-size:12px;pointer-events:none;white-space:nowrap;box-shadow:0 4px 14px rgba(0,0,0,.5)}
#hint{font-size:11px;color:#9aa}
</style></head><body>
<div id="bar">
  <select id="song"></select>
  <div class="lg">
    <span><i class="sw" style="background:#FFA726"></i>しゃくり</span>
    <span><i class="sw" style="background:#339CFF"></i>こぶし</span>
    <span><i class="sw" style="background:#D06BC8"></i>フォール</span>
    <span><i class="sw" style="background:#34C759"></i>ビブラート</span>
  </div>
  <span id="hint">ドラッグ=移動 / ホイール=ズーム / ダブルクリック=全体 / マーカーに近づくと詳細</span>
</div>
<div id="wrap"><canvas id="cv"></canvas><div id="pop"></div></div>
<script>
const DATA=__DATA__;
const COL={shakuri:"#FFA726",kobushi:"#339CFF",fall:"#D06BC8",vibrato:"#34C759"};
const NAME={shakuri:"しゃくり",kobushi:"こぶし",fall:"フォール",vibrato:"ビブラート"};
const cv=document.getElementById("cv"),ctx=cv.getContext("2d"),pop=document.getElementById("pop");
const sel=document.getElementById("song");
Object.keys(DATA).forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent=k.replace(/_\\d{8}_\\d{6}$/,"");sel.appendChild(o)});
let cur=null,t0=0,t1=1,drag=null;
function noteY(f){return 69+12*Math.log2(f/440)}
function view(){
  const d=DATA[sel.value];cur=d;
  const ts=d.pitch.map(p=>p[0]);
  t0=Math.min(...ts)-2;t1=Math.max(...ts)+2;draw();
}
function draw(){
  const W=cv.clientWidth,H=cv.clientHeight;
  if(cv.width!==W*devicePixelRatio){cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio}
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  ctx.fillStyle="#14161c";ctx.fillRect(0,0,W,H);
  if(!cur)return;
  const pts=cur.pitch.filter(p=>p[0]>=t0&&p[0]<=t1);
  let lo=45,hi=80;
  if(pts.length){const ys=pts.map(p=>noteY(p[1]));lo=Math.min(...ys)-2;hi=Math.max(...ys)+2}
  const X=t=>(t-t0)/(t1-t0)*W, Y=n=>H-((n-lo)/(hi-lo))*(H-30)-15;
  ctx.strokeStyle="#2a2e3a";ctx.fillStyle="#778";ctx.font="10px sans-serif";
  for(let n=Math.ceil(lo);n<=hi;n++){if(n%2)continue;const y=Y(n);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  const step=Math.max(1,Math.round((t1-t0)/12));
  for(let t=Math.ceil(t0);t<t1;t+=step){const x=X(t);ctx.strokeStyle="#232734";ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();ctx.fillText(t+"s",x+3,H-4)}
  for(const e of cur.events){
    if(e.t<t0||e.t>t1)continue;
    const x=X(e.t);
    ctx.strokeStyle=COL[e.c];ctx.globalAlpha=.9;ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(x,20);ctx.lineTo(x,H-20);ctx.stroke();ctx.lineWidth=1;ctx.globalAlpha=1;
    ctx.fillStyle=COL[e.c];ctx.fillText(e.n,x-6,14)
  }
  ctx.fillStyle="#7fd4ff";
  for(const p of pts){ctx.fillRect(X(p[0])-1,Y(noteY(p[1]))-1,2,2)}
}
sel.onchange=view;
window.addEventListener("resize",draw);
cv.addEventListener("wheel",ev=>{ev.preventDefault();
  const W=cv.clientWidth,f=ev.deltaY>0?1.25:0.8,tc=t0+(ev.offsetX/W)*(t1-t0);
  t0=tc-(tc-t0)*f;t1=tc+(t1-tc)*f;draw()},{passive:false});
cv.addEventListener("mousedown",ev=>{drag={x:ev.offsetX,t0,t1}});
window.addEventListener("mouseup",()=>drag=null);
cv.addEventListener("dblclick",()=>{const ts=cur.pitch.map(p=>p[0]);t0=Math.min(...ts)-2;t1=Math.max(...ts)+2;draw()});
cv.addEventListener("mousemove",ev=>{
  if(drag){const W=cv.clientWidth,dt=(drag.x-ev.offsetX)/W*(drag.t1-drag.t0);t0=drag.t0+dt;t1=drag.t1+dt;draw();return}
  if(!cur)return;
  const W=cv.clientWidth,t=t0+(ev.offsetX/W)*(t1-t0),px=8/W*(t1-t0);
  const near=cur.events.filter(e=>Math.abs(e.t-t)<px);
  if(near.length){
    const e=near[0];
    pop.innerHTML=NAME[e.c]+" "+e.n+"回目"+(e.d>1?"(+"+e.d+")":"")+"<br>"+e.t.toFixed(2)+"秒(録音基準)";
    pop.style.left=(ev.offsetX+14)+"px";pop.style.top=(ev.offsetY+10)+"px";pop.style.display="block";
  } else pop.style.display="none";
});
view();
</script></body></html>
"""
html = html.replace("__DATA__", data_js)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"viewer: {OUT} ({os.path.getsize(OUT)/1e6:.1f}MB, songs={len(songs)})")
