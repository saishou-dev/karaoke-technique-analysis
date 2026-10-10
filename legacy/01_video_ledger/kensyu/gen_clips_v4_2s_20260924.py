# -*- coding: utf-8 -*-
"""検収クリップ v4・2秒窓（2026-09-14）：gen_clips_v4_2s_20260911.py のコピーを 9/24 用に直したもの（2026-09-25・元は不変）。
   窓＝[vt-0.5, vt+1.5]（2秒）・最後の0.5秒（t>=1.5）に赤帯を焼き込む（04_research「次版でクリップを短縮」）。上昇＝0.5秒位置。
   入力＝manifest CSV（event_id, video, vt）。出力 clips3/<event_id>.mp4（960x540・60fps・無音）。既存ファイルは飛ばす。
   使い方: python scripts/gen_clips_v4_2s_20260924.py dataset/kensyu_sample_20260924_200_manifest.csv [--workers 12]  ログ: scripts/gen_clips_v4_20260924.log"""
import csv, os, sys, subprocess, time, argparse, concurrent.futures as cf
ROOT = r"G:\研究アーカイブ\成果物\動画解析"; SRC = r"G:\研究アーカイブ\ビデオ素材\20260924"
FF = os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe", "ffmpeg-9.0.1-full_build", "bin", "ffmpeg.exe")
PRE, DUR, BAND = 0.5, 2.0, 1.5  # 開始＝vt-0.5・長さ2.0・赤帯開始1.5
ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("--workers", type=int, default=12); a = ap.parse_args(); os.chdir(ROOT)
items = list(csv.DictReader(open(a.manifest, encoding="utf-8-sig"))); out = "clips3"; os.makedirs(out, exist_ok=True)
log = open(os.path.join("scripts", "gen_clips_v4_20260924.log"), "a", encoding="utf-8")
def L(s): print(s, flush=True); log.write(s + "\n"); log.flush()
todo = [it for it in items if not os.path.exists(os.path.join(out, it["event_id"] + ".mp4"))]
L(f"START {time.strftime('%H:%M:%S')} items={len(items)} todo={len(todo)}")
def cut(it):
    src = os.path.join(SRC, it["video"] + ".mp4"); vt = float(it["vt"])
    if not os.path.exists(src): return it["event_id"], "nosrc"
    st = vt - PRE; note = ""
    if st < 0: note = f" (vt={vt}<0.5: 先頭から切り出し・上昇位置ずれ)"; st = 0.0
    dst = os.path.join(out, it["event_id"] + ".mp4"); tmp = dst + ".part.mp4"
    vf = f"scale=960:540,drawbox=x=0:y=ih-12:w=iw:h=12:color=red@0.85:t=fill:enable='gte(t,{BAND:.3f})'"
    cmd = [FF, "-y", "-v", "error", "-ss", f"{st:.3f}", "-i", src, "-t", f"{DUR:.3f}", "-vf", vf, "-r", "60", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-an", "-movflags", "+faststart", tmp]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0: os.replace(tmp, dst); return it["event_id"], "ok" + note
    return it["event_id"], "err:" + p.stderr[-160:]
t0 = time.time(); done = 0; bad = []
with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
    for eid, st in ex.map(cut, todo):
        done += 1
        if not st.startswith("ok"): bad.append((eid, st))
        elif st != "ok": L(f"{eid}: {st}")
        if done % 25 == 0: L(f"{done}/{len(todo)} elapsed={time.time()-t0:.0f}s bad={len(bad)}")
L(f"FINISHED {time.strftime('%H:%M:%S')} done={done} bad={len(bad)} elapsed={time.time()-t0:.0f}s")
for b in bad: L(str(b))
