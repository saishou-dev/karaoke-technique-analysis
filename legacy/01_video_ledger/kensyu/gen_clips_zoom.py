# -*- coding: utf-8 -*-
"""検収クリップの拡大版（2026-09-08）：既存クリップ（960x540）の上に、bands_wao の帯動画を左右2分割で5倍相当に拡大して積む（出力 960x724）。
使い方: python gen_clips_zoom.py <manifest.csv> [--workers 6]
帯動画（bands_wao/<video>.mp4）は元動画と同じタイムラインなので、クリップと同じ開始時刻 st=max(0,vt-1.0) で切る。"""
import csv, os, re, json, sys, subprocess, time, argparse, concurrent.futures as cf
ROOT = r'G:\研究アーカイブ\成果物\動画解析'
FF = os.path.join(os.environ['LOCALAPPDATA'], 'Microsoft', 'WinGet', 'Packages', 'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe', 'ffmpeg-9.0.1-full_build', 'bin', 'ffmpeg.exe')
ap = argparse.ArgumentParser(); ap.add_argument('manifest'); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--out', default='clips_zoom')
a = ap.parse_args(); os.chdir(ROOT)
src = open('review_20260902.html', encoding='utf-8', errors='replace').read()
i = src.find('const EV='); j = src.find('];', i)
DUR = {e['id']: float(e['dur']) for e in json.loads(src[i + len('const EV='):j + 1])}
rows = list(csv.DictReader(open(a.manifest, encoding='utf-8-sig')))
os.makedirs(a.out, exist_ok=True)
def one(r):
    eid = r['event_id']; vt = float(r['vt']); dur = max(1.0, DUR.get(eid, 2.0)); st = max(0.0, vt - 1.0)
    clip = os.path.join('clips', eid + '.mp4'); band = os.path.join('bands_wao', r['video'] + '.mp4'); dst = os.path.join(a.out, eid + '.mp4')
    if os.path.exists(dst): return eid, 'exists'
    if not os.path.exists(clip): return eid, 'noclip'
    if not os.path.exists(band): return eid, 'noband'
    flt = ('[1:v]split[l][r];[l]crop=iw/2:ih:0:0,scale=960:-2[L];[r]crop=iw/2:ih:iw/2:0,scale=960:-2[R];'
           '[L][R]vstack[B];[0:v]scale=960:540[base];[B][base]vstack[out]')
    cmd = [FF, '-y', '-v', 'error', '-i', clip, '-ss', f'{st:.3f}', '-t', f'{dur:.3f}', '-i', band,
           '-filter_complex', flt, '-map', '[out]', '-map', '0:a?', '-r', '60', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22',
           '-c:a', 'copy', '-movflags', '+faststart', '-shortest', dst]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return eid, 'ok' if p.returncode == 0 else 'err:' + p.stderr[-160:]
t0 = time.time(); res = []
with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
    for k, (eid, st) in enumerate(ex.map(one, rows), 1):
        res.append((eid, st))
        if k % 50 == 0: print(f'{k}/{len(rows)} {time.time()-t0:.0f}s', flush=True)
from collections import Counter
print('DONE', Counter(s if s in ('ok', 'exists', 'noclip', 'noband') else 'err' for _, s in res), f'{time.time()-t0:.0f}s')
for eid, s in res:
    if s.startswith('err'): print(eid, s)
