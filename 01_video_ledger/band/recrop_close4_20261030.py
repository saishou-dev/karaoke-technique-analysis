# -*- coding: utf-8 -*-
"""帯の付け直し（再クロップ）2026-10-30。wincrop_20260915.py と同じ ffmpeg の変換・引数で、再プローブ行
   （dataset/close4_20261030/reprobe_rows.csv）から帯動画を作り直し、帯pts表＋ゲート（kv_config_20260915.Cfg.band_pts_gate）を通す。
   既存の bands_wao・dataset/pts_bands_<S> には書かない。出力＝results_wao_<S>/_close4_20261030/{bands,pts}/（ゲート表 pts/_gate.csv）。
   追加の検算＝新しい帯pts表が既存の dataset/pts_bands_<S>/<video>.csv と同一か（passthrough なら同じ元フレーム＝同一のはず）。
   CPU＝ffmpeg -threads 4・2並列・優先度 BelowNormal（同時に別の実験が走っているため）。
   使い方：python recrop_close4_20261030.py <session> [video ...]（省略＝reprobe_rows.csv の全行）
"""
import os, sys, csv, math, time, hashlib, subprocess, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, FFMPEG, ROOT
S = sys.argv[1]; only = sys.argv[2:]
C = Cfg(S)
OUT = os.path.join(ROOT, f"results_wao_{S}", "_close4_20261030"); BD = os.path.join(OUT, "bands"); PD = os.path.join(OUT, "pts")
os.makedirs(BD, exist_ok=True); os.makedirs(PD, exist_ok=True)
GC, GL = os.path.join(PD, "_gate.csv"), os.path.join(PD, "_gate.log")
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "dataset", "close4_20261030", "reprobe_rows.csv"), encoding="utf-8-sig")) if not only or r["video"] in only]
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
def one(r):
    name = r["video"]; src = os.path.join(C.vid_dir, name + ".mp4"); out = os.path.join(BD, name + ".mp4")
    if not os.path.exists(out):
        xL, yL, w, h = int(r["xL"]), int(r["yL"]), int(r["w"]), int(r["h"]); ang = float(r["angle_deg"]); mx, my = 20, 48
        vf = f"crop={w+2*mx}:{h+2*my}:{xL-mx}:{yL-my},rotate={-math.radians(ang):.6f}:c=black,crop={w}:{h}:{mx}:{my},scale=2480:120"
        t0 = time.time()
        p = subprocess.run([FFMPEG, "-y", "-v", "error", "-threads", "4", "-i", src, "-vf", vf,
                            "-fps_mode", "passthrough", "-enc_time_base", "1/90000", "-video_track_timescale", "90000",
                            "-c:v", "libx264", "-crf", "15", "-preset", "fast", "-an", out + ".part.mp4"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=0x00004000)
        if p.returncode != 0: return name, f"err ffmpeg rc={p.returncode} {p.stderr[-200:]}"
        os.replace(out + ".part.mp4", out); msg = f"ok {time.time()-t0:.0f}s vf={vf}"
    else: msg = "ok(existing)"
    g = C.band_pts_gate(name, band_path=out, out_dir=PD, gate_csv=GC, gate_log=GL)
    old = os.path.join(C.pts_band_dir, name + ".csv"); new = os.path.join(PD, name + ".csv")
    same = os.path.exists(old) and os.path.exists(new) and md5(old) == md5(new)
    return name, f"{msg} gate={g['status']} band={g.get('n_band')} src={g.get('n_src')} dropped={g.get('dropped')} pts表が既存と同一={same}"
with cf.ThreadPoolExecutor(max_workers=2) as ex:
    for n, s in ex.map(one, rows): C.log(f"RECROP {n} {s}", GL); print("RECROP", n, s, flush=True)
