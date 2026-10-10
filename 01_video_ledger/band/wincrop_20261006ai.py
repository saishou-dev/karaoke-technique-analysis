# -*- coding: utf-8 -*-
"""帯クロップ DAM Ai 版 2026-10-06（wincrop_20260915ai.py の派生・xL<0 のとき左に pad・既存は無変更）。元＝（2026-09-15・wincrop_20260915.py の Ai 派生・Windowsネイティブ ffmpeg・3並列・.part→rename）＋ 帯pts表の自動生成＋ゲート。
   入力＝band_probe_ai_<session>.csv（ai_probe_20260915ai.py の出力）。完成済み（帯動画とpts表の両方あり）はスキップ。
   フィルタ＝crop(余白20/48)→rotate(-angle)→crop→**scale=1500:180**（Ai の帯サイズ＝0628/0807 と同じ。WAO は 2480:120）。
   WAO版と同じ恒久修正：`-fps_mode passthrough -enc_time_base 1/90000 -video_track_timescale 90000`（フレームを落とさない）・
   クロップ直後に帯動画自身の moov から pts 表（<pts_bands_ai_<session>>/<video>.csv）・ゲート FAIL は bands_ai/_gatefail_<session>/ に退避。
   使い方: python scripts/wincrop_20260915ai.py [session]   （既定 20260915・環境変数 KV_SESSION でも可・KV_AI_BASE で出力先差し替え）
"""
import os, sys, csv, math, shutil, subprocess, time, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915ai import CfgAi, FFMPEG, BAND_W, BAND_H

C = CfgAi()
os.makedirs(C.bands_dir, exist_ok=True); os.makedirs(C.pts_band_dir, exist_ok=True)
rows = C.probe_rows()
if not rows:
    C.log(f"CROP ABORT: プローブ表が無い {C.probe_csv}", C.gate_log); raise SystemExit(3)
C.log(f"CROP(ai) start session={C.S} videos={len(rows)} size={BAND_W}x{BAND_H}", C.gate_log)

def crop_one(r):
    name = r["video"]; out = os.path.join(C.bands_dir, name + ".mp4"); src = os.path.join(C.vid_dir, name + ".mp4")
    pts_csv = os.path.join(C.pts_band_dir, name + ".csv")
    if os.path.exists(out) and os.path.exists(pts_csv): return name, "skip(done)"
    if not os.path.exists(src): return name, "err 元動画なし " + src
    if not os.path.exists(out):
        xL, yL, w, h = int(r["xL"]), int(r["yL"]), int(r["w"]), int(r["h"]); ang = float(r["angle_deg"]); mx, my = 20, 48
        pad = max(0, mx - xL)  # 帯の左端が画面外（10/6 Ai・xL が負）なら左に黒を足してから切る（ffmpeg crop は x を 0 に丸めるため）
        pre = f"pad=iw+{pad}:ih:{pad}:0:black," if pad else ""
        vf = pre + f"crop={w+2*mx}:{h+2*my}:{xL-mx+pad}:{yL-my},rotate={-math.radians(ang):.6f}:c=black,crop={w}:{h}:{mx}:{my},scale={BAND_W}:{BAND_H}"
        t0 = time.time()
        p = subprocess.run([FFMPEG, "-y", "-v", "error", "-threads", "4", "-i", src, "-vf", vf,
                            "-fps_mode", "passthrough", "-enc_time_base", "1/90000", "-video_track_timescale", "90000",
                            "-c:v", "libx264", "-crf", "15", "-preset", "fast", "-an", out + ".part.mp4"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            try: os.remove(out + ".part.mp4")
            except OSError: pass
            return name, "err ffmpeg " + p.stderr[-200:].replace("\n", " ")
        os.replace(out + ".part.mp4", out); msg = f"ok {time.time()-t0:.0f}s"
    else:
        msg = "ok(existing)"
    g = C.band_pts_gate(name)
    if g["status"] == "FAIL":
        os.makedirs(C.gatefail_dir, exist_ok=True)
        shutil.move(out, os.path.join(C.gatefail_dir, name + ".mp4"))
        return name, f"GATE_FAIL {g['reason']} → 退避 {C.gatefail_dir}"
    return name, f"{msg} gate={g['status']} band={g['n_band']} src={g['n_src']} dropped={g['dropped']}"

fails = 0
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    for n, s in ex.map(crop_one, rows):
        C.log(f"CROP {n} {s}", C.gate_log)
        if s.startswith("GATE_FAIL") or s.startswith("err"): fails += 1
C.log(f"CROP ALL DONE fails={fails}", C.gate_log)
print("ALL DONE" if fails == 0 else f"DONE WITH FAILURES ({fails}) — see {C.gate_log}", flush=True)
raise SystemExit(0 if fails == 0 else 2)
