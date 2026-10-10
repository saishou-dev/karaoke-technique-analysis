# -*- coding: utf-8 -*-
"""帯クロップ v2（2026-09-15版・Windowsネイティブ ffmpeg・3並列・.part→rename）＋ 帯pts表の自動生成＋ゲート。
   入力＝dataset/band_probe_v2_<session>.csv（wao_probe_v2_20260915.py の出力）。完成済み（帯動画とpts表の両方あり）はスキップ。
   フィルタ＝crop(余白20/48)→rotate(-angle)→crop→scale=2480:120（9/11と同じ）。
   9/11との違い：
     (1) `-fps_mode passthrough -enc_time_base 1/90000 -video_track_timescale 90000` を追加し、再エンコードでフレームを落とさない
         （9/11の既定＝vfrは 1/59秒に丸めた時刻の衝突で全曲48〜104枚を落としていた。9/14に CHEERS で検証＝11,477枚・ptsは元と一致 4e-5秒）。
     (2) クロップ直後に帯動画自身の moov から pts 表を作る（dataset/pts_bands_<session>/<video>.csv）。G1 はこの表だけを参照する。
     (3) ゲート：pts表が作れない／ffprobe のフレーム数と不一致／元動画より多い／半分未満／最終ptsの差>0.1秒 → FAIL。
         FAIL の帯動画は bands_wao/_gatefail_<session>/ に退避（読取ランナーが拾わない）し、_gate.log に理由を書き、終了コード2。
         落ち枚数>0 は WARN（帯pts表で時刻は正しいので続行・件数は _gate.csv に記録）。
   使い方: python scripts/wincrop_20260915.py [session]   （既定 20260915・環境変数 KV_SESSION でも可）
"""
import os, sys, csv, math, shutil, subprocess, time, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, FFMPEG

C = Cfg()
os.makedirs(C.bands_dir, exist_ok=True); os.makedirs(C.pts_band_dir, exist_ok=True)
rows = C.probe_rows()
if not rows:
    C.log(f"CROP ABORT: プローブ表が無い {C.probe_csv}", C.gate_log); raise SystemExit(3)
C.log(f"CROP start session={C.S} videos={len(rows)}", C.gate_log)

def crop_one(r):
    name = r["video"]; out = os.path.join(C.bands_dir, name + ".mp4"); src = os.path.join(C.vid_dir, name + ".mp4")
    pts_csv = os.path.join(C.pts_band_dir, name + ".csv")
    if os.path.exists(out) and os.path.exists(pts_csv): return name, "skip(done)"
    if not os.path.exists(src): return name, "err 元動画なし " + src
    if not os.path.exists(out):
        xL, yL, w, h = int(r["xL"]), int(r["yL"]), int(r["w"]), int(r["h"]); ang = float(r["angle_deg"]); mx, my = 20, 48
        vf = f"crop={w+2*mx}:{h+2*my}:{xL-mx}:{yL-my},rotate={-math.radians(ang):.6f}:c=black,crop={w}:{h}:{mx}:{my},scale=2480:120"
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
    # 帯pts表＋ゲート（クロップ直後・同じワーカー内）
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
