# -*- coding: utf-8 -*-
"""恒久修正のセルフテスト（2026-09-14）。9/11の帯動画2本（CHEERS・ダーリン）と 9/14 のパススルー試作で、
   ① 帯pts表ゲート（kv_config_20260915.band_pts_gate）が dataset/pts_bands_20260911 の同名ファイルと同一の表を出す
   ② パススルー試作（dataset/_selftest_20260914/CHEERS_passthrough_test.mp4）は落ち0で PASS
   ③ 非動画ファイルを渡すと FAIL になり表を残さない
   ④ g1_and_match_20260915.py が帯ptsのみで G1 を作り、9/14 の帯pts版 G1（fix_g1_bandpts）と時刻が一致する
   ⑤ 帯pts表を1本抜くと G1 がゲートで止まり（rc=2）G1 を書かない
   出力はすべて dataset/_selftest_20260914/ に置く（本番ファイルは触らない）。使い方: python scripts/selftest_bandpts_20260914.py
"""
import os, sys, csv, shutil, subprocess, filecmp
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_config_20260915 import Cfg, PYTHON, ROOT

C = Cfg("20260911"); T = os.path.join(ROOT, "dataset", "_selftest_20260914"); os.makedirs(T, exist_ok=True)
PB = os.path.join(T, "pts_bands_20260911"); GC = os.path.join(PB, "_gate.csv"); GL = os.path.join(PB, "_gate.log")
VIDS = ["CHEERS_20260911_105621", "ダーリン_20260911_103233"]
res = []
def ok(name, cond, detail=""):
    res.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name + (" — " + detail if detail else ""), flush=True)

# ① 帯pts表＝既存の pts_bands_20260911 と同一
for v in VIDS:
    g = C.band_pts_gate(v, band_path=os.path.join(ROOT, "bands_wao", v + ".mp4"), out_dir=PB, gate_csv=GC, gate_log=GL)
    ref = os.path.join(ROOT, "dataset", "pts_bands_20260911", v + ".csv"); new = os.path.join(PB, v + ".csv")
    same = os.path.exists(new) and filecmp.cmp(new, ref, shallow=False)
    ok(f"①帯pts表が既存と同一 {v}", g["status"] in ("PASS", "WARN") and same, f"status={g['status']} band={g['n_band']} src={g['n_src']} dropped={g['dropped']} identical={same}")

# ② パススルー試作＝落ち0で PASS
pt = os.path.join(T, "CHEERS_passthrough_test.mp4")
if os.path.exists(pt):
    g = C.band_pts_gate("CHEERS_20260911_105621", band_path=pt, out_dir=os.path.join(T, "pts_bands_passthrough"), gate_csv=os.path.join(T, "pts_bands_passthrough", "_gate.csv"), gate_log=os.path.join(T, "pts_bands_passthrough", "_gate.log"))
    ok("②パススルー試作は落ち0で PASS", g["status"] == "PASS" and g["dropped"] == 0, f"status={g['status']} band={g['n_band']} src={g['n_src']} dropped={g['dropped']} last={g['last_band']}/{g['last_src']}")
else:
    ok("②パススルー試作", False, "試作ファイルなし")

# ③ 非動画→FAIL・表を残さない
bad_dir = os.path.join(T, "pts_bands_bad")
g = C.band_pts_gate("CHEERS_20260911_105621", band_path=os.path.join(ROOT, "dataset", "pts_bands_20260911", "CHEERS_20260911_105621.csv"), out_dir=bad_dir, gate_csv=os.path.join(bad_dir, "_gate.csv"), gate_log=os.path.join(bad_dir, "_gate.log"))
ok("③非動画で FAIL・表なし", g["status"] == "FAIL" and not os.path.exists(os.path.join(bad_dir, "CHEERS_20260911_105621.csv")), g["reason"])

# ④ G1（帯ptsのみ）＝9/14 帯pts版と一致
RD = os.path.join(T, "results_wao_20260911"); os.makedirs(RD, exist_ok=True)
for v in VIDS:
    os.makedirs(os.path.join(RD, v), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "results_wao_20260911", v, "counter_events.csv"), os.path.join(RD, v, "counter_events.csv"))
env = dict(os.environ, PYTHONIOENCODING="utf-8", KV_SESSION="20260911", KV_RESULTS_DIR=RD, KV_PTS_BAND_DIR=PB, KV_G1_OUT=os.path.join(T, "G1_selftest.csv"),
           KV_PIPE_DIR=os.path.join(T, "pipeline"), KV_BLIND_DIR=os.path.join(T, "blind_none"))
p = subprocess.run([PYTHON, os.path.join(ROOT, "scripts", "g1_and_match_20260915.py"), "20260911"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
print(p.stdout[-800:], p.stderr[-800:])
if p.returncode == 0 and os.path.exists(env["KV_G1_OUT"]):
    G = pd.read_csv(env["KV_G1_OUT"], encoding="utf-8-sig"); R = pd.read_csv(os.path.join(ROOT, "dataset", "G1_events_20260911_bandpts_20260914.csv"), encoding="utf-8-sig")
    R = R[R["video"].isin(VIDS)]
    key = ["video", "counter", "count_value", "wav"]
    M = G.merge(R, on=key, suffixes=("_new", "_ref"))
    dv = (M["video_time_sec_new"] - M["video_time_sec_ref"]).abs().max(); ds = (M["stable_read_time_sec_new"] - M["stable_read_time_sec_ref"]).abs().max()
    da = (M["audio_time_sec_new"] - M["audio_time_sec_ref"]).abs().max()
    ok("④G1 が 9/14 帯pts版と一致", len(M) == len(R) == len(G) and dv <= 0.0015 and ds <= 0.0015 and da <= 0.0015,
       f"rows new={len(G)} ref={len(R)} merged={len(M)} max|Δvideo|={dv:.4f} max|Δstable|={ds:.4f} max|Δaudio|={da:.4f} pts_source={set(G['pts_source'])}")
else:
    ok("④G1 生成", False, f"rc={p.returncode}")

# ⑤ 表を1本抜く→rc=2・G1なし
PB2 = os.path.join(T, "pts_bands_missing_one"); shutil.rmtree(PB2, ignore_errors=True); os.makedirs(PB2)
shutil.copy(os.path.join(PB, VIDS[0] + ".csv"), os.path.join(PB2, VIDS[0] + ".csv"))
env2 = dict(env, KV_PTS_BAND_DIR=PB2, KV_G1_OUT=os.path.join(T, "G1_should_not_exist.csv"))
p2 = subprocess.run([PYTHON, os.path.join(ROOT, "scripts", "g1_and_match_20260915.py"), "20260911"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env2)
ok("⑤帯pts表が欠けると G1 ゲートで停止", p2.returncode == 2 and not os.path.exists(env2["KV_G1_OUT"]), f"rc={p2.returncode} stdout={p2.stdout.strip()[-200:]}")

with open(os.path.join(T, "SELFTEST_RESULT.txt"), "w", encoding="utf-8") as f:
    for n, c, dtl in res: f.write(("PASS " if c else "FAIL ") + n + (" — " + dtl if dtl else "") + "\n")
    f.write(f"ALL {'PASS' if all(c for _, c, _ in res) else 'FAIL'} ({sum(c for _, c, _ in res)}/{len(res)})\n")
print("ALL", "PASS" if all(c for _, c, _ in res) else "FAIL")
